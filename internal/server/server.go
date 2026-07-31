package server

import (
	"bufio"
	"crypto/sha256"
	"crypto/subtle"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/hnsw"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/persist"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/protocol"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/similarity"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
)

// Client represents an active client connection.
type Client struct {
	Conn          net.Conn
	Authenticated bool
}

// Server is the TCP server that accepts connections and dispatches commands.
type Server struct {
	store       *store.Store
	engine      *similarity.Engine
	aof         *persist.AOFWriter
	listener    net.Listener
	addr        string
	RequirePass string

	// HNSW indexes per namespace
	indexes   map[string]*hnsw.Index
	indexesMu sync.RWMutex

	// TLS configuration
	tlsEnabled bool
	certFile   string
	keyFile    string

	// Shutdown coordination
	quit     chan struct{}
	ready    chan struct{} // closed once the listener is bound
	wg       sync.WaitGroup
	mu       sync.Mutex
	shutdown bool
}

// New creates a new Server backed by the given store.
func New(addr string, password string, tlsEnabled bool, certFile string, keyFile string, s *store.Store) *Server {
	return &Server{
		store:       s,
		engine:      similarity.NewEngine(0), // 0 = GOMAXPROCS
		addr:        addr,
		RequirePass: password,
		tlsEnabled:  tlsEnabled,
		certFile:    certFile,
		keyFile:     keyFile,
		indexes:     make(map[string]*hnsw.Index),
		quit:        make(chan struct{}),
		ready:       make(chan struct{}),
	}
}

// ListenAndServe starts the TCP listener and enters the accept loop.
// It blocks until the server is shut down.
func (srv *Server) ListenAndServe() error {
	var ln net.Listener
	var err error

	if srv.tlsEnabled {
		cert, err := tls.LoadX509KeyPair(srv.certFile, srv.keyFile)
		if err != nil {
			return fmt.Errorf("failed to load TLS keys: %w", err)
		}
		config := &tls.Config{Certificates: []tls.Certificate{cert}}
		ln, err = tls.Listen("tcp", srv.addr, config)
		if err != nil {
			return err
		}
		log.Printf("server listening on %s (TLS Enabled)", srv.addr)
	} else {
		ln, err = net.Listen("tcp", srv.addr)
		if err != nil {
			return err
		}
		log.Printf("server listening on %s", srv.addr)
	}

	srv.mu.Lock()
	srv.listener = ln
	srv.mu.Unlock()
	close(srv.ready) // signal that the listener is bound
	log.Printf("nearby listening on %s", ln.Addr())

	// Start background expiry sweep
	srv.wg.Add(1)
	go srv.expirySweep()

	for {
		conn, err := ln.Accept()
		if err != nil {
			select {
			case <-srv.quit:
				return nil // graceful shutdown
			default:
				log.Printf("accept error: %v", err)
				continue
			}
		}
		srv.wg.Add(1)
		go srv.handleConnection(conn)
	}
}

// Shutdown gracefully stops the server.
func (srv *Server) Shutdown() {
	srv.mu.Lock()
	if srv.shutdown {
		srv.mu.Unlock()
		return
	}
	srv.shutdown = true
	srv.mu.Unlock()

	close(srv.quit)
	if srv.listener != nil {
		srv.listener.Close()
	}
	srv.wg.Wait()
	log.Println("nearby shut down gracefully")
}

// Addr returns the listener's address (useful for tests with port 0).
// Safe to call after WaitReady() returns.
func (srv *Server) Addr() string {
	srv.mu.Lock()
	ln := srv.listener
	srv.mu.Unlock()
	if ln != nil {
		return ln.Addr().String()
	}
	return srv.addr
}

// WaitReady blocks until the server's listener is bound.
func (srv *Server) WaitReady() {
	<-srv.ready
}

// SetAOF configures the server to write mutative commands to the given AOF writer.
func (srv *Server) SetAOF(aof *persist.AOFWriter) {
	srv.mu.Lock()
	defer srv.mu.Unlock()
	srv.aof = aof
}

// expirySweep runs a background loop that sweeps expired keys every 100ms.
func (srv *Server) expirySweep() {
	defer srv.wg.Done()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-srv.quit:
			return
		case <-ticker.C:
			srv.store.SweepExpired()
		}
	}
}

// handleConnection processes commands from a single client connection.
func (srv *Server) handleConnection(conn net.Conn) {
	defer srv.wg.Done()
	defer conn.Close()

	// Recover from handler panics
	defer func() {
		if r := recover(); r != nil {
			log.Printf("recovered panic in connection handler: %v", r)
		}
	}()

	reader := bufio.NewReader(conn)
	writer := bufio.NewWriter(conn)

	client := &Client{
		Conn:          conn,
		Authenticated: srv.RequirePass == "", // default to true if no password required
	}

	for {
		commands, err := protocol.Parse(reader)
		if err != nil {
			if err == io.EOF {
				return // client disconnected cleanly
			}
			select {
			case <-srv.quit:
				return // server shutting down
			default:
			}
			resp := &protocol.ErrorResponse{Message: err.Error()}
			writer.Write(resp.Serialize())
			writer.Flush()
			return
		}

		// Execute all parsed commands in batch
		for _, cmd := range commands {
			resp := srv.dispatch(client, cmd)
			writer.Write(resp.Serialize())
		}
		writer.Flush()

		// Graceful Draining: Check quit channel after fully dispatching and flushing current request
		select {
		case <-srv.quit:
			return
		default:
		}
	}
}

// dispatch routes a command to the appropriate handler, logs it to AOF if necessary, and returns a response.
func (srv *Server) dispatch(client *Client, cmd protocol.Command) protocol.Response {
	resp := srv.ExecuteCommand(client, cmd)

	srv.mu.Lock()
	aof := srv.aof
	srv.mu.Unlock()

	// Write mutative commands to AOF
	if aof != nil {
		if _, isErr := resp.(*protocol.ErrorResponse); !isErr {
			if isMutative(cmd.Name) {
				if err := aof.Write(cmd.String()); err != nil {
					log.Printf("aof write error: %v", err)
				}
			}
		}
	}

	return resp
}

func isMutative(name string) bool {
	switch name {
	case "SET", "DEL", "EXPIRE", "VSET", "VDEL", "VINDEX", "VMSET", "VEXPIRE", "VNS":
		return true
	}
	return false
}

// ExecuteCommand runs the actual command handler without logging to AOF.
func (srv *Server) ExecuteCommand(client *Client, cmd protocol.Command) protocol.Response {
	cmdName := strings.ToUpper(cmd.Name) // e.g., "AUTH", "SET"

	// Enforce authentication
	if srv.RequirePass != "" && !client.Authenticated {
		if cmdName != "AUTH" {
			// Reject unauthenticated commands
			return &protocol.ErrorResponse{Message: "ERR NOAUTH Authentication required"}
		}
	}

	switch cmdName {
	case "AUTH":
		if len(cmd.Args) != 1 {
			return &protocol.ErrorResponse{Message: "ERR wrong number of arguments for 'AUTH' command"}
		}
		if srv.RequirePass == "" {
			return &protocol.ErrorResponse{Message: "ERR Client sent AUTH, but no password is set"}
		}

		providedHash := sha256.Sum256([]byte(cmd.Args[0]))
		requiredHash := sha256.Sum256([]byte(srv.RequirePass))

		if subtle.ConstantTimeCompare(providedHash[:], requiredHash[:]) == 1 {
			client.Authenticated = true
			return protocol.RespOK
		}
		return &protocol.ErrorResponse{Message: "ERR invalid password"}
	case "PING":
		return srv.handlePing(cmd)
	case "SET":
		return srv.handleSet(cmd)
	case "GET":
		return srv.handleGet(cmd)
	case "DEL":
		return srv.handleDel(cmd)
	case "EXPIRE":
		return srv.handleExpire(cmd)
	case "TTL":
		return srv.handleTTL(cmd)
	case "INFO":
		return srv.handleInfo(cmd)
	case "VSET":
		return srv.handleVSet(cmd)
	case "VGET":
		return srv.handleVGet(cmd)
	case "VDEL":
		return srv.handleVDel(cmd)
	case "VCOUNT":
		return srv.handleVCount(cmd)
	case "VSIMILARITY":
		return srv.handleVSimilarity(cmd)
	case "VINDEX":
		return srv.handleVIndex(cmd)
	case "VMSET":
		return srv.handleVMSet(cmd)
	case "VEXPIRE":
		return srv.handleVExpire(cmd)
	case "VNS":
		return srv.handleVNS(cmd)
	default:
		return &protocol.ErrorResponse{Message: fmt.Sprintf("unknown command '%s'", cmd.Name)}
	}
}
