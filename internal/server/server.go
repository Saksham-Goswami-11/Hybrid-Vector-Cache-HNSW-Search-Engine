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
	"os"
	"path/filepath"
	"strconv"
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
		select {
		case <-srv.quit:
			return
		default:
		}

		commands, err := protocol.Parse(reader)
		if err != nil {
			if err == io.EOF {
				return // client disconnected
			}
			resp := &protocol.ErrorResponse{Message: err.Error()}
			writer.Write(resp.Serialize())
			writer.Flush()
			return
		}

		for _, cmd := range commands {
			resp := srv.dispatch(client, cmd)
			writer.Write(resp.Serialize())
		}
		writer.Flush()
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

// --- Command Handlers ---

func (srv *Server) handlePing(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) > 0 {
		return &protocol.SimpleString{Value: strings.Join(cmd.Args, " ")}
	}
	return protocol.RespPong
}

func (srv *Server) handleSet(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) < 2 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'SET' command"}
	}

	key := cmd.Args[0]
	value := cmd.Args[1]

	// Check for optional EX <seconds>
	if len(cmd.Args) >= 4 && strings.ToUpper(cmd.Args[2]) == "EX" {
		seconds, err := strconv.Atoi(cmd.Args[3])
		if err != nil || seconds <= 0 {
			return &protocol.ErrorResponse{Message: "invalid expire time in 'SET' command"}
		}
		srv.store.SetWithTTL(key, value, time.Duration(seconds)*time.Second)
	} else {
		srv.store.Set(key, value)
	}

	return protocol.RespOK
}

func (srv *Server) handleGet(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) != 1 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'GET' command"}
	}

	val, ok := srv.store.Get(cmd.Args[0])
	if !ok {
		return protocol.RespNil
	}
	return &protocol.BulkString{Value: val}
}

func (srv *Server) handleDel(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) == 0 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'DEL' command"}
	}
	count := srv.store.Del(cmd.Args...)
	return &protocol.IntegerResponse{Value: count}
}

func (srv *Server) handleExpire(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) != 2 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'EXPIRE' command"}
	}
	seconds, err := strconv.Atoi(cmd.Args[1])
	if err != nil || seconds <= 0 {
		return &protocol.ErrorResponse{Message: "invalid expire time"}
	}
	ok := srv.store.Expire(cmd.Args[0], time.Duration(seconds)*time.Second)
	if ok {
		return &protocol.IntegerResponse{Value: 1}
	}
	return &protocol.IntegerResponse{Value: 0}
}

func (srv *Server) handleTTL(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) != 1 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'TTL' command"}
	}
	ttl := srv.store.TTL(cmd.Args[0])
	return &protocol.IntegerResponse{Value: ttl}
}

func (srv *Server) handleInfo(_ protocol.Command) protocol.Response {
	kvCount := srv.store.KVCount()
	totalVectors := srv.store.TotalVectors()

	// Get per-namespace info
	nsInfos := srv.store.VNSList()

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# Server\r\nversion:1.1.0\r\n# Keyspace\r\nkv_keys:%d\r\n# Vectors\r\nvector_namespaces:%d\r\ntotal_vectors:%d\r\n",
		kvCount, len(nsInfos), totalVectors))

	// Per-namespace stats
	if len(nsInfos) > 0 {
		sb.WriteString("# Vector Namespaces\r\n")
		for _, ns := range nsInfos {
			sb.WriteString(fmt.Sprintf("ns:%s:vectors:%d\r\n", ns.Name, ns.VectorCount))
			sb.WriteString(fmt.Sprintf("ns:%s:approx_memory_bytes:%d\r\n", ns.Name, ns.ApproxMemory))
			sb.WriteString(fmt.Sprintf("ns:%s:ttl:%d\r\n", ns.Name, ns.TTLRemaining))
		}
	}

	return &protocol.BulkString{Value: sb.String()}
}

// --- Vector Command Handlers ---

func (srv *Server) handleVSet(cmd protocol.Command) protocol.Response {
	// VSET <namespace> <id> <dim> <f1> <f2> ... <fN> [META <k> <v> ...] [EX <seconds>]
	if len(cmd.Args) < 4 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VSET' command"}
	}

	namespace := cmd.Args[0]
	id := cmd.Args[1]
	dim, err := strconv.Atoi(cmd.Args[2])
	if err != nil || dim <= 0 {
		return &protocol.ErrorResponse{Message: "invalid dimension"}
	}

	// Parse floats, find META and EX boundaries
	metaIdx := -1
	exIdx := -1
	for i := 3; i < len(cmd.Args); i++ {
		upper := strings.ToUpper(cmd.Args[i])
		if upper == "META" && metaIdx < 0 {
			metaIdx = i
		} else if upper == "EX" && exIdx < 0 {
			exIdx = i
		}
	}

	// Determine float end boundary
	floatEnd := len(cmd.Args)
	if metaIdx >= 0 && metaIdx < floatEnd {
		floatEnd = metaIdx
	}
	if exIdx >= 0 && exIdx < floatEnd {
		floatEnd = exIdx
	}

	floatArgs := cmd.Args[3:floatEnd]

	if len(floatArgs) != dim {
		return &protocol.ErrorResponse{Message: fmt.Sprintf("dimension mismatch: declared %d, got %d floats", dim, len(floatArgs))}
	}

	vec := make([]float32, dim)
	for i, s := range floatArgs {
		f, err := strconv.ParseFloat(s, 32)
		if err != nil {
			return &protocol.ErrorResponse{Message: fmt.Sprintf("invalid float at position %d: %s", i, s)}
		}
		vec[i] = float32(f)
	}

	// Parse metadata pairs
	var meta map[string]string
	if metaIdx >= 0 {
		// META args end at EX or end of args
		metaEnd := len(cmd.Args)
		if exIdx > metaIdx {
			metaEnd = exIdx
		}
		metaArgs := cmd.Args[metaIdx+1 : metaEnd]
		if len(metaArgs)%2 != 0 {
			return &protocol.ErrorResponse{Message: "META requires an even number of arguments (key-value pairs)"}
		}
		meta = make(map[string]string, len(metaArgs)/2)
		for i := 0; i < len(metaArgs); i += 2 {
			meta[metaArgs[i]] = metaArgs[i+1]
		}
	}

	// Parse optional EX <seconds>
	var ttl time.Duration
	if exIdx >= 0 {
		if exIdx+1 >= len(cmd.Args) {
			return &protocol.ErrorResponse{Message: "EX requires a value"}
		}
		seconds, err := strconv.Atoi(cmd.Args[exIdx+1])
		if err != nil || seconds <= 0 {
			return &protocol.ErrorResponse{Message: "invalid expire time in 'VSET' command"}
		}
		ttl = time.Duration(seconds) * time.Second
	}

	if err := srv.store.VSetWithTTL(namespace, id, dim, vec, meta, ttl); err != nil {
		return &protocol.ErrorResponse{Message: err.Error()}
	}

	srv.indexesMu.RLock()
	idx, hasIndex := srv.indexes[namespace]
	srv.indexesMu.RUnlock()

	if hasIndex {
		idx.Insert(id, vec, meta)
	}

	return protocol.RespOK
}

func (srv *Server) handleVGet(cmd protocol.Command) protocol.Response {
	// VGET <namespace> <id>
	if len(cmd.Args) != 2 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VGET' command"}
	}

	entry, ok := srv.store.VGet(cmd.Args[0], cmd.Args[1])
	if !ok {
		return protocol.RespNil
	}

	// Return as array of float strings
	items := make([]protocol.Response, len(entry.Vector))
	for i, f := range entry.Vector {
		items[i] = &protocol.BulkString{Value: fmt.Sprintf("%.6f", f)}
	}
	return &protocol.ArrayResponse{Items: items}
}

func (srv *Server) handleVDel(cmd protocol.Command) protocol.Response {
	// VDEL <namespace> <id>
	if len(cmd.Args) != 2 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VDEL' command"}
	}
	if srv.store.VDel(cmd.Args[0], cmd.Args[1]) {
		return &protocol.IntegerResponse{Value: 1}
	}
	return &protocol.IntegerResponse{Value: 0}
}

func (srv *Server) handleVCount(cmd protocol.Command) protocol.Response {
	// VCOUNT <namespace>
	if len(cmd.Args) != 1 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VCOUNT' command"}
	}
	count := srv.store.VCount(cmd.Args[0])
	return &protocol.IntegerResponse{Value: count}
}

func (srv *Server) handleVSimilarity(cmd protocol.Command) protocol.Response {
	// VSIMILARITY <namespace> <dim> <f1> ... <fN> TOP <k>
	if len(cmd.Args) < 5 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VSIMILARITY' command"}
	}

	namespace := cmd.Args[0]
	dim, err := strconv.Atoi(cmd.Args[1])
	if err != nil || dim <= 0 {
		return &protocol.ErrorResponse{Message: "invalid dimension"}
	}

	// Find TOP keyword
	topIdx := -1
	for i := 2; i < len(cmd.Args); i++ {
		if strings.ToUpper(cmd.Args[i]) == "TOP" {
			topIdx = i
			break
		}
	}
	if topIdx < 0 || topIdx+1 >= len(cmd.Args) {
		return &protocol.ErrorResponse{Message: "missing TOP <k> argument"}
	}

	k, err := strconv.Atoi(cmd.Args[topIdx+1])
	if err != nil || k <= 0 || k > 1000 {
		return &protocol.ErrorResponse{Message: "TOP k must be between 1 and 1000"}
	}

	floatArgs := cmd.Args[2:topIdx]
	if len(floatArgs) != dim {
		return &protocol.ErrorResponse{Message: fmt.Sprintf("dimension mismatch: declared %d, got %d floats", dim, len(floatArgs))}
	}

	query := make([]float32, dim)
	for i, s := range floatArgs {
		f, err := strconv.ParseFloat(s, 32)
		if err != nil {
			return &protocol.ErrorResponse{Message: fmt.Sprintf("invalid float at position %d: %s", i, s)}
		}
		query[i] = float32(f)
	}

	// Snapshot vectors (lock-free after this)
	entries := srv.store.VSnapshot(namespace)
	if len(entries) == 0 {
		return &protocol.ArrayResponse{Items: nil}
	}

	// Check if HNSW index exists
	srv.indexesMu.RLock()
	idx, hasIndex := srv.indexes[namespace]
	srv.indexesMu.RUnlock()

	var results []similarity.SimilarityResult

	if hasIndex {
		hnswResults, err := idx.Search(query, k, 0)
		if err != nil {
			// fallback to brute force if empty index or error
			results = srv.engine.TopK(query, entries, k)
		} else {
			for _, r := range hnswResults {
				results = append(results, similarity.SimilarityResult{
					ID:       r.ID,
					Score:    r.Score,
					Metadata: r.Metadata,
				})
			}
		}
	} else {
		// Run similarity search via brute force
		results = srv.engine.TopK(query, entries, k)
	}

	// Format response: array of [id, score, metadata_pairs]
	var items []protocol.Response
	for _, r := range results {
		items = append(items, &protocol.BulkString{Value: r.ID})
		items = append(items, &protocol.SimpleString{Value: fmt.Sprintf("%.4f", r.Score)})

		// Metadata as sub-array
		var metaItems []protocol.Response
		for mk, mv := range r.Metadata {
			metaItems = append(metaItems, &protocol.BulkString{Value: mk})
			metaItems = append(metaItems, &protocol.BulkString{Value: mv})
		}
		items = append(items, &protocol.ArrayResponse{Items: metaItems})
	}

	return &protocol.ArrayResponse{Items: items}
}

func (srv *Server) handleVIndex(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) < 2 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VINDEX' command"}
	}

	subCmd := strings.ToUpper(cmd.Args[0])
	namespace := cmd.Args[1]

	switch subCmd {
	case "CREATE":
		m := 16
		efC := 200
		efS := 100

		for i := 2; i < len(cmd.Args); i += 2 {
			if i+1 >= len(cmd.Args) {
				return &protocol.ErrorResponse{Message: "missing value for VINDEX CREATE parameter"}
			}
			param := strings.ToUpper(cmd.Args[i])
			val, err := strconv.Atoi(cmd.Args[i+1])
			if err != nil || val <= 0 {
				return &protocol.ErrorResponse{Message: "invalid integer for VINDEX CREATE parameter"}
			}
			switch param {
			case "M":
				m = val
			case "EF_CONSTRUCTION":
				efC = val
			case "EF_SEARCH":
				efS = val
			default:
				return &protocol.ErrorResponse{Message: "unknown VINDEX CREATE parameter"}
			}
		}

		idx := hnsw.NewIndex(m, efC)
		idx.SetEF(efS)

		srv.indexesMu.Lock()
		if _, exists := srv.indexes[namespace]; exists {
			srv.indexesMu.Unlock()
			return &protocol.ErrorResponse{Message: "index already exists for namespace"}
		}
		srv.indexes[namespace] = idx
		srv.indexesMu.Unlock()

		// Bulk load existing vectors from namespace into the index
		entries := srv.store.VSnapshot(namespace)
		for _, entry := range entries {
			idx.Insert(entry.ID, entry.Vector, entry.Metadata)
		}

		return protocol.RespOK

	case "DROP":
		srv.indexesMu.Lock()
		if _, exists := srv.indexes[namespace]; !exists {
			srv.indexesMu.Unlock()
			return &protocol.ErrorResponse{Message: "no index for namespace"}
		}
		delete(srv.indexes, namespace)
		srv.indexesMu.Unlock()
		return protocol.RespOK

	case "INFO":
		srv.indexesMu.RLock()
		idx, exists := srv.indexes[namespace]
		srv.indexesMu.RUnlock()

		if !exists {
			return &protocol.ErrorResponse{Message: "no index for namespace"}
		}

		info := fmt.Sprintf("len:%d ef:%d", idx.Len(), idx.GetEF())
		return &protocol.SimpleString{Value: info}

	case "SET_EF":
		if len(cmd.Args) != 3 {
			return &protocol.ErrorResponse{Message: "wrong number of arguments for VINDEX SET_EF"}
		}
		ef, err := strconv.Atoi(cmd.Args[2])
		if err != nil || ef <= 0 {
			return &protocol.ErrorResponse{Message: "invalid EF value"}
		}

		srv.indexesMu.RLock()
		idx, exists := srv.indexes[namespace]
		srv.indexesMu.RUnlock()

		if !exists {
			return &protocol.ErrorResponse{Message: "no index for namespace"}
		}

		idx.SetEF(ef)
		return protocol.RespOK

	default:
		return &protocol.ErrorResponse{Message: "unknown VINDEX subcommand"}
	}
}

// SaveSnapshots writes all HNSW indexes to disk.
func (srv *Server) SaveSnapshots(dir string) error {
	srv.indexesMu.RLock()
	defer srv.indexesMu.RUnlock()

	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	for ns, idx := range srv.indexes {
		path := filepath.Join(dir, ns+".hnsw.snap")
		f, err := os.Create(path)
		if err != nil {
			log.Printf("failed to create snapshot file for namespace %s: %v", ns, err)
			continue
		}
		if err := idx.Snapshot(f); err != nil {
			log.Printf("failed to write snapshot for namespace %s: %v", ns, err)
			f.Close()
			continue
		}
		f.Close()
	}
	return nil
}

// LoadSnapshots reads all HNSW indexes from disk.
func (srv *Server) LoadSnapshots(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	srv.indexesMu.Lock()
	defer srv.indexesMu.Unlock()

	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".hnsw.snap") {
			ns := strings.TrimSuffix(entry.Name(), ".hnsw.snap")
			path := filepath.Join(dir, entry.Name())
			
			f, err := os.Open(path)
			if err != nil {
				log.Printf("failed to open snapshot %s: %v", path, err)
				continue
			}
			
			idx, err := hnsw.LoadSnapshot(f)
			f.Close()
			
			if err != nil {
				log.Printf("failed to load snapshot %s: %v", path, err)
				continue
			}
			
			srv.indexes[ns] = idx
			log.Printf("loaded HNSW snapshot for namespace %s (nodes: %d)", ns, idx.Len())
		}
	}
	return nil
}

// --- Multi-Agent / AutoGen Architecture Command Handlers ---
//
// The following handlers (VMSET, VEXPIRE, VNS DROP/LIST) were added to the Nearby
// wire protocol to support Microsoft AutoGen v0.4 integration. They are invoked by
// the Python client NearbyVectorStore (nearby_memory.py) via raw TCP socket.
//
// Wire protocol reference:
//   VMSET <ns> <dim> <count> <id1> <f1..fN> [META k v ...] ...   → batch ingest
//   VEXPIRE <ns> <seconds>                                       → namespace TTL
//   VNS DROP <ns>                                                 → teardown namespace
//   VNS LIST                                                      → enumerate namespaces

// handleVMSet processes the VMSET command for batch vector ingestion.
// AutoGen compatibility: NearbyVectorStore.ingest_batch() sends VMSET with automatic
// chunking (25 vectors per line) to stay within Nearby's 64KB TCP line length limit.
func (srv *Server) handleVMSet(cmd protocol.Command) protocol.Response {
	// VMSET <namespace> <dim> <count> <id1> <f1..fN> [META k v ...] <id2> <f1..fN> [META k v ...] ...
	if len(cmd.Args) < 4 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VMSET' command"}
	}

	namespace := cmd.Args[0]
	dim, err := strconv.Atoi(cmd.Args[1])
	if err != nil || dim <= 0 {
		return &protocol.ErrorResponse{Message: "invalid dimension"}
	}
	expectCount, err := strconv.Atoi(cmd.Args[2])
	if err != nil || expectCount <= 0 {
		return &protocol.ErrorResponse{Message: "invalid count"}
	}

	// Parse each entry: id followed by dim floats, optionally followed by META k v pairs
	entries := make([]store.VMSetEntry, 0, expectCount)
	i := 3
	for len(entries) < expectCount && i < len(cmd.Args) {
		id := cmd.Args[i]
		i++

		// Read dim floats
		if i+dim > len(cmd.Args) {
			return &protocol.ErrorResponse{Message: fmt.Sprintf("entry '%s': not enough floats (need %d)", id, dim)}
		}

		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			f, err := strconv.ParseFloat(cmd.Args[i], 32)
			if err != nil {
				return &protocol.ErrorResponse{Message: fmt.Sprintf("entry '%s': invalid float at position %d: %s", id, j, cmd.Args[i])}
			}
			vec[j] = float32(f)
			i++
		}

		// Check for optional META
		var meta map[string]string
		if i < len(cmd.Args) && strings.ToUpper(cmd.Args[i]) == "META" {
			i++ // skip META keyword
			var metaPairs []string
			for i+1 < len(cmd.Args) {
				// If there are more entries remaining, check if the current position
				// looks like the start of the next entry (an ID followed by dim floats).
				if len(entries)+1 < expectCount && i+1+dim <= len(cmd.Args) {
					if _, err := strconv.ParseFloat(cmd.Args[i+1], 64); err == nil {
						// cmd.Args[i] is the next entry's ID, cmd.Args[i+1] is its first float
						break
					}
				}
				metaPairs = append(metaPairs, cmd.Args[i], cmd.Args[i+1])
				i += 2
			}
			if len(metaPairs)%2 != 0 {
				return &protocol.ErrorResponse{Message: fmt.Sprintf("entry '%s': META requires even number of arguments", id)}
			}
			if len(metaPairs) > 0 {
				meta = make(map[string]string, len(metaPairs)/2)
				for j := 0; j < len(metaPairs); j += 2 {
					meta[metaPairs[j]] = metaPairs[j+1]
				}
			}
		}

		entries = append(entries, store.VMSetEntry{
			ID:       id,
			Vector:   vec,
			Metadata: meta,
		})
	}

	if len(entries) != expectCount {
		return &protocol.ErrorResponse{Message: fmt.Sprintf("expected %d entries, got %d", expectCount, len(entries))}
	}

	stored, err := srv.store.VMSet(namespace, dim, entries)
	if err != nil {
		return &protocol.ErrorResponse{Message: err.Error()}
	}

	// Update HNSW index if it exists
	srv.indexesMu.RLock()
	idx, hasIndex := srv.indexes[namespace]
	srv.indexesMu.RUnlock()

	if hasIndex {
		for _, e := range entries {
			idx.Insert(e.ID, e.Vector, e.Metadata)
		}
	}

	return &protocol.IntegerResponse{Value: stored}
}

// handleVExpire processes the VEXPIRE command for namespace-level TTL.
// AutoGen compatibility: called by NearbyVectorStore.set_namespace_ttl() to set
// a run-level expiration on all vectors in an agent swarm's namespace.
func (srv *Server) handleVExpire(cmd protocol.Command) protocol.Response {
	// VEXPIRE <namespace> <seconds>
	if len(cmd.Args) != 2 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VEXPIRE' command"}
	}

	namespace := cmd.Args[0]
	seconds, err := strconv.Atoi(cmd.Args[1])
	if err != nil || seconds <= 0 {
		return &protocol.ErrorResponse{Message: "invalid expire time"}
	}

	ok := srv.store.VExpireNamespace(namespace, time.Duration(seconds)*time.Second)
	if ok {
		return &protocol.IntegerResponse{Value: 1}
	}
	return &protocol.IntegerResponse{Value: 0}
}

// handleVNS processes VNS subcommands (DROP, LIST) for namespace lifecycle.
// AutoGen compatibility: VNS DROP is called by NearbyVectorStore.drop_namespace()
// at end-of-run to instantly reclaim memory. VNS LIST is used for monitoring.
func (srv *Server) handleVNS(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) < 1 {
		return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VNS' command"}
	}

	subCmd := strings.ToUpper(cmd.Args[0])

	switch subCmd {
	case "DROP":
		if len(cmd.Args) != 2 {
			return &protocol.ErrorResponse{Message: "wrong number of arguments for 'VNS DROP' command"}
		}
		namespace := cmd.Args[1]

		// Drop from store
		ok := srv.store.VNSDrop(namespace)

		// Also clean up HNSW index if it exists
		srv.indexesMu.Lock()
		if _, exists := srv.indexes[namespace]; exists {
			delete(srv.indexes, namespace)
		}
		srv.indexesMu.Unlock()

		if ok {
			return &protocol.IntegerResponse{Value: 1}
		}
		return &protocol.IntegerResponse{Value: 0}

	case "LIST":
		infos := srv.store.VNSList()
		if len(infos) == 0 {
			return &protocol.ArrayResponse{Items: nil}
		}

		// Return as array of [name, vectorCount, approxMemory, ttlRemaining]
		var items []protocol.Response
		for _, info := range infos {
			items = append(items,
				&protocol.BulkString{Value: info.Name},
				&protocol.IntegerResponse{Value: info.VectorCount},
				&protocol.IntegerResponse{Value: int(info.ApproxMemory)},
				&protocol.IntegerResponse{Value: info.TTLRemaining},
			)
		}
		return &protocol.ArrayResponse{Items: items}

	default:
		return &protocol.ErrorResponse{Message: fmt.Sprintf("unknown VNS subcommand '%s'", subCmd)}
	}
}
