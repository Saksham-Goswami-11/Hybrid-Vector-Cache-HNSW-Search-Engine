package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/persist"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/protocol"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/server"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
)

func main() {
	port := flag.Int("port", 6379, "TCP port to listen on")
	aofPath := flag.String("aof", "", "Path to AOF file for persistence")
	password := flag.String("password", "", "Password for client authentication")
	tlsEnabled := flag.Bool("tls", false, "Enable TLS encryption")
	certFile := flag.String("cert", "cert.pem", "Path to TLS certificate")
	keyFile := flag.String("key", "key.pem", "Path to TLS private key")
	snapshotsDir := flag.String("snapshots", "data/snapshots", "Directory to store HNSW snapshots")
	flag.Parse()

	// Override from environment variables
	if envPort := os.Getenv("NEARBY_PORT"); envPort != "" {
		fmt.Sscanf(envPort, "%d", port)
	}
	if envAOF := os.Getenv("NEARBY_AOF_PATH"); envAOF != "" {
		*aofPath = envAOF
	}
	if envPass := os.Getenv("NEARBY_PASSWORD"); envPass != "" {
		*password = envPass
	}

	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	log.Println("initializing nearby...")

	s := store.New()
	addr := fmt.Sprintf(":%d", *port)
	srv := server.New(addr, *password, *tlsEnabled, *certFile, *keyFile, s)

	// Replay AOF if specified
	var aofWriter *persist.AOFWriter
	if *aofPath != "" {
		log.Printf("loading AOF from %s...", *aofPath)
		lines, err := persist.ReplayAOF(*aofPath)
		if err != nil {
			log.Fatalf("failed to replay AOF: %v", err)
		}

		for i, line := range lines {
			cmds, parseErr := protocol.Parse(bufio.NewReader(strings.NewReader(line + "\n")))
			if parseErr != nil {
				log.Printf("aof replay: warning: skipping invalid command at line %d: %v", i+1, parseErr)
				continue
			}
			// Create a dummy authenticated client for AOF replay
			dummyClient := &server.Client{Authenticated: true}
			for _, cmd := range cmds {
				srv.ExecuteCommand(dummyClient, cmd)
			}
		}
		log.Printf("replayed %d commands from AOF", len(lines))

		aofWriter, err = persist.NewAOFWriter(*aofPath)
		if err != nil {
			log.Fatalf("failed to initialize AOF writer: %v", err)
		}
		srv.SetAOF(aofWriter)
	}

	// Load HNSW Snapshots
	log.Printf("loading HNSW snapshots from %s...", *snapshotsDir)
	if err := srv.LoadSnapshots(*snapshotsDir); err != nil {
		log.Printf("warning: failed to load snapshots: %v", err)
	}

	// Graceful shutdown on SIGTERM / SIGINT
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		<-sigCh
		log.Println("received shutdown signal")
		srv.Shutdown()
		if aofWriter != nil {
			aofWriter.Sync()
			aofWriter.Close()
		}
		
		log.Printf("saving HNSW snapshots to %s...", *snapshotsDir)
		if err := srv.SaveSnapshots(*snapshotsDir); err != nil {
			log.Printf("failed to save snapshots: %v", err)
		} else {
			log.Println("snapshots saved successfully")
		}
	}()

	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
