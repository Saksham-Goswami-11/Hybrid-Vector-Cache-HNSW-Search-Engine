package server

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/hnsw"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/protocol"
)

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
