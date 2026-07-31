package server

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/protocol"
)

func (srv *Server) handlePing(cmd protocol.Command) protocol.Response {
	if len(cmd.Args) > 0 {
		return &protocol.SimpleString{Value: strings.Join(cmd.Args, " ")}
	}
	return protocol.RespPong
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

// handleVExpire processes the VEXPIRE command for namespace-level TTL.
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
