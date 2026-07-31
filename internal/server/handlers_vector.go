package server

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/protocol"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/similarity"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
)

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
	namespace := cmd.Args[0]
	id := cmd.Args[1]

	deleted := srv.store.VDel(namespace, id)

	srv.indexesMu.RLock()
	idx, hasIndex := srv.indexes[namespace]
	srv.indexesMu.RUnlock()

	if hasIndex {
		idx.Delete(id)
	}

	if deleted {
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

// handleVMSet processes the VMSET command for batch vector ingestion.
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
				if len(entries)+1 < expectCount && i+1+dim <= len(cmd.Args) {
					if _, err := strconv.ParseFloat(cmd.Args[i+1], 64); err == nil {
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
