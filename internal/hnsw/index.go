package hnsw

import (
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
)

// Index is the HNSW graph index.
// It is constructed with NewIndex and supports concurrent Insert and Search
// once Phase 5 concurrency is wired in.
type Index struct {
	// Configuration — set at construction, immutable after build
	M              int     // max connections per node in layers 1+ (default 16)
	Mmax           int     // alias for M (upper layers cap)
	Mmax0          int     // max connections at layer 0 = 2*M
	EfConstruction int     // build-time candidate list size (default 200)
	Ef             int     // query-time candidate list size (default 100, adjustable)
	ML             float64 // level multiplier = 1 / ln(M)
	MaxLevel       int     // safety cap on layer assignment (default 16)

	// Graph state — protected by mu
	nodes           []*Node          // all nodes, indexed by internal ID
	idMap           map[string]uint32 // string key → internal ID
	reverseMap      []string         // internal ID → string key
	entryPoint      uint32           // ID of the entry point node
	currentMaxLayer int              // highest layer currently in use
	mu              sync.RWMutex

	// Stats
	insertCount atomic.Int64
	searchCount atomic.Int64

	// RNG for layer assignment — seeded per-index for reproducibility in tests
	rng *rand.Rand
	rmu sync.Mutex // protects rng
}

// NewIndex creates a new HNSW index with the given parameters.
// M is the max connections per node (default 16).
// efConstruction is the build-time search depth (default 200).
func NewIndex(M, efConstruction int) *Index {
	if M <= 0 {
		M = 16
	}
	if efConstruction <= 0 {
		efConstruction = 200
	}

	return &Index{
		M:              M,
		Mmax:           M,
		Mmax0:          2 * M,
		EfConstruction: efConstruction,
		Ef:             100, // default efSearch
		ML:             1.0 / math.Log(float64(M)),
		MaxLevel:       16,
		idMap:          make(map[string]uint32),
		rng:            rand.New(rand.NewSource(rand.Int63())),
	}
}

// Len returns the number of nodes in the index.
func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.nodes)
}

// SetEF adjusts the query-time search depth.
// Higher ef = better recall, higher latency.
// ef must be >= 1.
func (idx *Index) SetEF(ef int) {
	if ef < 1 {
		return
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.Ef = ef
}

// GetEF returns the current query-time search depth.
func (idx *Index) GetEF() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.Ef
}

// randomLevel samples a layer from an exponentially decaying distribution.
// P(level=0) ≈ 69.8%, P(level=1) ≈ 21.3%, P(level=2) ≈ 6.5%, ...
// From the Malkov & Yashunin paper: level = floor(-ln(uniform) * mL)
func (idx *Index) randomLevel() int {
	idx.rmu.Lock()
	r := idx.rng.Float64()
	idx.rmu.Unlock()

	// Avoid log(0) — extremely unlikely but must be safe
	if r == 0 {
		r = 1e-18
	}

	level := int(-math.Log(r) * idx.ML)
	if level > idx.MaxLevel {
		level = idx.MaxLevel
	}
	return level
}

// Stats returns insertion and search counts.
func (idx *Index) Stats() (inserts, searches int64) {
	return idx.insertCount.Load(), idx.searchCount.Load()
}

// EntryPointKey returns the string key of the current entry point node.
// Returns empty string if the index is empty.
func (idx *Index) EntryPointKey() string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if len(idx.nodes) == 0 {
		return ""
	}
	return idx.reverseMap[idx.entryPoint]
}

// CurrentMaxLayer returns the highest layer currently in use.
func (idx *Index) CurrentMaxLayer() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.currentMaxLayer
}

// TotalLinks returns the total number of links across all nodes and layers.
func (idx *Index) TotalLinks() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	total := 0
	for _, node := range idx.nodes {
		total += node.LinkCount()
	}
	return total
}

// MemoryBytes estimates the memory usage of the index in bytes.
// Accounts for vector data + graph links + metadata overhead.
func (idx *Index) MemoryBytes() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var total int64
	for _, node := range idx.nodes {
		// Vector data: dim * 4 bytes
		total += int64(len(node.Vector)) * 4
		// Links: each layer's links = numLinks * 4 bytes (uint32)
		for _, layerLinks := range node.Links {
			total += int64(len(layerLinks)) * 4
		}
		// Rough metadata overhead
		for k, v := range node.Metadata {
			total += int64(len(k) + len(v))
		}
		// Struct overhead estimate
		total += 100
	}
	return total
}

// Delete marks a node as tombstoned in the index and removes it from idMap.
// Returns true if the key existed and was marked deleted.
func (idx *Index) Delete(key string) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	id, exists := idx.idMap[key]
	if !exists {
		return false
	}

	delete(idx.idMap, key)
	node := idx.nodes[id]
	node.SetDeleted(true)

	// Elect a new non-deleted entry point if the deleted node was the entry point
	if id == idx.entryPoint {
		foundNewEP := false
		for newID, newKey := range idx.reverseMap {
			if newKey != "" && !idx.nodes[newID].IsDeleted() {
				idx.entryPoint = uint32(newID)
				foundNewEP = true
				break
			}
		}
		if !foundNewEP {
			idx.entryPoint = 0
		}
	}

	return true
}
