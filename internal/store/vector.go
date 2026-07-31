// Package store provides the core in-memory data structures for the Nearby vector cache.
//
// This file implements the vector namespace operations: storage, retrieval,
// batch ingestion, TTL-based expiry, and namespace lifecycle management.
//
// ## Microsoft AutoGen v0.4 Compatibility
//
// The following features were added to support ephemeral multi-agent swarm workloads
// as described in the Nearby × AutoGen integration (see AUTOGEN_NEARBY_INTEGRATION_REPORT.md):
//
//   - Per-vector and per-namespace TTL (ExpiresAt, HasTTL, VExpireNamespace)
//     Allows agent runs to set automatic memory reclamation on intermediate embeddings.
//
//   - Batch vector ingestion (VMSet / VMSetEntry)
//     Maps to AutoGen's NearbyVectorMemory.add_batch() for high-throughput ingestion.
//
//   - Namespace lifecycle (VNSDrop, VNSList)
//     Enables instant teardown of an entire swarm run's memory without per-key deletion.
//
//   - Lazy expiry on read (VGet) and background sweep (SweepExpired)
//     Ensures expired agent embeddings are reclaimed without blocking hot-path operations.
package store

import (
	"fmt"
	"sync"
	"time"
)

// VectorEntry holds a single vector with its ID, metadata, and optional TTL.
type VectorEntry struct {
	ID        string
	Vector    []float32
	Metadata  map[string]string
	ExpiresAt time.Time // AutoGen compatibility: per-vector expiration timestamp
	HasTTL    bool      // AutoGen compatibility: true if this vector has an active TTL
}

// isExpired reports whether this vector entry has exceeded its TTL.
func (e *VectorEntry) isExpired() bool {
	return e.HasTTL && time.Now().After(e.ExpiresAt)
}

// VectorNamespace is a named partition of vectors with optional default TTL.
type VectorNamespace struct {
	entries           map[string]*VectorEntry
	mu                sync.RWMutex
	DefaultTTL        time.Duration
	HasDefaultTTL     bool
	NamespaceTTLSetAt time.Time
}

// newVectorNamespace creates an empty namespace.
func newVectorNamespace() *VectorNamespace {
	return &VectorNamespace{
		entries: make(map[string]*VectorEntry),
	}
}

// VNamespaceInfo holds summary information about a vector namespace.
type VNamespaceInfo struct {
	Name         string
	VectorCount  int
	ApproxMemory int64 // bytes
	HasTTL       bool
	TTLRemaining int // seconds, -1 if no TTL
}

// VMSetEntry represents a single entry in a batch VMSET operation.
type VMSetEntry struct {
	ID       string
	Vector   []float32
	Metadata map[string]string
}

// getNamespace returns an existing VectorNamespace or (nil, false).
func (s *Store) getNamespace(name string) (*VectorNamespace, bool) {
	s.vecMu.RLock()
	ns, ok := s.vectors[name]
	s.vecMu.RUnlock()
	return ns, ok
}

// getOrCreateNamespace returns an existing or newly created VectorNamespace.
func (s *Store) getOrCreateNamespace(name string) *VectorNamespace {
	s.vecMu.RLock()
	ns, ok := s.vectors[name]
	s.vecMu.RUnlock()
	if ok {
		return ns
	}

	s.vecMu.Lock()
	defer s.vecMu.Unlock()
	ns, ok = s.vectors[name]
	if !ok {
		ns = newVectorNamespace()
		s.vectors[name] = ns
	}
	return ns
}

// --- Vector Store Operations (Per-Namespace Lock Sharding) ---

// VSet stores a vector in the given namespace with no TTL.
func (s *Store) VSet(namespace, id string, dim int, vec []float32, meta map[string]string) error {
	return s.VSetWithTTL(namespace, id, dim, vec, meta, 0)
}

// VSetWithTTL stores a vector in the given namespace with an optional TTL.
func (s *Store) VSetWithTTL(namespace, id string, dim int, vec []float32, meta map[string]string, ttl time.Duration) error {
	if len(vec) != dim {
		return fmt.Errorf("dimension mismatch: declared %d, got %d floats", dim, len(vec))
	}

	ns := s.getOrCreateNamespace(namespace)
	ns.mu.Lock()
	defer ns.mu.Unlock()

	// Copy the vector to own the memory
	vecCopy := make([]float32, len(vec))
	copy(vecCopy, vec)

	// Copy metadata
	var metaCopy map[string]string
	if len(meta) > 0 {
		metaCopy = make(map[string]string, len(meta))
		for k, v := range meta {
			metaCopy[k] = v
		}
	}

	entry := &VectorEntry{
		ID:       id,
		Vector:   vecCopy,
		Metadata: metaCopy,
	}

	// Resolve TTL: explicit > namespace default > none
	if ttl > 0 {
		entry.ExpiresAt = time.Now().Add(ttl)
		entry.HasTTL = true
	} else if ttl == 0 && ns.HasDefaultTTL {
		entry.ExpiresAt = time.Now().Add(ns.DefaultTTL)
		entry.HasTTL = true
	}

	ns.entries[id] = entry
	return nil
}

// VMSet batch-stores multiple vectors in the given namespace in a single call.
func (s *Store) VMSet(namespace string, dim int, entries []VMSetEntry) (int, error) {
	ns := s.getOrCreateNamespace(namespace)
	ns.mu.Lock()
	defer ns.mu.Unlock()

	count := 0
	for i, e := range entries {
		if len(e.Vector) != dim {
			return count, fmt.Errorf("entry %d (%s): dimension mismatch: declared %d, got %d floats", i, e.ID, dim, len(e.Vector))
		}

		vecCopy := make([]float32, len(e.Vector))
		copy(vecCopy, e.Vector)

		var metaCopy map[string]string
		if len(e.Metadata) > 0 {
			metaCopy = make(map[string]string, len(e.Metadata))
			for k, v := range e.Metadata {
				metaCopy[k] = v
			}
		}

		entry := &VectorEntry{
			ID:       e.ID,
			Vector:   vecCopy,
			Metadata: metaCopy,
		}

		if ns.HasDefaultTTL {
			entry.ExpiresAt = time.Now().Add(ns.DefaultTTL)
			entry.HasTTL = true
		}

		ns.entries[e.ID] = entry
		count++
	}

	return count, nil
}

// VGet retrieves a vector entry from a namespace. Performs lazy expiry.
func (s *Store) VGet(namespace, id string) (*VectorEntry, bool) {
	ns, ok := s.getNamespace(namespace)
	if !ok {
		return nil, false
	}

	ns.mu.RLock()
	entry, ok := ns.entries[id]
	if !ok {
		ns.mu.RUnlock()
		return nil, false
	}
	if entry.isExpired() {
		ns.mu.RUnlock()
		ns.mu.Lock()
		entry, ok = ns.entries[id]
		if ok && entry.isExpired() {
			delete(ns.entries, id)
		}
		ns.mu.Unlock()
		return nil, false
	}
	ns.mu.RUnlock()
	return entry, true
}

// VDel removes a vector from a namespace. Returns true if it existed.
func (s *Store) VDel(namespace, id string) bool {
	ns, ok := s.getNamespace(namespace)
	if !ok {
		return false
	}
	ns.mu.Lock()
	defer ns.mu.Unlock()

	if _, ok := ns.entries[id]; !ok {
		return false
	}
	delete(ns.entries, id)
	return true
}

// VCount returns the number of non-expired vectors in a namespace.
func (s *Store) VCount(namespace string) int {
	ns, ok := s.getNamespace(namespace)
	if !ok {
		return 0
	}
	ns.mu.RLock()
	defer ns.mu.RUnlock()

	count := 0
	for _, entry := range ns.entries {
		if !entry.isExpired() {
			count++
		}
	}
	return count
}

// VSnapshot returns a snapshot of all non-expired vector entries in a namespace.
func (s *Store) VSnapshot(namespace string) []*VectorEntry {
	ns, ok := s.getNamespace(namespace)
	if !ok {
		return nil
	}
	ns.mu.RLock()
	defer ns.mu.RUnlock()

	snapshot := make([]*VectorEntry, 0, len(ns.entries))
	for _, entry := range ns.entries {
		if !entry.isExpired() {
			snapshot = append(snapshot, entry)
		}
	}
	if len(snapshot) == 0 {
		return nil
	}
	return snapshot
}

// VExpireNamespace sets a default TTL for an entire namespace.
func (s *Store) VExpireNamespace(namespace string, ttl time.Duration) bool {
	ns, ok := s.getNamespace(namespace)
	if !ok {
		return false
	}
	ns.mu.Lock()
	defer ns.mu.Unlock()

	ns.DefaultTTL = ttl
	ns.HasDefaultTTL = true
	ns.NamespaceTTLSetAt = time.Now()

	expiresAt := time.Now().Add(ttl)
	for _, entry := range ns.entries {
		if !entry.HasTTL {
			entry.ExpiresAt = expiresAt
			entry.HasTTL = true
		}
	}

	return true
}

// VNSDrop atomically removes an entire namespace and all its entries.
func (s *Store) VNSDrop(namespace string) bool {
	s.vecMu.Lock()
	defer s.vecMu.Unlock()

	_, ok := s.vectors[namespace]
	if !ok {
		return false
	}

	delete(s.vectors, namespace)
	return true
}

// VNSList returns summary information about all active vector namespaces.
func (s *Store) VNSList() []VNamespaceInfo {
	s.vecMu.RLock()
	namespaces := make([]*VectorNamespace, 0, len(s.vectors))
	names := make([]string, 0, len(s.vectors))
	for name, ns := range s.vectors {
		names = append(names, name)
		namespaces = append(namespaces, ns)
	}
	s.vecMu.RUnlock()

	infos := make([]VNamespaceInfo, 0, len(namespaces))
	for i, ns := range namespaces {
		name := names[i]
		ns.mu.RLock()
		count := 0
		var approxMem int64
		for _, entry := range ns.entries {
			if !entry.isExpired() {
				count++
				approxMem += int64(len(entry.Vector)*4) + 64
				for k, v := range entry.Metadata {
					approxMem += int64(len(k) + len(v))
				}
			}
		}

		info := VNamespaceInfo{
			Name:         name,
			VectorCount:  count,
			ApproxMemory: approxMem,
			HasTTL:       ns.HasDefaultTTL,
			TTLRemaining: -1,
		}

		if ns.HasDefaultTTL {
			elapsed := time.Since(ns.NamespaceTTLSetAt)
			remaining := ns.DefaultTTL - elapsed
			if remaining > 0 {
				info.TTLRemaining = int(remaining.Seconds())
			} else {
				info.TTLRemaining = 0
			}
		}
		ns.mu.RUnlock()

		infos = append(infos, info)
	}
	return infos
}

// VectorNamespaces returns the names of all vector namespaces (for INFO).
func (s *Store) VectorNamespaces() []string {
	s.vecMu.RLock()
	defer s.vecMu.RUnlock()

	names := make([]string, 0, len(s.vectors))
	for name := range s.vectors {
		names = append(names, name)
	}
	return names
}

// TotalVectors returns the total number of non-expired vectors across all namespaces.
func (s *Store) TotalVectors() int {
	s.vecMu.RLock()
	namespaces := make([]*VectorNamespace, 0, len(s.vectors))
	for _, ns := range s.vectors {
		namespaces = append(namespaces, ns)
	}
	s.vecMu.RUnlock()

	total := 0
	for _, ns := range namespaces {
		ns.mu.RLock()
		for _, entry := range ns.entries {
			if !entry.isExpired() {
				total++
			}
		}
		ns.mu.RUnlock()
	}
	return total
}

// SweepExpiredVectors removes all expired vector entries across all namespaces.
func (s *Store) SweepExpiredVectors() int {
	s.vecMu.Lock()
	defer s.vecMu.Unlock()
	return s.sweepExpiredVectorsLocked()
}
