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
	ExpiresAt time.Time
	HasTTL    bool
}

// isExpired reports whether this vector entry has exceeded its TTL.
func (e *VectorEntry) isExpired() bool {
	return e.HasTTL && time.Now().After(e.ExpiresAt)
}

// VectorNamespace is a named partition of vectors with optional default TTL.
type VectorNamespace struct {
	entries       map[string]*VectorEntry
	mu            sync.RWMutex
	DefaultTTL    time.Duration
	HasDefaultTTL bool
	// Track when the namespace-level TTL was set for VNS LIST reporting
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

// --- Vector Store Operations (on the main Store) ---

// VSet stores a vector in the given namespace with no TTL.
// Returns an error if the provided float count doesn't match dim.
func (s *Store) VSet(namespace, id string, dim int, vec []float32, meta map[string]string) error {
	return s.VSetWithTTL(namespace, id, dim, vec, meta, 0)
}

// VSetWithTTL stores a vector in the given namespace with an optional TTL.
// If ttl is 0, the vector inherits the namespace's default TTL (if any).
// If ttl is negative, the vector has no TTL regardless of namespace default.
func (s *Store) VSetWithTTL(namespace, id string, dim int, vec []float32, meta map[string]string, ttl time.Duration) error {
	if len(vec) != dim {
		return fmt.Errorf("dimension mismatch: declared %d, got %d floats", dim, len(vec))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	ns, ok := s.vectors[namespace]
	if !ok {
		ns = newVectorNamespace()
		s.vectors[namespace] = ns
	}

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
	// ttl < 0 means explicitly no TTL

	ns.entries[id] = entry
	return nil
}

// VMSet batch-stores multiple vectors in the given namespace in a single call.
// All vectors must match the declared dimension. Returns the count of successfully stored vectors.
func (s *Store) VMSet(namespace string, dim int, entries []VMSetEntry) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	ns, ok := s.vectors[namespace]
	if !ok {
		ns = newVectorNamespace()
		s.vectors[namespace] = ns
	}

	count := 0
	for i, e := range entries {
		if len(e.Vector) != dim {
			return count, fmt.Errorf("entry %d (%s): dimension mismatch: declared %d, got %d floats", i, e.ID, dim, len(e.Vector))
		}

		// Copy vector
		vecCopy := make([]float32, len(e.Vector))
		copy(vecCopy, e.Vector)

		// Copy metadata
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

		// Apply namespace default TTL if set
		if ns.HasDefaultTTL {
			entry.ExpiresAt = time.Now().Add(ns.DefaultTTL)
			entry.HasTTL = true
		}

		ns.entries[e.ID] = entry
		count++
	}

	return count, nil
}

// VGet retrieves a vector entry from a namespace.
// Performs lazy expiry: returns (nil, false) if the entry has expired.
func (s *Store) VGet(namespace, id string) (*VectorEntry, bool) {
	s.mu.RLock()
	ns, ok := s.vectors[namespace]
	if !ok {
		s.mu.RUnlock()
		return nil, false
	}
	entry, ok := ns.entries[id]
	if !ok {
		s.mu.RUnlock()
		return nil, false
	}
	if entry.isExpired() {
		s.mu.RUnlock()
		// Upgrade to write lock for lazy deletion
		s.mu.Lock()
		ns, ok = s.vectors[namespace]
		if ok {
			entry, ok = ns.entries[id]
			if ok && entry.isExpired() {
				delete(ns.entries, id)
			}
		}
		s.mu.Unlock()
		return nil, false
	}
	s.mu.RUnlock()
	return entry, true
}

// VDel removes a vector from a namespace. Returns true if it existed.
func (s *Store) VDel(namespace, id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	ns, ok := s.vectors[namespace]
	if !ok {
		return false
	}
	if _, ok := ns.entries[id]; !ok {
		return false
	}
	delete(ns.entries, id)
	return true
}

// VCount returns the number of non-expired vectors in a namespace.
func (s *Store) VCount(namespace string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ns, ok := s.vectors[namespace]
	if !ok {
		return 0
	}

	count := 0
	for _, entry := range ns.entries {
		if !entry.isExpired() {
			count++
		}
	}
	return count
}

// VSnapshot returns a snapshot of all non-expired vector entries in a namespace.
// The slice headers are copied so similarity computation can happen
// outside the lock without racing with concurrent writes.
func (s *Store) VSnapshot(namespace string) []*VectorEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ns, ok := s.vectors[namespace]
	if !ok {
		return nil
	}

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
// All existing non-expired entries in the namespace get this TTL applied.
// Future entries inherit this TTL unless they specify their own.
// Returns false if the namespace doesn't exist.
func (s *Store) VExpireNamespace(namespace string, ttl time.Duration) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	ns, ok := s.vectors[namespace]
	if !ok {
		return false
	}

	ns.DefaultTTL = ttl
	ns.HasDefaultTTL = true
	ns.NamespaceTTLSetAt = time.Now()

	// Apply TTL to all existing entries that don't already have a per-entry TTL
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
// Returns true if the namespace existed.
func (s *Store) VNSDrop(namespace string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, ok := s.vectors[namespace]
	if !ok {
		return false
	}

	delete(s.vectors, namespace)
	return true
}

// VNSList returns summary information about all active vector namespaces.
func (s *Store) VNSList() []VNamespaceInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	infos := make([]VNamespaceInfo, 0, len(s.vectors))
	for name, ns := range s.vectors {
		count := 0
		var approxMem int64
		for _, entry := range ns.entries {
			if !entry.isExpired() {
				count++
				// Approximate memory: vector data + overhead
				// float32 = 4 bytes per element + string keys/values estimate
				approxMem += int64(len(entry.Vector)*4) + 64 // 64 bytes struct overhead
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
			// Calculate remaining TTL from when it was set + duration
			elapsed := time.Since(ns.NamespaceTTLSetAt)
			remaining := ns.DefaultTTL - elapsed
			if remaining > 0 {
				info.TTLRemaining = int(remaining.Seconds())
			} else {
				info.TTLRemaining = 0
			}
		}

		infos = append(infos, info)
	}
	return infos
}

// VectorNamespaces returns the names of all vector namespaces (for INFO).
func (s *Store) VectorNamespaces() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	names := make([]string, 0, len(s.vectors))
	for name := range s.vectors {
		names = append(names, name)
	}
	return names
}

// TotalVectors returns the total number of non-expired vectors across all namespaces.
func (s *Store) TotalVectors() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	total := 0
	for _, ns := range s.vectors {
		for _, entry := range ns.entries {
			if !entry.isExpired() {
				total++
			}
		}
	}
	return total
}

// SweepExpiredVectors removes all expired vector entries across all namespaces.
// Also removes empty namespaces left after sweeping.
// Returns the count of expired vectors removed.
// This is the public API — acquires its own lock. For internal use under
// an existing lock, see sweepExpiredVectorsLocked in store.go.
func (s *Store) SweepExpiredVectors() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sweepExpiredVectorsLocked()
}
