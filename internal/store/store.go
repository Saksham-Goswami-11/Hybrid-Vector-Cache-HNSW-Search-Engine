package store

import (
	"sync"
	"time"
)

// kvEntry holds a string value with optional TTL.
type kvEntry struct {
	Value     string
	ExpiresAt time.Time
	HasTTL    bool
}

// isExpired reports whether this entry has exceeded its TTL.
func (e *kvEntry) isExpired() bool {
	return e.HasTTL && time.Now().After(e.ExpiresAt)
}

// Store is the thread-safe in-memory data store.
// It uses sharded locking: kvMu protects KV keys, vecMu protects namespace map operations,
// and each VectorNamespace has its own RWMutex for high-concurrency vector operations across namespaces.
type Store struct {
	kv      map[string]*kvEntry
	kvMu    sync.RWMutex

	vectors map[string]*VectorNamespace
	vecMu   sync.RWMutex
}

// New creates an empty Store.
func New() *Store {
	return &Store{
		kv:      make(map[string]*kvEntry),
		vectors: make(map[string]*VectorNamespace),
	}
}

// --- Key-Value Operations ---

// Set stores a key-value pair with no expiration.
func (s *Store) Set(key, value string) {
	s.kvMu.Lock()
	defer s.kvMu.Unlock()
	s.kv[key] = &kvEntry{Value: value}
}

// SetWithTTL stores a key-value pair that expires after ttl.
func (s *Store) SetWithTTL(key, value string, ttl time.Duration) {
	s.kvMu.Lock()
	defer s.kvMu.Unlock()
	s.kv[key] = &kvEntry{
		Value:     value,
		ExpiresAt: time.Now().Add(ttl),
		HasTTL:    true,
	}
}

// Get retrieves the value for a key.
// Returns ("", false) if the key doesn't exist or has expired (lazy expiry).
func (s *Store) Get(key string) (string, bool) {
	s.kvMu.RLock()
	entry, ok := s.kv[key]
	if !ok {
		s.kvMu.RUnlock()
		return "", false
	}
	if entry.isExpired() {
		s.kvMu.RUnlock()
		// Upgrade to write lock to delete expired key
		s.kvMu.Lock()
		// Double-check after acquiring write lock
		entry, ok = s.kv[key]
		if ok && entry.isExpired() {
			delete(s.kv, key)
		}
		s.kvMu.Unlock()
		return "", false
	}
	val := entry.Value
	s.kvMu.RUnlock()
	return val, true
}

// Del deletes one or more keys. Returns the count of keys that were actually deleted.
func (s *Store) Del(keys ...string) int {
	s.kvMu.Lock()
	defer s.kvMu.Unlock()
	count := 0
	for _, key := range keys {
		if _, ok := s.kv[key]; ok {
			delete(s.kv, key)
			count++
		}
	}
	return count
}

// Expire sets a TTL on an existing key. Returns true if the key exists.
func (s *Store) Expire(key string, ttl time.Duration) bool {
	s.kvMu.Lock()
	defer s.kvMu.Unlock()
	entry, ok := s.kv[key]
	if !ok || entry.isExpired() {
		return false
	}
	entry.ExpiresAt = time.Now().Add(ttl)
	entry.HasTTL = true
	return true
}

// TTL returns the remaining TTL for a key in seconds.
// Returns -1 if the key exists but has no TTL, -2 if the key doesn't exist.
func (s *Store) TTL(key string) int {
	s.kvMu.RLock()
	defer s.kvMu.RUnlock()
	entry, ok := s.kv[key]
	if !ok || entry.isExpired() {
		return -2
	}
	if !entry.HasTTL {
		return -1
	}
	remaining := time.Until(entry.ExpiresAt)
	if remaining <= 0 {
		return -2
	}
	return int(remaining.Seconds())
}

// KVCount returns the number of non-expired keys (for INFO command).
func (s *Store) KVCount() int {
	s.kvMu.RLock()
	defer s.kvMu.RUnlock()
	count := 0
	for _, entry := range s.kv {
		if !entry.isExpired() {
			count++
		}
	}
	return count
}

// SweepExpired removes all expired KV keys and expired vector entries.
// Called by the background expiry goroutine (every 100ms from server.expirySweep).
func (s *Store) SweepExpired() int {
	// Sweep KV keys under kvMu
	s.kvMu.Lock()
	count := 0
	for key, entry := range s.kv {
		if entry.isExpired() {
			delete(s.kv, key)
			count++
		}
	}
	s.kvMu.Unlock()

	// Sweep vector namespaces under vecMu
	s.vecMu.Lock()
	defer s.vecMu.Unlock()

	var emptyNamespaces []string
	for nsName, ns := range s.vectors {
		ns.mu.Lock()
		for id, entry := range ns.entries {
			if entry.isExpired() {
				delete(ns.entries, id)
				count++
			}
		}
		if len(ns.entries) == 0 {
			emptyNamespaces = append(emptyNamespaces, nsName)
		}
		ns.mu.Unlock()
	}

	for _, nsName := range emptyNamespaces {
		delete(s.vectors, nsName)
	}

	return count
}

// sweepExpiredVectorsLocked is kept for backwards compatibility with internal callers.
func (s *Store) sweepExpiredVectorsLocked() int {
	count := 0
	var emptyNamespaces []string

	for nsName, ns := range s.vectors {
		ns.mu.Lock()
		for id, entry := range ns.entries {
			if entry.isExpired() {
				delete(ns.entries, id)
				count++
			}
		}
		if len(ns.entries) == 0 {
			emptyNamespaces = append(emptyNamespaces, nsName)
		}
		ns.mu.Unlock()
	}

	for _, nsName := range emptyNamespaces {
		delete(s.vectors, nsName)
	}

	return count
}
