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
// It manages both the key-value namespace and the vector namespace.
type Store struct {
	kv      map[string]*kvEntry
	vectors map[string]*VectorNamespace
	mu      sync.RWMutex
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
	s.mu.Lock()
	defer s.mu.Unlock()
	s.kv[key] = &kvEntry{Value: value}
}

// SetWithTTL stores a key-value pair that expires after ttl.
func (s *Store) SetWithTTL(key, value string, ttl time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.kv[key] = &kvEntry{
		Value:     value,
		ExpiresAt: time.Now().Add(ttl),
		HasTTL:    true,
	}
}

// Get retrieves the value for a key.
// Returns ("", false) if the key doesn't exist or has expired (lazy expiry).
func (s *Store) Get(key string) (string, bool) {
	s.mu.RLock()
	entry, ok := s.kv[key]
	if !ok {
		s.mu.RUnlock()
		return "", false
	}
	if entry.isExpired() {
		s.mu.RUnlock()
		// Upgrade to write lock to delete expired key
		s.mu.Lock()
		// Double-check after acquiring write lock
		entry, ok = s.kv[key]
		if ok && entry.isExpired() {
			delete(s.kv, key)
		}
		s.mu.Unlock()
		return "", false
	}
	val := entry.Value
	s.mu.RUnlock()
	return val, true
}

// Del deletes one or more keys. Returns the count of keys that were actually deleted.
func (s *Store) Del(keys ...string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
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
	s.mu.Lock()
	defer s.mu.Unlock()
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
	s.mu.RLock()
	defer s.mu.RUnlock()
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
	s.mu.RLock()
	defer s.mu.RUnlock()
	count := 0
	for _, entry := range s.kv {
		if !entry.isExpired() {
			count++
		}
	}
	return count
}

// SweepExpired removes all expired KV keys and expired vector entries.
// Called by the background expiry goroutine.
func (s *Store) SweepExpired() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	count := 0
	for key, entry := range s.kv {
		if entry.isExpired() {
			delete(s.kv, key)
			count++
		}
	}
	// Sweep expired vectors under the same lock
	count += s.sweepExpiredVectorsLocked()
	return count
}

// sweepExpiredVectorsLocked removes expired vectors. Caller must hold s.mu.Lock().
func (s *Store) sweepExpiredVectorsLocked() int {
	count := 0
	var emptyNamespaces []string

	for nsName, ns := range s.vectors {
		for id, entry := range ns.entries {
			if entry.isExpired() {
				delete(ns.entries, id)
				count++
			}
		}
		if len(ns.entries) == 0 {
			emptyNamespaces = append(emptyNamespaces, nsName)
		}
	}

	for _, nsName := range emptyNamespaces {
		delete(s.vectors, nsName)
	}

	return count
}
