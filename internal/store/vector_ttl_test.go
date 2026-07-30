// vector_ttl_test.go — Tests for ephemeral vector TTL features.
//
// These tests validate the per-vector and per-namespace TTL, batch ingestion (VMSet),
// namespace lifecycle (VNSDrop, VNSList), lazy expiry, and background sweep functionality
// that were added for Microsoft AutoGen v0.4 multi-agent swarm compatibility.
//
// See AUTOGEN_NEARBY_INTEGRATION_REPORT.md §2.1 for the design rationale.
package store

import (
	"testing"
	"time"
)

func TestVSetWithTTL(t *testing.T) {
	s := New()
	vec := []float32{0.1, 0.2, 0.3}

	err := s.VSetWithTTL("docs", "chunk:1", 3, vec, nil, 1*time.Second)
	if err != nil {
		t.Fatalf("VSetWithTTL failed: %v", err)
	}

	// Should be retrievable immediately
	entry, ok := s.VGet("docs", "chunk:1")
	if !ok || entry == nil {
		t.Fatal("VGet should return entry before TTL expires")
	}

	// Wait for expiry
	time.Sleep(1100 * time.Millisecond)

	// Should be expired now (lazy expiry on read)
	_, ok = s.VGet("docs", "chunk:1")
	if ok {
		t.Error("VGet should return not-found after TTL expires")
	}
}

func TestVSetInheritsNamespaceDefaultTTL(t *testing.T) {
	s := New()

	// Set a vector first to create the namespace
	s.VSet("swarm:run-1", "seed", 2, []float32{1.0, 0.0}, nil)

	// Set namespace default TTL
	ok := s.VExpireNamespace("swarm:run-1", 1*time.Second)
	if !ok {
		t.Fatal("VExpireNamespace should succeed for existing namespace")
	}

	// New vectors should inherit namespace TTL
	s.VSet("swarm:run-1", "chunk:2", 2, []float32{0.0, 1.0}, nil)

	// Should be retrievable immediately
	_, ok = s.VGet("swarm:run-1", "chunk:2")
	if !ok {
		t.Fatal("VGet should return entry before namespace TTL expires")
	}

	// Wait for expiry
	time.Sleep(1100 * time.Millisecond)

	_, ok = s.VGet("swarm:run-1", "chunk:2")
	if ok {
		t.Error("VGet should return not-found after namespace TTL expires")
	}
}

func TestVSetExplicitTTLOverridesNamespace(t *testing.T) {
	s := New()

	// Create namespace with 5s default TTL
	s.VSet("ns", "a", 2, []float32{1.0, 0.0}, nil)
	s.VExpireNamespace("ns", 5*time.Second)

	// Set a vector with explicit 1s TTL (shorter than namespace)
	s.VSetWithTTL("ns", "short-lived", 2, []float32{0.5, 0.5}, nil, 1*time.Second)

	time.Sleep(1100 * time.Millisecond)

	// Explicitly TTL'd vector should be expired
	_, ok := s.VGet("ns", "short-lived")
	if ok {
		t.Error("explicitly TTL'd vector should have expired")
	}

	// Other vectors (with 5s namespace TTL) should still exist
	_, ok = s.VGet("ns", "a")
	if !ok {
		t.Error("namespace-TTL'd vector should still be alive")
	}
}

func TestVExpireNamespace(t *testing.T) {
	s := New()

	// Non-existent namespace returns false
	if s.VExpireNamespace("nope", 10*time.Second) {
		t.Error("VExpireNamespace should return false for non-existent namespace")
	}

	// Create namespace with 3 vectors
	s.VSet("agents", "v1", 2, []float32{1.0, 0.0}, nil)
	s.VSet("agents", "v2", 2, []float32{0.0, 1.0}, nil)
	s.VSet("agents", "v3", 2, []float32{0.5, 0.5}, nil)

	ok := s.VExpireNamespace("agents", 1*time.Second)
	if !ok {
		t.Fatal("VExpireNamespace should succeed")
	}

	// All should be alive immediately
	if s.VCount("agents") != 3 {
		t.Errorf("expected 3 vectors, got %d", s.VCount("agents"))
	}

	// Wait for expiry
	time.Sleep(1100 * time.Millisecond)

	if s.VCount("agents") != 0 {
		t.Errorf("expected 0 vectors after TTL, got %d", s.VCount("agents"))
	}
}

func TestVNSDrop(t *testing.T) {
	s := New()

	// Drop non-existent namespace
	if s.VNSDrop("nope") {
		t.Error("VNSDrop should return false for non-existent namespace")
	}

	// Create namespace with vectors
	s.VSet("run:42", "a", 2, []float32{1.0, 0.0}, nil)
	s.VSet("run:42", "b", 2, []float32{0.0, 1.0}, nil)

	if !s.VNSDrop("run:42") {
		t.Error("VNSDrop should return true for existing namespace")
	}

	// Verify completely removed
	if s.VCount("run:42") != 0 {
		t.Error("VCount should be 0 after VNSDrop")
	}
	_, ok := s.VGet("run:42", "a")
	if ok {
		t.Error("VGet should return not-found after VNSDrop")
	}

	// Verify not in namespace list
	for _, name := range s.VectorNamespaces() {
		if name == "run:42" {
			t.Error("dropped namespace should not appear in VectorNamespaces")
		}
	}
}

func TestVNSList(t *testing.T) {
	s := New()

	// Empty initially
	infos := s.VNSList()
	if len(infos) != 0 {
		t.Errorf("expected 0 namespaces, got %d", len(infos))
	}

	// Create namespaces
	s.VSet("alpha", "v1", 3, []float32{1.0, 0.0, 0.0}, map[string]string{"stage": "researcher"})
	s.VSet("beta", "v1", 3, []float32{0.0, 1.0, 0.0}, nil)
	s.VSet("beta", "v2", 3, []float32{0.0, 0.0, 1.0}, nil)

	infos = s.VNSList()
	if len(infos) != 2 {
		t.Fatalf("expected 2 namespaces, got %d", len(infos))
	}

	// Find each namespace info
	found := make(map[string]VNamespaceInfo)
	for _, info := range infos {
		found[info.Name] = info
	}

	alphaInfo, ok := found["alpha"]
	if !ok {
		t.Fatal("namespace 'alpha' not found")
	}
	if alphaInfo.VectorCount != 1 {
		t.Errorf("alpha: expected 1 vector, got %d", alphaInfo.VectorCount)
	}
	if alphaInfo.ApproxMemory <= 0 {
		t.Error("alpha: approx memory should be > 0")
	}
	if alphaInfo.HasTTL {
		t.Error("alpha: should not have TTL")
	}
	if alphaInfo.TTLRemaining != -1 {
		t.Errorf("alpha: TTLRemaining should be -1, got %d", alphaInfo.TTLRemaining)
	}

	betaInfo, ok := found["beta"]
	if !ok {
		t.Fatal("namespace 'beta' not found")
	}
	if betaInfo.VectorCount != 2 {
		t.Errorf("beta: expected 2 vectors, got %d", betaInfo.VectorCount)
	}
}

func TestVNSListWithTTL(t *testing.T) {
	s := New()
	s.VSet("ephemeral", "v1", 2, []float32{1.0, 0.0}, nil)
	s.VExpireNamespace("ephemeral", 60*time.Second)

	infos := s.VNSList()
	if len(infos) != 1 {
		t.Fatalf("expected 1 namespace, got %d", len(infos))
	}

	if !infos[0].HasTTL {
		t.Error("namespace should have TTL")
	}
	if infos[0].TTLRemaining <= 0 || infos[0].TTLRemaining > 60 {
		t.Errorf("TTLRemaining should be between 1-60, got %d", infos[0].TTLRemaining)
	}
}

func TestVMSet(t *testing.T) {
	s := New()

	entries := []VMSetEntry{
		{ID: "chunk:1", Vector: []float32{1.0, 0.0, 0.0}, Metadata: map[string]string{"source": "a.pdf"}},
		{ID: "chunk:2", Vector: []float32{0.0, 1.0, 0.0}, Metadata: nil},
		{ID: "chunk:3", Vector: []float32{0.0, 0.0, 1.0}, Metadata: map[string]string{"source": "b.pdf"}},
	}

	count, err := s.VMSet("batch-ns", 3, entries)
	if err != nil {
		t.Fatalf("VMSet failed: %v", err)
	}
	if count != 3 {
		t.Errorf("expected 3 stored, got %d", count)
	}

	// Verify all entries
	if s.VCount("batch-ns") != 3 {
		t.Errorf("expected VCount 3, got %d", s.VCount("batch-ns"))
	}

	e, ok := s.VGet("batch-ns", "chunk:1")
	if !ok {
		t.Fatal("chunk:1 not found")
	}
	if e.Metadata["source"] != "a.pdf" {
		t.Errorf("expected source=a.pdf, got %s", e.Metadata["source"])
	}
}

func TestVMSetDimensionMismatch(t *testing.T) {
	s := New()

	entries := []VMSetEntry{
		{ID: "good", Vector: []float32{1.0, 0.0, 0.0}},
		{ID: "bad", Vector: []float32{1.0, 0.0}}, // wrong dim
	}

	count, err := s.VMSet("ns", 3, entries)
	if err == nil {
		t.Fatal("expected dimension mismatch error")
	}
	if count != 1 {
		t.Errorf("expected 1 stored before error, got %d", count)
	}
}

func TestVMSetInheritsNamespaceTTL(t *testing.T) {
	s := New()

	// Create namespace with TTL
	s.VSet("ttl-ns", "seed", 2, []float32{1.0, 0.0}, nil)
	s.VExpireNamespace("ttl-ns", 1*time.Second)

	// Batch insert
	entries := []VMSetEntry{
		{ID: "b1", Vector: []float32{0.5, 0.5}},
		{ID: "b2", Vector: []float32{0.3, 0.7}},
	}
	s.VMSet("ttl-ns", 2, entries)

	// Should exist now
	if s.VCount("ttl-ns") != 3 { // seed + b1 + b2
		t.Errorf("expected 3 vectors, got %d", s.VCount("ttl-ns"))
	}

	time.Sleep(1100 * time.Millisecond)

	// All should be expired
	if s.VCount("ttl-ns") != 0 {
		t.Errorf("expected 0 after TTL, got %d", s.VCount("ttl-ns"))
	}
}

func TestSweepExpiredVectors(t *testing.T) {
	s := New()

	// Create some vectors with short TTL
	s.VSetWithTTL("sweep-ns", "v1", 2, []float32{1.0, 0.0}, nil, 100*time.Millisecond)
	s.VSetWithTTL("sweep-ns", "v2", 2, []float32{0.0, 1.0}, nil, 100*time.Millisecond)
	// One without TTL
	s.VSet("sweep-ns", "v3", 2, []float32{0.5, 0.5}, nil)

	time.Sleep(200 * time.Millisecond)

	swept := s.SweepExpiredVectors()
	if swept != 2 {
		t.Errorf("expected 2 swept, got %d", swept)
	}

	// v3 should still exist
	_, ok := s.VGet("sweep-ns", "v3")
	if !ok {
		t.Error("non-TTL vector should survive sweep")
	}
}

func TestSweepCleansUpEmptyNamespaces(t *testing.T) {
	s := New()

	// All entries with short TTL
	s.VSetWithTTL("temp-ns", "v1", 2, []float32{1.0, 0.0}, nil, 100*time.Millisecond)
	s.VSetWithTTL("temp-ns", "v2", 2, []float32{0.0, 1.0}, nil, 100*time.Millisecond)

	// Another namespace that persists
	s.VSet("persist-ns", "v1", 2, []float32{1.0, 0.0}, nil)

	time.Sleep(200 * time.Millisecond)

	s.SweepExpiredVectors()

	// temp-ns should be gone
	namespaces := s.VectorNamespaces()
	for _, ns := range namespaces {
		if ns == "temp-ns" {
			t.Error("empty namespace should be cleaned up after sweep")
		}
	}

	// persist-ns should remain
	found := false
	for _, ns := range namespaces {
		if ns == "persist-ns" {
			found = true
			break
		}
	}
	if !found {
		t.Error("persist-ns should still exist")
	}
}

func TestExpiredVectorsExcludedFromSnapshot(t *testing.T) {
	s := New()

	s.VSetWithTTL("snap-ns", "expired", 2, []float32{1.0, 0.0}, nil, 100*time.Millisecond)
	s.VSet("snap-ns", "alive", 2, []float32{0.0, 1.0}, nil)

	time.Sleep(200 * time.Millisecond)

	snap := s.VSnapshot("snap-ns")
	if len(snap) != 1 {
		t.Errorf("expected 1 entry in snapshot, got %d", len(snap))
	}
	if snap[0].ID != "alive" {
		t.Errorf("expected alive in snapshot, got %s", snap[0].ID)
	}
}

func TestExpiredVectorsExcludedFromVCount(t *testing.T) {
	s := New()

	s.VSetWithTTL("count-ns", "v1", 2, []float32{1.0, 0.0}, nil, 100*time.Millisecond)
	s.VSet("count-ns", "v2", 2, []float32{0.0, 1.0}, nil)

	time.Sleep(200 * time.Millisecond)

	count := s.VCount("count-ns")
	if count != 1 {
		t.Errorf("expected VCount 1 (excluding expired), got %d", count)
	}
}

func TestTotalVectorsExcludesExpired(t *testing.T) {
	s := New()

	s.VSetWithTTL("ns1", "v1", 2, []float32{1.0, 0.0}, nil, 100*time.Millisecond)
	s.VSet("ns2", "v1", 2, []float32{0.0, 1.0}, nil)

	time.Sleep(200 * time.Millisecond)

	total := s.TotalVectors()
	if total != 1 {
		t.Errorf("expected TotalVectors 1, got %d", total)
	}
}

func TestVSetNegativeTTLBypassesNamespaceDefault(t *testing.T) {
	s := New()

	// Create namespace with short TTL
	s.VSet("ns", "seed", 2, []float32{1.0, 0.0}, nil)
	s.VExpireNamespace("ns", 100*time.Millisecond)

	// Set vector with negative TTL = no expiry
	s.VSetWithTTL("ns", "persistent", 2, []float32{0.5, 0.5}, nil, -1)

	time.Sleep(200 * time.Millisecond)

	// seed should be expired, but persistent should remain
	_, ok := s.VGet("ns", "seed")
	if ok {
		t.Error("seed should have expired")
	}

	_, ok = s.VGet("ns", "persistent")
	if !ok {
		t.Error("persistent vector should not expire (negative TTL bypasses namespace default)")
	}
}
