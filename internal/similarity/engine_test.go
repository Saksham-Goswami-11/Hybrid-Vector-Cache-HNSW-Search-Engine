package similarity

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
)

func makeEntry(id string, dim int) *store.VectorEntry {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1 // random in [-1, 1]
	}
	return &store.VectorEntry{ID: id, Vector: vec}
}

func TestTopKBasic(t *testing.T) {
	engine := NewEngine(1) // single-threaded for determinism

	entries := []*store.VectorEntry{
		{ID: "a", Vector: []float32{1, 0, 0}},
		{ID: "b", Vector: []float32{0, 1, 0}},
		{ID: "c", Vector: []float32{0.9, 0.1, 0}},
	}

	query := []float32{1, 0, 0}
	results := engine.TopK(query, entries, 2)

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// First result should be "a" (identical to query)
	if results[0].ID != "a" {
		t.Errorf("expected top result 'a', got '%s'", results[0].ID)
	}
	// Second should be "c" (most similar after a)
	if results[1].ID != "c" {
		t.Errorf("expected second result 'c', got '%s'", results[1].ID)
	}

	// Scores should be descending
	if results[0].Score < results[1].Score {
		t.Error("results should be in descending order")
	}
}

func TestTopKLargerThanEntries(t *testing.T) {
	engine := NewEngine(1)
	entries := []*store.VectorEntry{
		{ID: "a", Vector: []float32{1, 0}},
		{ID: "b", Vector: []float32{0, 1}},
	}

	results := engine.TopK([]float32{1, 0}, entries, 10)
	if len(results) != 2 {
		t.Errorf("expected 2 results (clamped to entries), got %d", len(results))
	}
}

func TestTopKEmpty(t *testing.T) {
	engine := NewEngine(1)
	results := engine.TopK([]float32{1, 0}, nil, 5)
	if results != nil {
		t.Errorf("expected nil for empty entries, got %v", results)
	}
}

func TestTopKParallel(t *testing.T) {
	engine := NewEngine(4) // force parallel path

	// Create 200 entries to trigger parallel processing
	entries := make([]*store.VectorEntry, 200)
	for i := range entries {
		entries[i] = makeEntry(fmt.Sprintf("vec:%d", i), 128)
	}

	query := entries[0].Vector // search for the first entry

	results := engine.TopK(query, entries, 5)
	if len(results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(results))
	}

	// First result should be vec:0 with score ~1.0
	if results[0].ID != "vec:0" {
		t.Errorf("expected top result 'vec:0', got '%s' (score: %f)", results[0].ID, results[0].Score)
	}

	// All scores should be descending
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not in descending order at index %d", i)
		}
	}
}

func TestTopKConcurrent(t *testing.T) {
	engine := NewEngine(4)

	entries := make([]*store.VectorEntry, 500)
	for i := range entries {
		entries[i] = makeEntry(fmt.Sprintf("vec:%d", i), 64)
	}

	// Run multiple TopK searches concurrently — no races allowed
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			query := entries[idx].Vector
			results := engine.TopK(query, entries, 3)
			if len(results) != 3 {
				t.Errorf("goroutine %d: expected 3 results, got %d", idx, len(results))
			}
		}(i)
	}
	wg.Wait()
}

func TestTopKWithMetadata(t *testing.T) {
	engine := NewEngine(1)

	entries := []*store.VectorEntry{
		{ID: "a", Vector: []float32{1, 0}, Metadata: map[string]string{"page": "1"}},
		{ID: "b", Vector: []float32{0, 1}, Metadata: map[string]string{"page": "2"}},
	}

	results := engine.TopK([]float32{1, 0}, entries, 1)
	if results[0].Metadata["page"] != "1" {
		t.Errorf("expected metadata page=1, got %s", results[0].Metadata["page"])
	}
}
