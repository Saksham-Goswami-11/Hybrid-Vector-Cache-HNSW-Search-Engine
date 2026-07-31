package hnsw

import (
	"testing"
)

func TestHNSWDelete(t *testing.T) {
	idx := NewIndex(16, 200)

	v1 := []float32{1.0, 0.0, 0.0, 0.0}
	v2 := []float32{0.0, 1.0, 0.0, 0.0}
	v3 := []float32{0.0, 0.0, 1.0, 0.0}

	idx.Insert("key1", v1, map[string]string{"doc": "1"})
	idx.Insert("key2", v2, map[string]string{"doc": "2"})
	idx.Insert("key3", v3, map[string]string{"doc": "3"})

	// Verify key1 is returned
	results, err := idx.Search(v1, 1, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 1 || results[0].ID != "key1" {
		t.Fatalf("Expected key1, got %v", results)
	}

	// Delete key1
	ok := idx.Delete("key1")
	if !ok {
		t.Fatalf("Expected Delete to return true")
	}

	// Double delete returns false
	if idx.Delete("key1") {
		t.Fatalf("Expected second Delete to return false")
	}

	// Search v1 again -> should return key2 or key3, but NOT key1
	resultsAfter, err := idx.Search(v1, 2, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	for _, r := range resultsAfter {
		if r.ID == "key1" {
			t.Fatalf("Search returned tombstoned node key1")
		}
	}
}
