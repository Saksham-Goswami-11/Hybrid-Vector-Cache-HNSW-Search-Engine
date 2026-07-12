package hnsw_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/hnsw"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/similarity"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
)

func randomVector(rng *rand.Rand, dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}
	return vec
}

func BenchmarkHNSWInsert(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 1536
	idx := hnsw.NewIndex(16, 200)

	vecs := make([][]float32, b.N)
	for i := 0; i < b.N; i++ {
		vecs[i] = randomVector(rng, dim)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("vec:%d", i)
		_ = idx.Insert(key, vecs[i], nil)
	}
}

func BenchmarkSearch(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 1536
	n := 10000 // 10K database
	K := 10

	vecs := make([][]float32, n)
	entries := make([]*store.VectorEntry, n)
	idx := hnsw.NewIndex(16, 200)

	for i := 0; i < n; i++ {
		vecs[i] = randomVector(rng, dim)
		key := fmt.Sprintf("vec:%d", i)
		
		_ = idx.Insert(key, vecs[i], nil)
		
		entries[i] = &store.VectorEntry{
			ID:     key,
			Vector: vecs[i],
		}
	}

	queries := make([][]float32, 100)
	for i := 0; i < 100; i++ {
		queries[i] = randomVector(rng, dim)
	}

	b.Run("HNSW", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			_, _ = idx.Search(q, K, 100)
		}
	})

	b.Run("BruteForce_SingleThread", func(b *testing.B) {
		engine := similarity.NewEngine(1)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			_ = engine.TopK(q, entries, K)
		}
	})
}
