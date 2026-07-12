package bench

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/server"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/pkg/client"
)

func startServer(b *testing.B) (string, func()) {
	b.Helper()
	s := store.New()
	// Disable TLS for benchmarks to test raw DB throughput
	srv := server.New(":0", "", false, "", "", s)
	go srv.ListenAndServe()
	srv.WaitReady()

	return srv.Addr(), func() { srv.Shutdown() }
}

func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

func BenchmarkSetGet(b *testing.B) {
	addr, shutdown := startServer(b)
	defer shutdown()

	c, err := client.New(client.Options{Addr: addr})
	if err != nil {
		b.Fatalf("connect: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	val := string(make([]byte, 100))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("key:%d", i%1000)
		_ = c.Set(ctx, key, val, 0)
		_, _ = c.Get(ctx, key)
	}
}

func BenchmarkVSet1536(b *testing.B) {
	addr, shutdown := startServer(b)
	defer shutdown()

	c, err := client.New(client.Options{Addr: addr})
	if err != nil {
		b.Fatalf("connect: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	vec := generateRandomVector(1536)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = c.VSet(ctx, client.VSetArgs{
			Namespace: "docs",
			ID:        fmt.Sprintf("vec:%d", i),
			Vector:    vec,
		})
	}
}

func benchmarkVSimilarity(b *testing.B, numVectors int) {
	addr, shutdown := startServer(b)
	defer shutdown()

	c, err := client.New(client.Options{Addr: addr})
	if err != nil {
		b.Fatalf("connect: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	dim := 1536

	// Pre-load vectors
	b.StopTimer()
	for i := 0; i < numVectors; i++ {
		vec := generateRandomVector(dim)
		err := c.VSet(ctx, client.VSetArgs{
			Namespace: "docs",
			ID:        fmt.Sprintf("vec:%d", i),
			Vector:    vec,
		})
		if err != nil {
			b.Fatalf("VSet failed: %v", err)
		}
	}
	b.StartTimer()

	query := generateRandomVector(dim)
	for i := 0; i < b.N; i++ {
		_, err := c.VSimilarity(ctx, client.VSimilarityArgs{
			Namespace: "docs",
			Vector:    query,
			TopK:      10,
		})
		if err != nil {
			b.Fatalf("VSimilarity failed: %v", err)
		}
	}
}

func BenchmarkVSimilarity1K(b *testing.B)   { benchmarkVSimilarity(b, 1000) }
func BenchmarkVSimilarity10K(b *testing.B)  { benchmarkVSimilarity(b, 10000) }
func BenchmarkVSimilarity100K(b *testing.B) { benchmarkVSimilarity(b, 100000) }
