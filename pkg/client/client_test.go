package client

import (
	"context"
	"testing"
	"time"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/server"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
)

func startServer(t *testing.T) (string, func()) {
	t.Helper()
	s := store.New()
	srv := server.New(":0", "", false, "", "", s)

	go srv.ListenAndServe()
	srv.WaitReady()

	return srv.Addr(), func() { srv.Shutdown() }
}

func TestClientPing(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	if err := c.Ping(context.Background()); err != nil {
		t.Errorf("Ping: %v", err)
	}
}

func TestClientSetGet(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx := context.Background()

	// Set
	if err := c.Set(ctx, "key1", "val1", 0); err != nil {
		t.Fatalf("Set: %v", err)
	}

	// Get
	val, err := c.Get(ctx, "key1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if val != "val1" {
		t.Errorf("expected val1, got %s", val)
	}

	// Get missing key
	_, err = c.Get(ctx, "missing")
	if err != ErrNil {
		t.Errorf("expected ErrNil, got %v", err)
	}
}

func TestClientDel(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	c.Set(ctx, "a", "1", 0)
	c.Set(ctx, "b", "2", 0)

	n, err := c.Del(ctx, "a", "b")
	if err != nil {
		t.Fatalf("Del: %v", err)
	}
	if n != 2 {
		t.Errorf("expected 2 deleted, got %d", n)
	}
}

func TestClientSetWithTTL(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	c.Set(ctx, "temp", "val", 1*time.Second)

	val, err := c.Get(ctx, "temp")
	if err != nil || val != "val" {
		t.Fatalf("expected val, got %s (err: %v)", val, err)
	}

	time.Sleep(1100 * time.Millisecond)

	_, err = c.Get(ctx, "temp")
	if err != ErrNil {
		t.Errorf("expected ErrNil after TTL, got %v", err)
	}
}

func TestClientVSetAndVCount(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx := context.Background()

	err = c.VSet(ctx, VSetArgs{
		Namespace: "docs",
		ID:        "chunk:1",
		Vector:    []float32{0.1, 0.2, 0.3},
		Metadata:  map[string]string{"source": "test.pdf"},
	})
	if err != nil {
		t.Fatalf("VSet: %v", err)
	}

	count, err := c.VCount(ctx, "docs")
	if err != nil {
		t.Fatalf("VCount: %v", err)
	}
	if count != 1 {
		t.Errorf("expected count 1, got %d", count)
	}
}

func TestClientVSimilarity(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx := context.Background()

	// Store some vectors
	c.VSet(ctx, VSetArgs{Namespace: "docs", ID: "a", Vector: []float32{1, 0, 0}})
	c.VSet(ctx, VSetArgs{Namespace: "docs", ID: "b", Vector: []float32{0, 1, 0}})
	c.VSet(ctx, VSetArgs{Namespace: "docs", ID: "c", Vector: []float32{0.9, 0.1, 0}})

	results, err := c.VSimilarity(ctx, VSimilarityArgs{
		Namespace: "docs",
		Vector:    []float32{1, 0, 0},
		TopK:      2,
	})
	if err != nil {
		t.Fatalf("VSimilarity: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// First result should be "a" (identical)
	if results[0].ID != "a" {
		t.Errorf("expected top result 'a', got '%s'", results[0].ID)
	}
}

func TestClientContextCancellation(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
	defer cancel()
	time.Sleep(1 * time.Millisecond) // ensure context is expired

	_, err = c.Get(ctx, "key")
	if err == nil {
		t.Error("expected error from cancelled context")
	}
}

func TestClientExpireAndTTL(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	c.Set(ctx, "temp", "val", 0)

	// Set Expiration
	ok, err := c.Expire(ctx, "temp", 2*time.Second)
	if err != nil {
		t.Fatalf("Expire: %v", err)
	}
	if !ok {
		t.Error("expected Expire to return true")
	}

	// Check TTL
	ttl, err := c.TTL(ctx, "temp")
	if err != nil {
		t.Fatalf("TTL: %v", err)
	}
	if ttl <= 0 || ttl > 2 {
		t.Errorf("expected TTL between 1 and 2, got %d", ttl)
	}
}

func TestClientInfo(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	info, err := c.Info(ctx)
	if err != nil {
		t.Fatalf("Info: %v", err)
	}
	if info == "" {
		t.Error("expected non-empty info string")
	}
}

func TestClientVGetAndVDel(t *testing.T) {
	addr, shutdown := startServer(t)
	defer shutdown()

	c, err := New(Options{Addr: addr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()

	ctx := context.Background()

	// Store vector
	vec := []float32{0.5, 0.5, 0.5}
	c.VSet(ctx, VSetArgs{Namespace: "docs", ID: "chunk:1", Vector: vec})

	// VGet
	got, err := c.VGet(ctx, "docs", "chunk:1")
	if err != nil {
		t.Fatalf("VGet: %v", err)
	}
	if len(got) != len(vec) {
		t.Fatalf("expected vector length %d, got %d", len(vec), len(got))
	}

	// VDel
	ok, err := c.VDel(ctx, "docs", "chunk:1")
	if err != nil {
		t.Fatalf("VDel: %v", err)
	}
	if !ok {
		t.Error("expected VDel to return true")
	}

	// VGet missing
	_, err = c.VGet(ctx, "docs", "chunk:1")
	if err != ErrNil {
		t.Errorf("expected ErrNil for missing vector, got %v", err)
	}
}
