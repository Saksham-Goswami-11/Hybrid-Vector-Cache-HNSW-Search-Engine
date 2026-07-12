package client

import (
	"context"
	"strings"
	"testing"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/server"
	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
)

func startAuthServer(t *testing.T, password string) (string, func()) {
	t.Helper()
	s := store.New()
	srv := server.New(":0", password, false, "", "", s)

	go srv.ListenAndServe()
	srv.WaitReady()

	return srv.Addr(), func() { srv.Shutdown() }
}

func TestClientAuthSuccess(t *testing.T) {
	addr, shutdown := startAuthServer(t, "secret123")
	defer shutdown()

	c, err := New(Options{Addr: addr, Password: "secret123"})
	if err != nil {
		t.Fatalf("expected connect to succeed, got %v", err)
	}
	defer c.Close()

	if err := c.Ping(context.Background()); err != nil {
		t.Errorf("Ping failed: %v", err)
	}
}

func TestClientAuthFailure(t *testing.T) {
	addr, shutdown := startAuthServer(t, "secret123")
	defer shutdown()

	_, err := New(Options{Addr: addr, Password: "wrong"})
	if err == nil {
		t.Fatal("expected connect to fail with wrong password, but succeeded")
	}
	if !strings.Contains(err.Error(), "ERR invalid password") {
		t.Errorf("expected ERR invalid password, got %v", err)
	}
}

func TestClientNoAuthWhenRequired(t *testing.T) {
	addr, shutdown := startAuthServer(t, "secret123")
	defer shutdown()

	// Connect without password. dial() succeeds because TCP connects.
	// But the first command should fail with NOAUTH.
	c, err := New(Options{Addr: addr, Password: ""})
	if err != nil {
		t.Fatalf("dial should succeed without password, but got: %v", err)
	}
	defer c.Close()

	err = c.Ping(context.Background())
	if err == nil {
		t.Fatal("expected ping to fail due to NOAUTH")
	}
	if !strings.Contains(err.Error(), "ERR NOAUTH") {
		t.Errorf("expected ERR NOAUTH, got %v", err)
	}
}
