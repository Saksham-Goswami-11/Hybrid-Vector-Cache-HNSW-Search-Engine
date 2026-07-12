package server

import (
	"bufio"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
)

// startTestServer starts a server on a random port and returns its address.
func startTestServer(t *testing.T) (*Server, string) {
	t.Helper()
	s := store.New()
	srv := New(":0", "", false, "", "", s) // port 0 = OS picks a random free port

	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.ListenAndServe()
	}()

	// Wait for the listener to bind (race-free synchronization)
	srv.WaitReady()

	select {
	case err := <-errCh:
		t.Fatalf("server failed to start: %v", err)
	default:
	}

	return srv, srv.Addr()
}

// sendCommand dials the server, sends a raw command, and returns the response.
func sendCommand(t *testing.T, addr, command string) string {
	t.Helper()

	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	conn.SetDeadline(time.Now().Add(2 * time.Second))

	fmt.Fprint(conn, command)

	reader := bufio.NewReader(conn)
	var response strings.Builder

	// Read all available response lines
	for {
		conn.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
		line, err := reader.ReadString('\n')
		response.WriteString(line)
		if err != nil {
			break
		}
	}

	return response.String()
}

// sendCommands sends multiple commands on a single connection.
func sendCommands(t *testing.T, addr string, commands []string) []string {
	t.Helper()

	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	conn.SetDeadline(time.Now().Add(2 * time.Second))

	for _, cmd := range commands {
		fmt.Fprint(conn, cmd)
	}

	reader := bufio.NewReader(conn)
	var responses []string

	for range commands {
		conn.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		responses = append(responses, line)
	}

	return responses
}

func TestPingPong(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	resp := sendCommand(t, addr, "PING\r\n")
	if !strings.HasPrefix(resp, "+PONG") {
		t.Errorf("expected +PONG, got %q", resp)
	}
}

func TestPingWithMessage(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	resp := sendCommand(t, addr, "PING hello world\r\n")
	if !strings.HasPrefix(resp, "+hello world") {
		t.Errorf("expected +hello world, got %q", resp)
	}
}

func TestSetGet(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	reader := bufio.NewReader(conn)

	// SET
	fmt.Fprint(conn, "SET mykey myvalue\r\n")
	line, _ := reader.ReadString('\n')
	if !strings.HasPrefix(line, "+OK") {
		t.Errorf("SET: expected +OK, got %q", line)
	}

	// GET
	fmt.Fprint(conn, "GET mykey\r\n")
	line, _ = reader.ReadString('\n')
	if !strings.HasPrefix(line, "$7") {
		t.Errorf("GET: expected $7, got %q", line)
	}
	line, _ = reader.ReadString('\n')
	if !strings.HasPrefix(line, "myvalue") {
		t.Errorf("GET: expected myvalue, got %q", line)
	}
}

func TestGetMissing(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	resp := sendCommand(t, addr, "GET nonexistent\r\n")
	if !strings.HasPrefix(resp, "$-1") {
		t.Errorf("expected $-1 (nil), got %q", resp)
	}
}

func TestDel(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	reader := bufio.NewReader(conn)

	// SET two keys
	fmt.Fprint(conn, "SET key1 val1\r\n")
	reader.ReadString('\n')
	fmt.Fprint(conn, "SET key2 val2\r\n")
	reader.ReadString('\n')

	// DEL both
	fmt.Fprint(conn, "DEL key1 key2\r\n")
	line, _ := reader.ReadString('\n')
	if !strings.HasPrefix(line, ":2") {
		t.Errorf("DEL: expected :2, got %q", line)
	}

	// GET should return nil
	fmt.Fprint(conn, "GET key1\r\n")
	line, _ = reader.ReadString('\n')
	if !strings.HasPrefix(line, "$-1") {
		t.Errorf("GET after DEL: expected $-1, got %q", line)
	}
}

func TestUnknownCommand(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	resp := sendCommand(t, addr, "FOOBAR\r\n")
	if !strings.Contains(resp, "-ERR") {
		t.Errorf("expected error for unknown command, got %q", resp)
	}
}

func TestSetWithTTL(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	reader := bufio.NewReader(conn)

	// SET with 1 second TTL
	fmt.Fprint(conn, "SET tempkey tempval EX 1\r\n")
	line, _ := reader.ReadString('\n')
	if !strings.HasPrefix(line, "+OK") {
		t.Errorf("SET EX: expected +OK, got %q", line)
	}

	// GET should succeed immediately
	fmt.Fprint(conn, "GET tempkey\r\n")
	line, _ = reader.ReadString('\n')
	if !strings.HasPrefix(line, "$7") {
		t.Errorf("GET before expiry: expected value, got %q", line)
	}
	reader.ReadString('\n') // read the value line

	// Wait for expiry
	time.Sleep(1100 * time.Millisecond)

	// GET should return nil
	fmt.Fprint(conn, "GET tempkey\r\n")
	line, _ = reader.ReadString('\n')
	if !strings.HasPrefix(line, "$-1") {
		t.Errorf("GET after expiry: expected $-1, got %q", line)
	}
}

func TestExpireCommand(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	reader := bufio.NewReader(conn)

	// SET
	fmt.Fprint(conn, "SET mykey myvalue\r\n")
	reader.ReadString('\n')

	// EXPIRE
	fmt.Fprint(conn, "EXPIRE mykey 10\r\n")
	line, _ := reader.ReadString('\n')
	if !strings.HasPrefix(line, ":1") {
		t.Errorf("EXPIRE: expected :1, got %q", line)
	}

	// EXPIRE on nonexistent key
	fmt.Fprint(conn, "EXPIRE nokey 10\r\n")
	line, _ = reader.ReadString('\n')
	if !strings.HasPrefix(line, ":0") {
		t.Errorf("EXPIRE missing key: expected :0, got %q", line)
	}
}

func TestTTLCommand(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	reader := bufio.NewReader(conn)

	// TTL on nonexistent key → -2
	fmt.Fprint(conn, "TTL nokey\r\n")
	line, _ := reader.ReadString('\n')
	if !strings.HasPrefix(line, ":-2") {
		t.Errorf("TTL missing key: expected :-2, got %q", line)
	}

	// SET without TTL → TTL returns -1
	fmt.Fprint(conn, "SET mykey myvalue\r\n")
	reader.ReadString('\n')

	fmt.Fprint(conn, "TTL mykey\r\n")
	line, _ = reader.ReadString('\n')
	if !strings.HasPrefix(line, ":-1") {
		t.Errorf("TTL no expiry: expected :-1, got %q", line)
	}

	// SET with TTL → TTL returns positive number
	fmt.Fprint(conn, "SET ttlkey ttlval EX 100\r\n")
	reader.ReadString('\n')

	fmt.Fprint(conn, "TTL ttlkey\r\n")
	line, _ = reader.ReadString('\n')
	if strings.HasPrefix(line, ":-") {
		t.Errorf("TTL with expiry: expected positive, got %q", line)
	}
}
