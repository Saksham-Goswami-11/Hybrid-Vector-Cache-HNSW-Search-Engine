// server_vns_test.go — Integration tests for the AutoGen-compatible wire protocol commands.
//
// Tests the VSET EX, VMSET, VEXPIRE, VNS DROP, and VNS LIST commands that were
// added to the Nearby wire protocol for Microsoft AutoGen v0.4 multi-agent swarm
// integration. These commands are invoked by the Python client NearbyVectorStore
// (nearby_memory.py) via raw TCP socket.
//
// See AUTOGEN_NEARBY_INTEGRATION_REPORT.md §2.2 for the wire protocol reference.
package server

import (
	"strings"
	"testing"
	"time"
)

func TestVSetWithEXFlag(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// VSET with EX flag
	resp := sendCommand(t, addr, "VSET testns vec1 3 0.1 0.2 0.3 EX 1\n")
	if !strings.Contains(resp, "+OK") {
		t.Fatalf("expected OK, got: %s", resp)
	}

	// Should be retrievable immediately
	resp = sendCommand(t, addr, "VGET testns vec1\n")
	if strings.Contains(resp, "$-1") {
		t.Fatal("VGET should return vector before TTL expires")
	}

	// Wait for expiry
	time.Sleep(1100 * time.Millisecond)

	resp = sendCommand(t, addr, "VGET testns vec1\n")
	if !strings.Contains(resp, "$-1") {
		t.Fatalf("VGET should return nil after TTL, got: %s", resp)
	}
}

func TestVSetWithMetaAndEX(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// VSET with both META and EX
	resp := sendCommand(t, addr, "VSET ns1 chunk1 2 0.5 0.5 META stage researcher EX 60\n")
	if !strings.Contains(resp, "+OK") {
		t.Fatalf("expected OK, got: %s", resp)
	}

	// Verify it was stored
	resp = sendCommand(t, addr, "VCOUNT ns1\n")
	if !strings.Contains(resp, ":1") {
		t.Fatalf("expected count 1, got: %s", resp)
	}
}

func TestVMSETCommand(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// VMSET with 3 entries, dim=2
	resp := sendCommand(t, addr, "VMSET batch-ns 2 3 id1 0.1 0.2 id2 0.3 0.4 id3 0.5 0.6\n")
	if !strings.Contains(resp, ":3") {
		t.Fatalf("expected :3, got: %s", resp)
	}

	// Verify count
	resp = sendCommand(t, addr, "VCOUNT batch-ns\n")
	if !strings.Contains(resp, ":3") {
		t.Fatalf("expected count 3, got: %s", resp)
	}

	// Verify individual retrieval
	resp = sendCommand(t, addr, "VGET batch-ns id2\n")
	if strings.Contains(resp, "$-1") {
		t.Fatal("id2 should be retrievable")
	}
}

func TestVMSETInvalidDimension(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// Not enough floats for second entry
	resp := sendCommand(t, addr, "VMSET ns 3 2 id1 0.1 0.2 0.3 id2 0.4\n")
	if !strings.Contains(resp, "-ERR") {
		t.Fatalf("expected error for dimension mismatch, got: %s", resp)
	}
}

func TestVEXPIRECommand(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// Create vectors
	sendCommand(t, addr, "VSET expire-ns v1 2 1.0 0.0\n")
	sendCommand(t, addr, "VSET expire-ns v2 2 0.0 1.0\n")

	// Set namespace TTL
	resp := sendCommand(t, addr, "VEXPIRE expire-ns 1\n")
	if !strings.Contains(resp, ":1") {
		t.Fatalf("expected :1, got: %s", resp)
	}

	// Non-existent namespace
	resp = sendCommand(t, addr, "VEXPIRE nonexistent 10\n")
	if !strings.Contains(resp, ":0") {
		t.Fatalf("expected :0 for non-existent ns, got: %s", resp)
	}

	// Wait for expiry
	time.Sleep(1100 * time.Millisecond)

	resp = sendCommand(t, addr, "VCOUNT expire-ns\n")
	if !strings.Contains(resp, ":0") {
		t.Fatalf("expected count 0 after VEXPIRE TTL, got: %s", resp)
	}
}

func TestVNSDropCommand(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// Create namespace with vectors
	sendCommand(t, addr, "VSET drop-ns v1 2 1.0 0.0\n")
	sendCommand(t, addr, "VSET drop-ns v2 2 0.0 1.0\n")

	// Drop it
	resp := sendCommand(t, addr, "VNS DROP drop-ns\n")
	if !strings.Contains(resp, ":1") {
		t.Fatalf("expected :1, got: %s", resp)
	}

	// Verify gone
	resp = sendCommand(t, addr, "VCOUNT drop-ns\n")
	if !strings.Contains(resp, ":0") {
		t.Fatalf("expected count 0 after drop, got: %s", resp)
	}

	// Drop again (should fail)
	resp = sendCommand(t, addr, "VNS DROP drop-ns\n")
	if !strings.Contains(resp, ":0") {
		t.Fatalf("expected :0 for already-dropped ns, got: %s", resp)
	}
}

func TestVNSDropCleansHNSWIndex(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// Create vectors and HNSW index
	sendCommand(t, addr, "VSET indexed-ns v1 3 1.0 0.0 0.0\n")
	sendCommand(t, addr, "VSET indexed-ns v2 3 0.0 1.0 0.0\n")
	resp := sendCommand(t, addr, "VINDEX CREATE indexed-ns\n")
	if !strings.Contains(resp, "+OK") {
		t.Fatalf("expected OK for VINDEX CREATE, got: %s", resp)
	}

	// Drop namespace
	resp = sendCommand(t, addr, "VNS DROP indexed-ns\n")
	if !strings.Contains(resp, ":1") {
		t.Fatalf("expected :1, got: %s", resp)
	}

	// HNSW index should be gone
	resp = sendCommand(t, addr, "VINDEX INFO indexed-ns\n")
	if !strings.Contains(resp, "-ERR") {
		t.Fatalf("expected error for dropped index, got: %s", resp)
	}
}

func TestVNSListCommand(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// Empty list
	resp := sendCommand(t, addr, "VNS LIST\n")
	if !strings.Contains(resp, "*0") {
		t.Fatalf("expected empty array, got: %s", resp)
	}

	// Create namespaces
	sendCommand(t, addr, "VSET alpha v1 2 1.0 0.0\n")
	sendCommand(t, addr, "VSET beta v1 2 0.0 1.0\n")
	sendCommand(t, addr, "VSET beta v2 2 0.5 0.5\n")

	resp = sendCommand(t, addr, "VNS LIST\n")
	// Should contain both namespace names
	if !strings.Contains(resp, "alpha") || !strings.Contains(resp, "beta") {
		t.Fatalf("expected both namespaces in response, got: %s", resp)
	}
}

func TestInfoExtended(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// Create some data
	sendCommand(t, addr, "SET mykey myvalue\n")
	sendCommand(t, addr, "VSET info-ns v1 2 1.0 0.0 META stage test\n")

	resp := sendCommand(t, addr, "INFO\n")

	// Should contain version bump
	if !strings.Contains(resp, "version:1.1.0") {
		t.Fatalf("expected version 1.1.0 in INFO, got: %s", resp)
	}

	// Should contain per-namespace stats
	if !strings.Contains(resp, "ns:info-ns:vectors:1") {
		t.Fatalf("expected per-namespace vector count in INFO, got: %s", resp)
	}
	if !strings.Contains(resp, "ns:info-ns:approx_memory_bytes:") {
		t.Fatalf("expected per-namespace memory info in INFO, got: %s", resp)
	}
	if !strings.Contains(resp, "ns:info-ns:ttl:-1") {
		t.Fatalf("expected per-namespace ttl in INFO, got: %s", resp)
	}
}

func TestVEXPIREInvalidArgs(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	// Missing seconds
	resp := sendCommand(t, addr, "VEXPIRE ns-only\n")
	if !strings.Contains(resp, "-ERR") {
		t.Fatalf("expected error for missing seconds, got: %s", resp)
	}

	// Invalid seconds
	resp = sendCommand(t, addr, "VEXPIRE ns abc\n")
	if !strings.Contains(resp, "-ERR") {
		t.Fatalf("expected error for invalid seconds, got: %s", resp)
	}

	// Negative seconds
	resp = sendCommand(t, addr, "VEXPIRE ns -5\n")
	if !strings.Contains(resp, "-ERR") {
		t.Fatalf("expected error for negative seconds, got: %s", resp)
	}
}

func TestVNSUnknownSubcommand(t *testing.T) {
	srv, addr := startTestServer(t)
	defer srv.Shutdown()

	resp := sendCommand(t, addr, "VNS INVALID\n")
	if !strings.Contains(resp, "-ERR") {
		t.Fatalf("expected error for unknown VNS subcommand, got: %s", resp)
	}
}
