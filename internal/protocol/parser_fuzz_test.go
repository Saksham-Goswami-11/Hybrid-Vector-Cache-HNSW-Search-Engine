package protocol

import (
	"bufio"
	"bytes"
	"testing"
)

// FuzzParse feeds random byte streams into protocol.Parse and verifies it never panics.
func FuzzParse(f *testing.F) {
	// Seed corpus with valid and edge-case RESP lines
	f.Add([]byte("PING\r\n"))
	f.Add([]byte("SET key \"hello world\"\r\n"))
	f.Add([]byte("VSET ns v1 4 0.1 0.2 0.3 0.4 META k v EX 60\r\n"))
	f.Add([]byte("VSIMILARITY ns 4 0.1 0.2 0.3 0.4 TOP 5\r\n"))
	f.Add([]byte("SET k \"escaped \\\" quote\"\r\n"))
	f.Add([]byte("INVALID_COMMAND\r\n"))

	f.Fuzz(func(t *testing.T, data []byte) {
		reader := bufio.NewReader(bytes.NewReader(data))
		// Parse should return either valid commands or an error, but must never panic.
		_, _ = Parse(reader)
	})
}

// FuzzTokenize feeds random strings into tokenize and verifies it never panics or deadlocks.
func FuzzTokenize(f *testing.F) {
	f.Add("SET key \"value with spaces\"")
	f.Add("VSET ns key 2 1.0 2.0 META \\\"quoted\\\" val")
	f.Add("   spaced   tokens   'single quoted'   ")
	f.Add("\\\\\\\"\\n\\r\\t")

	f.Fuzz(func(t *testing.T, line string) {
		_ = tokenize(line)
	})
}
