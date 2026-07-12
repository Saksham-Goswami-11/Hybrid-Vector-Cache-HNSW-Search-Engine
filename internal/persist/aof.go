package persist

import (
	"bufio"
	"fmt"
	"hash/crc32"
	"log"
	"os"
	"strings"
	"sync"
)

// AOFWriter appends commands to an append-only file.
type AOFWriter struct {
	file *os.File
	mu   sync.Mutex
}

// NewAOFWriter opens (or creates) an AOF file for appending.
func NewAOFWriter(path string) (*AOFWriter, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open AOF file %s: %w", path, err)
	}
	return &AOFWriter{file: f}, nil
}

// Write appends a command line to the AOF with a CRC32 checksum prefix.
func (w *AOFWriter) Write(cmdLine string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	checksum := crc32.ChecksumIEEE([]byte(cmdLine))
	_, err := fmt.Fprintf(w.file, "%08x:%s\n", checksum, cmdLine)
	return err
}

// Sync flushes the AOF to disk.
func (w *AOFWriter) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.file.Sync()
}

// Close closes the AOF file.
func (w *AOFWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.file.Close()
}

// ReplayAOF reads commands from an AOF file line by line, verifying CRC32 checksums.
// It returns the raw command lines. Invalid/empty lines are skipped with a warning.
func ReplayAOF(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // no AOF file — clean start
		}
		return nil, fmt.Errorf("failed to open AOF file %s: %w", path, err)
	}
	defer f.Close()

	var commands []string
	scanner := bufio.NewScanner(f)
	lineNo := 0

	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			log.Printf("aof: warning: invalid format at line %d (skipping)", lineNo)
			continue
		}

		var expectedCRC uint32
		_, err := fmt.Sscanf(parts[0], "%08x", &expectedCRC)
		if err != nil {
			log.Printf("aof: warning: invalid checksum format at line %d (skipping)", lineNo)
			continue
		}

		cmd := parts[1]
		actualCRC := crc32.ChecksumIEEE([]byte(cmd))
		if actualCRC != expectedCRC {
			log.Printf("aof: warning: checksum mismatch at line %d (skipping)", lineNo)
			continue
		}

		commands = append(commands, cmd)
	}

	if err := scanner.Err(); err != nil {
		log.Printf("aof: warning: read error at line %d: %v (partial replay)", lineNo, err)
	}

	return commands, nil
}
