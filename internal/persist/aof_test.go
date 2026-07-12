package persist

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAOFWriteAndReplay(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.aof")

	// Write commands
	w, err := NewAOFWriter(path)
	if err != nil {
		t.Fatalf("NewAOFWriter: %v", err)
	}

	commands := []string{
		"SET key1 val1",
		"SET key2 val2 EX 100",
		"VSET docs chunk:1 2 0.5 0.5",
		"DEL key1",
	}
	for _, cmd := range commands {
		if err := w.Write(cmd); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}
	w.Sync()
	w.Close()

	// Replay
	replayed, err := ReplayAOF(path)
	if err != nil {
		t.Fatalf("ReplayAOF: %v", err)
	}

	if len(replayed) != len(commands) {
		t.Fatalf("expected %d commands, got %d", len(commands), len(replayed))
	}
	for i, cmd := range commands {
		if replayed[i] != cmd {
			t.Errorf("command %d: expected %q, got %q", i, cmd, replayed[i])
		}
	}
}

func TestReplayMissingFile(t *testing.T) {
	cmds, err := ReplayAOF("/nonexistent/path/test.aof")
	if err != nil {
		t.Fatalf("expected nil error for missing file, got %v", err)
	}
	if cmds != nil {
		t.Errorf("expected nil commands, got %v", cmds)
	}
}

func TestReplayTruncatedFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.aof")

	w, err := NewAOFWriter(path)
	if err != nil {
		t.Fatalf("NewAOFWriter: %v", err)
	}
	w.Write("SET key1 val1")
	w.Write("SET key2 val2")
	w.Close()

	// Append corrupt/truncated data
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0644)
	f.WriteString("deadbeef:SET trun") // bad CRC and truncated
	f.Close()

	cmds, err := ReplayAOF(path)
	if err != nil {
		t.Fatalf("ReplayAOF: %v", err)
	}
	if len(cmds) != 2 {
		t.Errorf("expected 2 commands, got %d", len(cmds))
	}
}

func TestAOFConcurrentWrites(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.aof")

	w, err := NewAOFWriter(path)
	if err != nil {
		t.Fatalf("NewAOFWriter: %v", err)
	}
	defer w.Close()

	// Write from multiple goroutines
	done := make(chan struct{})
	for i := 0; i < 10; i++ {
		go func(id int) {
			defer func() { done <- struct{}{} }()
			for j := 0; j < 100; j++ {
				w.Write("SET key val")
			}
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}
	w.Sync()

	cmds, _ := ReplayAOF(path)
	if len(cmds) != 1000 {
		t.Errorf("expected 1000 commands, got %d", len(cmds))
	}
}
