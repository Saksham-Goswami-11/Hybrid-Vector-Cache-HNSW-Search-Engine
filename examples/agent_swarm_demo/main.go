// agent_swarm_demo demonstrates Nearby as an ephemeral vector memory grid
// for multi-agent architectures.
//
// It simulates a 3-agent pipeline:
//   1. Researcher agent → writes document embeddings to a shared namespace
//   2. Coder agent → queries similar vectors to find relevant context
//   3. Orchestrator → tears down the namespace when the run completes
//
// Prerequisites:
//   go run cmd/server/main.go -port 6379
//
// Usage:
//   go run examples/agent_swarm_demo/main.go

package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
	"time"

	client "github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/pkg/client"
)

const (
	nearbyAddr = "localhost:6379"
	namespace  = "swarm:run-42"
	dim        = 128 // embedding dimension
	ttlSeconds = 300 // 5 minute TTL
)

func main() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)
	log.Println("═══════════════════════════════════════════════════════")
	log.Println("  Nearby — Agent Swarm Demo (Memory Grid)")
	log.Println("═══════════════════════════════════════════════════════")

	ctx := context.Background()

	// Connect to Nearby
	c, err := client.New(client.Options{
		Addr:        nearbyAddr,
		MaxConns:    4,
		DialTimeout: 3 * time.Second,
	})
	if err != nil {
		log.Fatalf("❌ Failed to connect to Nearby at %s: %v", nearbyAddr, err)
	}
	defer c.Close()

	log.Println("✅ Connected to Nearby")

	// Show server info
	info, _ := c.Info(ctx)
	log.Printf("📊 Server Info:\n%s", indent(info))

	fmt.Println()

	// ──────────────────────────────────────────────
	// Phase 1: Researcher Agent writes embeddings
	// ──────────────────────────────────────────────
	log.Println("🔬 [Researcher Agent] Starting document processing...")

	documents := []struct {
		id       string
		topic    string
		category string
	}{
		{"chunk:doc1-p1", "machine learning fundamentals", "ml"},
		{"chunk:doc1-p2", "neural network architectures", "ml"},
		{"chunk:doc2-p1", "kubernetes deployment patterns", "infra"},
		{"chunk:doc2-p2", "container orchestration", "infra"},
		{"chunk:doc3-p1", "vector similarity search", "search"},
		{"chunk:doc3-p2", "approximate nearest neighbors", "search"},
		{"chunk:doc4-p1", "agent communication protocols", "agents"},
		{"chunk:doc4-p2", "multi-agent coordination", "agents"},
		{"chunk:doc5-p1", "embedding models comparison", "ml"},
		{"chunk:doc5-p2", "transformer attention mechanisms", "ml"},
	}

	// Generate pseudo-embeddings based on topic (vectors in the same category
	// will be more similar to each other)
	categorySeeds := map[string][]float32{
		"ml":     generateSeedVector(dim, 42),
		"infra":  generateSeedVector(dim, 99),
		"search": generateSeedVector(dim, 7),
		"agents": generateSeedVector(dim, 13),
	}

	writeStart := time.Now()
	for _, doc := range documents {
		vec := perturbVector(categorySeeds[doc.category], 0.1)

		// VSET with TTL and metadata
		cmd := fmt.Sprintf("VSET %s %s %d", namespace, doc.id, dim)
		for _, f := range vec {
			cmd += fmt.Sprintf(" %f", f)
		}
		cmd += fmt.Sprintf(" META stage researcher topic \"%s\" category %s EX %d",
			doc.topic, doc.category, ttlSeconds)

		_, err := c.RawExec(ctx, cmd)
		if err != nil {
			log.Fatalf("❌ Researcher failed to write %s: %v", doc.id, err)
		}
	}
	writeDuration := time.Since(writeStart)

	log.Printf("✅ [Researcher] Wrote %d embeddings in %v (%.2f µs/vector)",
		len(documents), writeDuration, float64(writeDuration.Microseconds())/float64(len(documents)))

	// Check count
	count, _ := c.VCount(ctx, namespace)
	log.Printf("📊 [Researcher] Namespace '%s' now has %d vectors", namespace, count)

	fmt.Println()

	// ──────────────────────────────────────────────
	// Phase 2: Coder Agent queries for relevant context
	// ──────────────────────────────────────────────
	log.Println("💻 [Coder Agent] Searching for relevant context...")

	queries := []struct {
		name     string
		category string
	}{
		{"How do neural networks work?", "ml"},
		{"Deploy containers to production", "infra"},
		{"Find similar vectors efficiently", "search"},
	}

	for _, q := range queries {
		queryVec := perturbVector(categorySeeds[q.category], 0.05)

		searchStart := time.Now()
		results, err := c.VSimilarity(ctx, client.VSimilarityArgs{
			Namespace: namespace,
			Vector:    queryVec,
			TopK:      3,
		})
		searchDuration := time.Since(searchStart)

		if err != nil {
			log.Printf("❌ [Coder] Query '%s' failed: %v", q.name, err)
			continue
		}

		log.Printf("🔍 [Coder] Query: \"%s\" (%v)", q.name, searchDuration)
		for i, r := range results {
			topic := r.Metadata["topic"]
			if topic == "" {
				topic = "(no metadata)"
			}
			log.Printf("   %d. %s (score: %.4f) — topic: %s",
				i+1, r.ID, r.Score, topic)
		}
	}

	fmt.Println()

	// ──────────────────────────────────────────────
	// Phase 3: Batch ingestion (reranker staging)
	// ──────────────────────────────────────────────
	log.Println("📦 [Reranker] Batch-loading candidate embeddings...")

	rerankNS := "rerank:req-9182"
	batchCount := 50
	batchCmd := fmt.Sprintf("VMSET %s %d %d", rerankNS, dim, batchCount)
	for i := 0; i < batchCount; i++ {
		batchCmd += fmt.Sprintf(" candidate:%d", i)
		vec := generateSeedVector(dim, int64(i*7+3))
		for _, f := range vec {
			batchCmd += fmt.Sprintf(" %f", f)
		}
	}

	batchStart := time.Now()
	resp, err := c.RawExec(ctx, batchCmd)
	batchDuration := time.Since(batchStart)

	if err != nil {
		log.Printf("❌ [Reranker] VMSET failed: %v", err)
	} else {
		log.Printf("✅ [Reranker] Batch-loaded %s vectors in %v (%.2f µs/vector)",
			resp.Str, batchDuration, float64(batchDuration.Microseconds())/float64(batchCount))
	}

	fmt.Println()

	// ──────────────────────────────────────────────
	// Phase 4: Namespace lifecycle management
	// ──────────────────────────────────────────────
	log.Println("🎛️  [Orchestrator] Managing namespace lifecycle...")

	// Set namespace-level TTL on the reranker namespace
	_, err = c.RawExec(ctx, fmt.Sprintf("VEXPIRE %s 60", rerankNS))
	if err != nil {
		log.Printf("❌ VEXPIRE failed: %v", err)
	} else {
		log.Printf("⏰ [Orchestrator] Set 60s TTL on namespace '%s'", rerankNS)
	}

	// List all namespaces
	nsResp, err := c.RawExec(ctx, "VNS LIST")
	if err != nil {
		log.Printf("❌ VNS LIST failed: %v", err)
	} else {
		log.Println("📋 [Orchestrator] Active namespaces:")
		// Parse array: [name, count, memory, ttl, name, count, memory, ttl, ...]
		for i := 0; i+3 < len(nsResp.Array); i += 4 {
			name := nsResp.Array[i].Str
			vCount := nsResp.Array[i+1].Str
			mem := nsResp.Array[i+2].Str
			ttl := nsResp.Array[i+3].Str
			log.Printf("   • %s — vectors: %s, memory: %s bytes, ttl: %ss",
				name, vCount, mem, ttl)
		}
	}

	// Show enriched INFO
	info, _ = c.Info(ctx)
	log.Printf("📊 Server Info (enriched):\n%s", indent(info))

	fmt.Println()

	// ──────────────────────────────────────────────
	// Phase 5: Orchestrator tears down completed run
	// ──────────────────────────────────────────────
	log.Println("🗑️  [Orchestrator] Run complete — tearing down namespaces...")

	// Drop swarm namespace
	dropResp, err := c.RawExec(ctx, fmt.Sprintf("VNS DROP %s", namespace))
	if err != nil {
		log.Printf("❌ VNS DROP failed: %v", err)
	} else {
		log.Printf("✅ [Orchestrator] Dropped '%s' (result: %s)", namespace, dropResp.Str)
	}

	// Drop reranker namespace
	dropResp, err = c.RawExec(ctx, fmt.Sprintf("VNS DROP %s", rerankNS))
	if err != nil {
		log.Printf("❌ VNS DROP failed: %v", err)
	} else {
		log.Printf("✅ [Orchestrator] Dropped '%s' (result: %s)", rerankNS, dropResp.Str)
	}

	// Verify cleanup
	nsResp, _ = c.RawExec(ctx, "VNS LIST")
	log.Printf("📋 [Orchestrator] Remaining namespaces: %d", len(nsResp.Array)/4)

	info, _ = c.Info(ctx)
	log.Printf("📊 Final server state:\n%s", indent(info))

	fmt.Println()
	log.Println("═══════════════════════════════════════════════════════")
	log.Println("  Demo complete! Memory returned to baseline.")
	log.Println("═══════════════════════════════════════════════════════")
}

// generateSeedVector creates a deterministic unit vector for a category.
func generateSeedVector(dim int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	vec := make([]float32, dim)
	var norm float64
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
		norm += float64(vec[i] * vec[i])
	}
	// Normalize
	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] /= float32(norm)
	}
	return vec
}

// perturbVector adds small random noise to a seed vector to simulate
// different documents in the same category.
func perturbVector(seed []float32, noise float64) []float32 {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	vec := make([]float32, len(seed))
	var norm float64
	for i := range vec {
		vec[i] = seed[i] + float32(rng.NormFloat64()*noise)
		norm += float64(vec[i] * vec[i])
	}
	// Re-normalize
	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] /= float32(norm)
	}
	return vec
}

// indent adds 4-space indentation to each line.
func indent(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	for i, line := range lines {
		lines[i] = "    " + line
	}
	return strings.Join(lines, "\n")
}
