// Stress Test Suite for Nearby Ephemeral Vector Memory Grid
//
// Simulates extreme multi-agent swarm workloads:
// 1. Concurrency Bomb (500–1000 parallel agent goroutines executing VSET & VMSET)
// 2. TTL Sweep Stress Test (Vectors with 2-5s TTL expiring under heavy load)
// 3. Read/Write Collision (Parallel VSIMILARITY queries alongside heavy writes)
// 4. Resource & Memory Footprint Monitoring (Tracks RAM before, during peak, and post-cleanup)
//
// Usage:
//   go run cmd/server/main.go -port 6379 &
//   go run examples/stress_test/main.go -agents 500 -readers 100 -duration 10s -ttl 3s

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	client "github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/pkg/client"
)

type LatencyTracker struct {
	mu        sync.Mutex
	durations []time.Duration
}

func (l *LatencyTracker) Record(d time.Duration) {
	l.mu.Lock()
	l.durations = append(l.durations, d)
	l.mu.Unlock()
}

func (l *LatencyTracker) Percentiles() (p50, p90, p99, max time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if len(l.durations) == 0 {
		return 0, 0, 0, 0
	}

	sorted := make([]time.Duration, len(l.durations))
	copy(sorted, l.durations)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	n := len(sorted)
	p50 = sorted[int(float64(n)*0.50)]
	p90 = sorted[int(float64(n)*0.90)]
	p99 = sorted[int(float64(n)*0.99)]
	max = sorted[n-1]
	return
}

func main() {
	addr := flag.String("addr", "localhost:6379", "Nearby server address")
	numAgents := flag.Int("agents", 500, "Number of concurrent write goroutines (agents)")
	numReaders := flag.Int("readers", 100, "Number of concurrent reader goroutines")
	duration := flag.Duration("duration", 10*time.Second, "Stress test duration")
	ttlSec := flag.Int("ttl", 3, "TTL for expiring vectors in seconds")
	dim := flag.Int("dim", 128, "Vector dimension")
	flag.Parse()

	log.SetFlags(log.Ltime | log.Lmicroseconds)
	fmt.Println("================================================================================")
	fmt.Println("                💣 NEARBY STRESS TEST & SWARM COLLISION SUITE                   ")
	fmt.Println("================================================================================")
	log.Printf("Target:              %s", *addr)
	log.Printf("Writer Goroutines:   %d (Agents)", *numAgents)
	log.Printf("Reader Goroutines:   %d (Rerankers / Searchers)", *numReaders)
	log.Printf("Stress Duration:     %v", *duration)
	log.Printf("Vector TTL:          %ds", *ttlSec)
	log.Printf("Vector Dimension:    %d", *dim)
	fmt.Println("================================================================================")

	// Step 0: Connectivity Check & Initial Memory
	ctx := context.Background()
	initialClient, err := client.New(client.Options{Addr: *addr, MaxConns: 5, DialTimeout: 5 * time.Second})
	if err != nil {
		log.Fatalf("❌ Failed to connect to Nearby server at %s: %v", *addr, err)
	}
	defer initialClient.Close()

	log.Println("✅ Connection verified.")
	initialInfo, _ := initialClient.Info(ctx)
	log.Printf("📊 Initial Server State:\n%s", indent(initialInfo))

	// Global counters & latency tracking
	var totalWrites uint64
	var totalBatchWrites uint64
	var totalVectorsIngested uint64
	var totalReads uint64
	var totalErrors uint64

	writeLatencies := &LatencyTracker{}
	batchWriteLatencies := &LatencyTracker{}
	readLatencies := &LatencyTracker{}

	stopCh := make(chan struct{})
	var wg sync.WaitGroup

	startTime := time.Now()

	// -------------------------------------------------------------------------
	// 1 & 2: CONCURRENCY BOMB & TTL SWEEP (Writers)
	// -------------------------------------------------------------------------
	log.Printf("🚀 Launching %d Writer Agent Goroutines (VSET & VMSET with %ds TTL)...", *numAgents, *ttlSec)

	for i := 0; i < *numAgents; i++ {
		wg.Add(1)
		go func(agentID int) {
			defer wg.Done()

			// Each agent gets a pooled client connection
			c, err := client.New(client.Options{Addr: *addr, MaxConns: 2, DialTimeout: 5 * time.Second})
			if err != nil {
				atomic.AddUint64(&totalErrors, 1)
				return
			}
			defer c.Close()

			ns := fmt.Sprintf("swarm:agent-%d", agentID)

			// Optionally set namespace-level TTL for half of the agents
			if agentID%2 == 0 {
				_, _ = c.RawExec(ctx, fmt.Sprintf("VEXPIRE %s %d", ns, *ttlSec))
			}

			localRand := rand.New(rand.NewSource(time.Now().UnixNano() + int64(agentID)))
			vec := make([]float32, *dim)

			for {
				select {
				case <-stopCh:
					return
				default:
				}

				// Generate random unit vector
				generateUnitVector(vec, localRand)

				if localRand.Float32() < 0.7 {
					// 70% VSET operations (Single vector write with EX flag)
					id := fmt.Sprintf("vec:%d", localRand.Intn(10000))
					cmd := fmt.Sprintf("VSET %s %s %d", ns, id, *dim)
					for _, f := range vec {
						cmd += fmt.Sprintf(" %f", f)
					}
					cmd += fmt.Sprintf(" META stage researcher agent %d EX %d", agentID, *ttlSec)

					t0 := time.Now()
					_, err := c.RawExec(ctx, cmd)
					elapsed := time.Since(t0)

					if err != nil {
						atomic.AddUint64(&totalErrors, 1)
					} else {
						atomic.AddUint64(&totalWrites, 1)
						atomic.AddUint64(&totalVectorsIngested, 1)
						writeLatencies.Record(elapsed)
					}

				} else {
					// 30% VMSET operations (Batch vector write)
					batchSize := 10 + localRand.Intn(20) // 10 to 30 vectors per batch
					cmd := fmt.Sprintf("VMSET %s %d %d", ns, *dim, batchSize)

					for b := 0; b < batchSize; b++ {
						generateUnitVector(vec, localRand)
						id := fmt.Sprintf("batchVec:%d", localRand.Intn(100000))
						cmd += fmt.Sprintf(" %s", id)
						for _, f := range vec {
							cmd += fmt.Sprintf(" %f", f)
						}
					}

					t0 := time.Now()
					_, err := c.RawExec(ctx, cmd)
					elapsed := time.Since(t0)

					if err != nil {
						atomic.AddUint64(&totalErrors, 1)
					} else {
						atomic.AddUint64(&totalBatchWrites, 1)
						atomic.AddUint64(&totalVectorsIngested, uint64(batchSize))
						batchWriteLatencies.Record(elapsed)
					}
				}

				// Minimal yield to prevent OS-level TCP socket starvation
				time.Sleep(100 * time.Microsecond)
			}
		}(i)
	}

	// -------------------------------------------------------------------------
	// 3: READ/WRITE COLLISION SIMULATION (Readers)
	// -------------------------------------------------------------------------
	log.Printf("🔍 Launching %d Reader Goroutines (Parallel VSIMILARITY Queries)...", *numReaders)

	for i := 0; i < *numReaders; i++ {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()

			c, err := client.New(client.Options{Addr: *addr, MaxConns: 2, DialTimeout: 5 * time.Second})
			if err != nil {
				atomic.AddUint64(&totalErrors, 1)
				return
			}
			defer c.Close()

			localRand := rand.New(rand.NewSource(time.Now().UnixNano() + int64(readerID+10000)))
			queryVec := make([]float32, *dim)

			for {
				select {
				case <-stopCh:
					return
				default:
				}

				generateUnitVector(queryVec, localRand)
				targetAgent := localRand.Intn(*numAgents)
				ns := fmt.Sprintf("swarm:agent-%d", targetAgent)

				t0 := time.Now()
				_, err := c.VSimilarity(ctx, client.VSimilarityArgs{
					Namespace: ns,
					Vector:    queryVec,
					TopK:      5,
				})
				elapsed := time.Since(t0)

				if err != nil && !isNilErr(err) {
					atomic.AddUint64(&totalErrors, 1)
				} else {
					atomic.AddUint64(&totalReads, 1)
					readLatencies.Record(elapsed)
				}

				time.Sleep(200 * time.Microsecond)
			}
		}(i)
	}

	// Run stress duration
	log.Printf("⚡ Stress test running for %v...", *duration)
	time.Sleep(*duration)

	// Signal workers to stop
	close(stopCh)
	wg.Wait()
	totalDuration := time.Since(startTime)

	// -------------------------------------------------------------------------
	// 4: RESULTS ANALYSIS & RESOURCE MONITORING
	// -------------------------------------------------------------------------
	fmt.Println()
	fmt.Println("================================================================================")
	fmt.Println("                          📊 STRESS TEST RESULTS                                ")
	fmt.Println("================================================================================")

	wOps := atomic.LoadUint64(&totalWrites)
	bOps := atomic.LoadUint64(&totalBatchWrites)
	vIngested := atomic.LoadUint64(&totalVectorsIngested)
	rOps := atomic.LoadUint64(&totalReads)
	errOps := atomic.LoadUint64(&totalErrors)

	totalOps := wOps + bOps + rOps
	opsPerSec := float64(totalOps) / totalDuration.Seconds()
	vecPerSec := float64(vIngested) / totalDuration.Seconds()

	log.Printf("Total Elapsed Time:        %v", totalDuration.Round(time.Millisecond))
	log.Printf("Total Requests Processed:  %d", totalOps)
	log.Printf("Overall Throughput:        %.2f ops/sec", opsPerSec)
	log.Printf("Vector Ingestion Throughput: %.2f vectors/sec", vecPerSec)
	log.Printf("Total Errors Encountered:  %d", errOps)
	fmt.Println("--------------------------------------------------------------------------------")
	log.Printf("Single Writes (VSET):      %d ops", wOps)
	log.Printf("Batch Writes (VMSET):     %d ops", bOps)
	log.Printf("Total Vectors Ingested:   %d vectors", vIngested)
	log.Printf("Similarity Queries (VSIMILARITY): %d ops", rOps)

	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Println("                      ⏱️  LATENCY PERCENTILE BREAKDOWN                         ")
	fmt.Println("--------------------------------------------------------------------------------")

	wP50, wP90, wP99, wMax := writeLatencies.Percentiles()
	bP50, bP90, bP99, bMax := batchWriteLatencies.Percentiles()
	rP50, rP90, rP99, rMax := readLatencies.Percentiles()

	log.Printf("VSET Latency:         P50: %-8v | P90: %-8v | P99: %-8v | Max: %-8v", wP50, wP90, wP99, wMax)
	log.Printf("VMSET Latency:        P50: %-8v | P90: %-8v | P99: %-8v | Max: %-8v", bP50, bP90, bP99, bMax)
	log.Printf("VSIMILARITY Latency:  P50: %-8v | P90: %-8v | P99: %-8v | Max: %-8v", rP50, rP90, rP99, rMax)

	fmt.Println("================================================================================")
	fmt.Println("                🧹 TTL EXPIRATION & RECLAMATION AUDIT                         ")
	fmt.Println("================================================================================")

	peakInfo, _ := initialClient.Info(ctx)
	log.Printf("📊 Peak Load Server State:\n%s", indent(peakInfo))

	// Wait for TTL expiration sweep
	waitDuration := time.Duration(*ttlSec+2) * time.Second
	log.Printf("⏳ Waiting %v for TTL sweep and background reclamation...", waitDuration)
	time.Sleep(waitDuration)

	postTTLInfo, _ := initialClient.Info(ctx)
	log.Printf("📊 Post-TTL Server State:\n%s", indent(postTTLInfo))

	// Explicit Teardown of any remaining namespaces
	log.Println("🗑️  Cleaning up remaining namespaces via VNS LIST + VNS DROP...")
	nsResp, err := initialClient.RawExec(ctx, "VNS LIST")
	if err == nil && len(nsResp.Array) > 0 {
		dropped := 0
		for i := 0; i+3 < len(nsResp.Array); i += 4 {
			nsName := nsResp.Array[i].Str
			_, _ = initialClient.RawExec(ctx, fmt.Sprintf("VNS DROP %s", nsName))
			dropped++
		}
		log.Printf("✅ Explicitly dropped %d active namespaces.", dropped)
	}

	finalInfo, _ := initialClient.Info(ctx)
	log.Printf("📊 Final Baseline Server State:\n%s", indent(finalInfo))

	fmt.Println("================================================================================")
	if errOps == 0 {
		fmt.Println("🎉 STRESS TEST COMPLETED SUCCESSFULLY WITH ZERO ERRORS AND ZERO DEADLOCKS!")
	} else {
		fmt.Printf("⚠️ STRESS TEST COMPLETED WITH %d ERRORS.\n", errOps)
	}
	fmt.Println("================================================================================")
}

func generateUnitVector(vec []float32, rng *rand.Rand) {
	var norm float64
	for i := range vec {
		val := rng.NormFloat64()
		vec[i] = float32(val)
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= float32(norm)
		}
	}
}

func isNilErr(err error) bool {
	return err != nil && (err == client.ErrNil || strings.Contains(err.Error(), "nil"))
}

func indent(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	for i, line := range lines {
		lines[i] = "    " + line
	}
	return strings.Join(lines, "\n")
}
