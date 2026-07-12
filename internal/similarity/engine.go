package similarity

import (
	"container/heap"
	"runtime"
	"sync"

	"github.com/sakshamgoswami/Hybrid-Vector-Cache-HNSW-Search-Engine/internal/store"
)

// SimilarityResult holds a single similarity search result.
type SimilarityResult struct {
	ID       string
	Score    float32
	Metadata map[string]string
}

// Engine orchestrates top-K similarity search using a worker pool.
type Engine struct {
	Workers int
}

// NewEngine creates a similarity engine with the given worker count.
// If workers <= 0, defaults to runtime.GOMAXPROCS(0).
func NewEngine(workers int) *Engine {
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	return &Engine{Workers: workers}
}

// TopK finds the K most similar vectors to the query vector from the given entries.
// Results are returned sorted in descending order of similarity score.
func (e *Engine) TopK(query []float32, entries []*store.VectorEntry, k int) []SimilarityResult {
	if len(entries) == 0 || k <= 0 {
		return nil
	}
	if k > len(entries) {
		k = len(entries)
	}

	// For small entry counts, don't bother with a worker pool
	if len(entries) <= 100 || e.Workers <= 1 {
		return e.topKSingleThread(query, entries, k)
	}

	return e.topKParallel(query, entries, k)
}

// topKSingleThread runs the search on the calling goroutine.
func (e *Engine) topKSingleThread(query []float32, entries []*store.VectorEntry, k int) []SimilarityResult {
	h := &minHeap{}
	heap.Init(h)

	for _, entry := range entries {
		score, err := CosineSimilarity(query, entry.Vector)
		if err != nil {
			continue // skip dimension mismatches and zero vectors
		}

		if h.Len() < k {
			heap.Push(h, SimilarityResult{
				ID:       entry.ID,
				Score:    score,
				Metadata: entry.Metadata,
			})
		} else if score > (*h)[0].Score {
			(*h)[0] = SimilarityResult{
				ID:       entry.ID,
				Score:    score,
				Metadata: entry.Metadata,
			}
			heap.Fix(h, 0)
		}
	}

	// Extract results in descending order
	results := make([]SimilarityResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SimilarityResult)
	}
	return results
}

// topKParallel fans out similarity computation across a worker pool.
func (e *Engine) topKParallel(query []float32, entries []*store.VectorEntry, k int) []SimilarityResult {
	type scored struct {
		entry *store.VectorEntry
		score float32
	}

	resultsCh := make(chan scored, len(entries))
	chunkSize := (len(entries) + e.Workers - 1) / e.Workers

	var wg sync.WaitGroup

	for i := 0; i < len(entries); i += chunkSize {
		end := i + chunkSize
		if end > len(entries) {
			end = len(entries)
		}
		chunk := entries[i:end]

		wg.Add(1)
		go func(chunk []*store.VectorEntry) {
			defer wg.Done()
			for _, entry := range chunk {
				score, err := CosineSimilarity(query, entry.Vector)
				if err != nil {
					continue
				}
				resultsCh <- scored{entry: entry, score: score}
			}
		}(chunk)
	}

	// Close channel when all workers are done
	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	// Collect into a min-heap of size K
	h := &minHeap{}
	heap.Init(h)

	for s := range resultsCh {
		if h.Len() < k {
			heap.Push(h, SimilarityResult{
				ID:       s.entry.ID,
				Score:    s.score,
				Metadata: s.entry.Metadata,
			})
		} else if s.score > (*h)[0].Score {
			(*h)[0] = SimilarityResult{
				ID:       s.entry.ID,
				Score:    s.score,
				Metadata: s.entry.Metadata,
			}
			heap.Fix(h, 0)
		}
	}

	// Extract results in descending order
	results := make([]SimilarityResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SimilarityResult)
	}
	return results
}

// --- Min-Heap for Top-K ---

type minHeap []SimilarityResult

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool   { return h[i].Score < h[j].Score }
func (h minHeap) Swap(i, j int)        { h[i], h[j] = h[j], h[i] }

func (h *minHeap) Push(x interface{}) {
	*h = append(*h, x.(SimilarityResult))
}

func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
