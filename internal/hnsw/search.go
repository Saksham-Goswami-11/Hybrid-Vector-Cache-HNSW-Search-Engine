package hnsw

import (
	"container/heap"
	"fmt"
)

// searchLayer performs a greedy best-first search on a single graph layer.
// It is the core subroutine used by both Insert and Search (Paper Algorithm 5).
//
// Caller must NOT hold idx.mu — this function acquires read locks as needed.
func (idx *Index) searchLayer(query []float32, entryIDs []uint32, ef int, layer int) *MaxHeap {
	visited := make(map[uint32]bool, ef*2)

	candidates := &MinHeap{} // explore closest first
	found := &MaxHeap{}      // track ef best, evict furthest
	heap.Init(candidates)
	heap.Init(found)

	for _, epID := range entryIDs {
		if visited[epID] {
			continue
		}
		visited[epID] = true

		idx.mu.RLock()
		if int(epID) >= len(idx.nodes) {
			idx.mu.RUnlock()
			continue
		}
		node := idx.nodes[epID]
		idx.mu.RUnlock()

		dist := CosineDistance(query, node.Vector)
		item := Item{ID: epID, Dist: dist}
		heap.Push(candidates, item)
		heap.Push(found, item)
	}

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(Item)
		f := found.Peek()

		if c.Dist > f.Dist {
			break
		}

		// Get neighbors of c at this layer
		idx.mu.RLock()
		if int(c.ID) >= len(idx.nodes) {
			idx.mu.RUnlock()
			continue
		}
		cNode := idx.nodes[c.ID]
		idx.mu.RUnlock()

		neighbors := cNode.GetLinks(layer)

		for _, nID := range neighbors {
			if visited[nID] {
				continue
			}
			visited[nID] = true

			idx.mu.RLock()
			if int(nID) >= len(idx.nodes) {
				idx.mu.RUnlock()
				continue
			}
			nNode := idx.nodes[nID]
			idx.mu.RUnlock()

			dist := CosineDistance(query, nNode.Vector)
			f = found.Peek()

			if found.Len() < ef || dist < f.Dist {
				item := Item{ID: nID, Dist: dist}
				heap.Push(candidates, item)
				heap.Push(found, item)

				if found.Len() > ef {
					heap.Pop(found)
				}
			}
		}
	}

	return found
}

// Insert adds a vector to the HNSW graph. Implements Paper Algorithm 1.
// Thread-safe — uses idx.mu for graph mutation and per-node locks for link updates.
func (idx *Index) Insert(key string, vector []float32, metadata map[string]string) error {
	idx.mu.Lock()

	// Check for duplicate key — replace semantics
	if existingID, exists := idx.idMap[key]; exists {
		node := idx.nodes[existingID]
		idx.mu.Unlock()
		node.mu.Lock()
		node.Vector = vector
		node.Metadata = metadata
		node.mu.Unlock()
		return nil
	}

	// Assign internal ID
	internalID := uint32(len(idx.nodes))

	// Sample layer assignment
	l := idx.randomLevel()

	// Create node
	node := NewNode(internalID, vector, metadata, l)

	// Register in maps — all under the write lock
	idx.nodes = append(idx.nodes, node)
	idx.idMap[key] = internalID
	idx.reverseMap = append(idx.reverseMap, key)

	// First node — set as entry point
	if len(idx.nodes) == 1 {
		idx.entryPoint = internalID
		idx.currentMaxLayer = l
		idx.mu.Unlock()
		idx.insertCount.Add(1)
		return nil
	}

	// Capture current graph state
	ep := idx.entryPoint
	L := idx.currentMaxLayer
	idx.mu.Unlock()

	// Phase 1: Greedy descent from top layer down to l+1
	currentEP := []uint32{ep}
	for lc := L; lc > l; lc-- {
		found := idx.searchLayer(vector, currentEP, 1, lc)
		if found.Len() > 0 {
			nearest := heap.Pop(found).(Item)
			currentEP = []uint32{nearest.ID}
		}
	}

	// Phase 2: At each layer from min(L, l) down to 0
	topInsertLayer := l
	if L < topInsertLayer {
		topInsertLayer = L
	}

	for lc := topInsertLayer; lc >= 0; lc-- {
		found := idx.searchLayer(vector, currentEP, idx.EfConstruction, lc)

		foundItems := make([]Item, 0, found.Len())
		for found.Len() > 0 {
			foundItems = append(foundItems, heap.Pop(found).(Item))
		}

		maxConn := idx.M
		if lc == 0 {
			maxConn = idx.Mmax0
		}

		neighbors := idx.selectNeighborsHeuristic(vector, foundItems, maxConn, lc, false, true)

		// Add bidirectional links
		for _, neighborID := range neighbors {
			if neighborID == internalID {
				continue // Prevent double-lock deadlock and self-linking
			}

			idx.mu.RLock()
			if int(neighborID) >= len(idx.nodes) {
				idx.mu.RUnlock()
				continue
			}
			nNode := idx.nodes[neighborID]
			idx.mu.RUnlock()

			// Use per-node locks — lock in consistent order (lower ID first)
			a, b := node, nNode
			if internalID > neighborID {
				a, b = nNode, node
			}

			a.mu.Lock()
			b.mu.Lock()

			node.Links[lc] = append(node.Links[lc], neighborID)
			nNode.Links[lc] = append(nNode.Links[lc], internalID)

			needsPrune := len(nNode.Links[lc]) > maxConn
			var nLinks []uint32
			if needsPrune {
				nLinks = make([]uint32, len(nNode.Links[lc]))
				copy(nLinks, nNode.Links[lc])
			}

			b.mu.Unlock()
			a.mu.Unlock()

			// Prune outside locks
			if needsPrune {
				idx.pruneNeighbor(nNode, nLinks, maxConn, lc)
			}
		}

		// Use found nodes as entry points for next layer
		currentEP = make([]uint32, 0, len(foundItems))
		for _, item := range foundItems {
			currentEP = append(currentEP, item.ID)
		}
	}

	// Update entry point if the new node reaches a higher layer
	if l > L {
		idx.mu.Lock()
		if l > idx.currentMaxLayer {
			idx.entryPoint = internalID
			idx.currentMaxLayer = l
		}
		idx.mu.Unlock()
	}

	idx.insertCount.Add(1)
	return nil
}

// pruneNeighbor re-selects neighbors for a node that exceeds max connections.
func (idx *Index) pruneNeighbor(node *Node, currentLinks []uint32, maxConn int, layer int) {
	nItems := make([]Item, 0, len(currentLinks))
	for _, linkID := range currentLinks {
		idx.mu.RLock()
		if int(linkID) >= len(idx.nodes) {
			idx.mu.RUnlock()
			continue
		}
		lNode := idx.nodes[linkID]
		idx.mu.RUnlock()

		dist := CosineDistance(node.Vector, lNode.Vector)
		nItems = append(nItems, Item{ID: linkID, Dist: dist})
	}
	pruned := idx.selectNeighborsHeuristic(node.Vector, nItems, maxConn, layer, false, true)
	node.SetLinks(layer, pruned)
}

// Search performs a K-nearest-neighbor search on the HNSW graph.
// Implements Paper Algorithm 2 (KNN_SEARCH). Thread-safe.
func (idx *Index) Search(query []float32, K int, ef int) ([]SearchResult, error) {
	idx.mu.RLock()
	if len(idx.nodes) == 0 {
		idx.mu.RUnlock()
		return nil, fmt.Errorf("index is empty")
	}

	ep := idx.entryPoint
	L := idx.currentMaxLayer

	if ef <= 0 {
		ef = idx.Ef
	}
	if ef < K {
		ef = K
	}

	idx.mu.RUnlock()

	// Phase 1: Greedy descent from top layer to layer 1
	currentEP := []uint32{ep}
	for lc := L; lc >= 1; lc-- {
		found := idx.searchLayer(query, currentEP, 1, lc)
		if found.Len() > 0 {
			nearest := heap.Pop(found).(Item)
			currentEP = []uint32{nearest.ID}
		}
	}

	// Phase 2: Full search at ground layer
	found := idx.searchLayer(query, currentEP, ef, 0)

	// Extract K nearest from found set (max-heap → reverse for ascending)
	all := make([]Item, 0, found.Len())
	for found.Len() > 0 {
		all = append(all, heap.Pop(found).(Item))
	}

	results := make([]SearchResult, 0, K)

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	for i := len(all) - 1; i >= 0 && len(results) < K; i-- {
		item := all[i]
		if int(item.ID) >= len(idx.nodes) {
			continue
		}
		node := idx.nodes[item.ID]
		if node.IsDeleted() {
			continue
		}
		results = append(results, SearchResult{
			ID:       idx.reverseMap[item.ID],
			Score:    1.0 - item.Dist,
			Metadata: node.Metadata,
		})
	}

	idx.searchCount.Add(1)
	return results, nil
}

// SearchResult is a single result from an HNSW search.
type SearchResult struct {
	ID       string            // the string key
	Score    float32           // cosine similarity [−1, 1]
	Metadata map[string]string // attached metadata
}


