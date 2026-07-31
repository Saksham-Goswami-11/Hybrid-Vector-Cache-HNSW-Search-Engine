package hnsw

import "sync"

// Node represents a single vector in the HNSW graph.
// Each node holds its embedding, optional metadata, and per-layer neighbor links.
type Node struct {
	ID       uint32            // internal integer ID (maps to string key via Index.reverseMap)
	Vector   []float32         // the embedding — owned by this node
	Metadata map[string]string // arbitrary key-value pairs from VSET META

	// Links[layer] = []neighborID
	// Links[0] has up to Mmax0 (2*M) neighbors (ground layer).
	// Links[1..L] have up to M neighbors each.
	Links [][]uint32

	Deleted bool         // tombstone flag for soft deletion
	mu      sync.RWMutex // per-node lock for concurrent link mutation
}

// IsDeleted reports whether this node is tombstoned.
func (n *Node) IsDeleted() bool {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.Deleted
}

// SetDeleted marks or unmarks this node as tombstoned.
func (n *Node) SetDeleted(deleted bool) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.Deleted = deleted
}

// NewNode creates a node assigned to the given maximum layer.
// It allocates empty link slices for layers 0..maxLayer.
func NewNode(id uint32, vec []float32, meta map[string]string, maxLayer int) *Node {
	links := make([][]uint32, maxLayer+1)
	for i := range links {
		links[i] = make([]uint32, 0)
	}
	return &Node{
		ID:       id,
		Vector:   vec,
		Metadata: meta,
		Links:    links,
	}
}

// MaxLayer returns the highest layer this node exists in.
func (n *Node) MaxLayer() int {
	return len(n.Links) - 1
}

// GetLinks returns a copy of the neighbor list at the given layer.
// Safe for concurrent use — takes a read lock.
func (n *Node) GetLinks(layer int) []uint32 {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if layer >= len(n.Links) {
		return nil
	}

	// Return a copy so callers can iterate without holding the lock
	cp := make([]uint32, len(n.Links[layer]))
	copy(cp, n.Links[layer])
	return cp
}

// SetLinks replaces the neighbor list at the given layer.
// Caller must ensure proper locking (typically idx-level lock ordering).
func (n *Node) SetLinks(layer int, links []uint32) {
	n.mu.Lock()
	defer n.mu.Unlock()

	if layer >= len(n.Links) {
		return
	}
	n.Links[layer] = links
}

// AddLink appends a neighbor to the given layer's link list.
// Caller must ensure proper locking (typically idx-level lock ordering).
func (n *Node) AddLink(layer int, neighborID uint32) {
	n.mu.Lock()
	defer n.mu.Unlock()

	if layer >= len(n.Links) {
		return
	}
	n.Links[layer] = append(n.Links[layer], neighborID)
}

// LinkCount returns the total number of links across all layers.
func (n *Node) LinkCount() int {
	n.mu.RLock()
	defer n.mu.RUnlock()

	total := 0
	for _, layerLinks := range n.Links {
		total += len(layerLinks)
	}
	return total
}
