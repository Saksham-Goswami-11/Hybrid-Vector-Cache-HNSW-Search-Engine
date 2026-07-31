package similarity

import (
	"errors"
	"math"
)

// Sentinel errors for cosine similarity computation.
var (
	ErrDimensionMismatch = errors.New("dimension mismatch")
	ErrZeroVector        = errors.New("zero-magnitude vector")
)

// CosineSimilarity computes the cosine similarity between two float32 vectors.
// Uses 4-way loop unrolling to maximize CPU instruction-level parallelism (ILP).
// Returns a value in [-1, 1]. Returns an error if dimensions differ or either
// vector has zero magnitude.
func CosineSimilarity(a, b []float32) (float32, error) {
	n := len(a)
	if n != len(b) {
		return 0, ErrDimensionMismatch
	}

	var dot0, dot1, dot2, dot3 float32
	var normA0, normA1, normA2, normA3 float32
	var normB0, normB1, normB2, normB3 float32

	i := 0
	// 4-way unrolled loop for instruction-level parallelism
	for ; i <= n-4; i += 4 {
		a0, a1, a2, a3 := a[i], a[i+1], a[i+2], a[i+3]
		b0, b1, b2, b3 := b[i], b[i+1], b[i+2], b[i+3]

		dot0 += a0 * b0
		dot1 += a1 * b1
		dot2 += a2 * b2
		dot3 += a3 * b3

		normA0 += a0 * a0
		normA1 += a1 * a1
		normA2 += a2 * a2
		normA3 += a3 * a3

		normB0 += b0 * b0
		normB1 += b1 * b1
		normB2 += b2 * b2
		normB3 += b3 * b3
	}

	dot := dot0 + dot1 + dot2 + dot3
	normA := normA0 + normA1 + normA2 + normA3
	normB := normB0 + normB1 + normB2 + normB3

	// Tail loop for remaining dimensions
	for ; i < n; i++ {
		ai := a[i]
		bi := b[i]
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	if normA == 0 || normB == 0 {
		return 0, ErrZeroVector
	}

	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))), nil
}
