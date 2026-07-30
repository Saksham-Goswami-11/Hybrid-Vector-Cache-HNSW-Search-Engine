# SYNAPSE CACHE

## An In-Memory Vector Database & Similarity Engine

### Written in Go

|**Document Type** | Product Requirements Document (PRD)|
|--- | ---|
|**Version** | 1.0 Initial Release|
|**Status** | Ready for Development|
|**Author** | Engineering Lead|
|**Last Updated** | June 2025|
|**Classification** | Open Source Portfolio Project|

[github.com/your-handle/nearby](https://github.com/your-handle/nearby)

# 1. Executive Summary

Synapse Cache is a purpose-built, in-memory database written in Go that solves a real and painful gap in the AI engineering ecosystem: the absence of a lightweight, zero-dependency vector cache that developers can drop into any RAG or agentic pipeline without standing up a full-blown vector database.

Today's developers who build Retrieval-Augmented Generation (RAG) systems are forced into a bad tradeoff: use Redis and serialize embeddings as JSON strings paying a heavy deserialization cost on every query or deploy Chroma, Pinecone, or Weaviate, which are full database systems with operational overhead that dwarfs the use case. Synapse Cache sits between these two options.

> [!NOTE]
> **The Core Insight**
> Standard Redis stores embeddings as base64 or JSON strings and deserializes them on every similarity query. Synapse Cache stores raw []float32 slices natively in memory and computes cosine similarity in-place eliminating an entire serialization round-trip on the hot path. This is the architectural choice that makes benchmarks compelling and the README story clear.

From a portfolio standpoint, Synapse Cache demonstrates command of three distinct engineering disciplines simultaneously: high-concurrency systems design (goroutines, RWMutex, worker pools), applied linear algebra (cosine similarity, dot products, vector normalization), and protocol design (a custom wire protocol with pipelining support). Very few portfolio projects touch all three.

# 2. Problem Statement

## 2.1 The Gap in the Ecosystem

The rise of LLM-powered applications has created enormous demand for fast embedding lookup. Every RAG pipeline, semantic search feature, and memory-augmented AI agent needs to answer the same core question at query time: "Given this query vector, what are the K most similar stored vectors?"

The current options force developers to choose between operational complexity and engineering compromise:

|**Option** | **The appeal** | **The problem**|
|--- | --- | ---|
|**Redis + JSON** | Zero new infra | Embeds serialize/deserialize on every query; no native similarity ops; O(n) scan in Lua|
|**Chroma / Weaviate** | Full vector DB capabilities | Heavy operational overhead; Docker required; overkill for sub-100K embedding caches|
|**Pinecone / Qdrant** | Managed, scalable | Network latency; cost; external dependency; no local dev story|
|**FAISS (Python)** | Fast ANN search | Python-only; no server mode; not embeddable in Go services; no TTL or eviction|

## 2.2 What Developers Actually Need

- A server they can start with one binary and zero configuration
- A protocol they can speak with netcat or a five-line Go client
- Native float32 vector storage no JSON, no base64, no codec overhead
- Cosine similarity search that returns top-K results in a single command
- TTL-based eviction so embeddings for expired sessions are cleaned up automatically
- A persistence story for warm restarts, even if it's just a snapshot file

> [!NOTE]
> **Why Go?**
> Go gives us goroutine-per-connection concurrency with predictable GC latency, first-class support for []float32 slice operations, easy static binary compilation (docker scratch containers), and a strong standard library for TCP server construction all without the JVM or Python GIL standing in the way.

# 3. Goals & Non-Goals

## 3.1 Goals

- **G1 Correct** Cosine similarity results must be mathematically exact (brute-force, not approximate) for datasets up to 500K vectors.
- **G2 Fast** VSIMILARITY on a 10K-vector namespace must complete in under 5ms on a modern laptop CPU (single-threaded baseline).
- **G3 Concurrent** The server must handle 1,000 simultaneous client connections without data races. Verified via Go's race detector.
- **G4 Operable** A developer must be able to run the server, load 1,000 embeddings, and issue a similarity query in under 10 minutes from a fresh clone.
- **G5 Demonstrable** The repository must include a working RAG demo: embed 50 text chunks with OpenAI, store them in Synapse Cache, and answer a natural-language query via nearest-neighbor lookup.

## 3.2 Non-Goals (v1.0)

- **Approximate Nearest Neighbor (ANN)** No HNSW, IVF, or PQ. Brute-force is the right choice for the target scale and makes the code auditable.
- **Replication & clustering** Single-node only. Distributed systems concerns are a v2 problem.
- **Authentication** No auth in v1. Designed for localhost and trusted network use. A token-auth flag is a one-day v1.1 addition.
- **Persistent indexing** AOF log for warm restart only not a transactional store. If the process crashes mid-write, the snapshot is the source of truth.
- **Multi-tenancy** Namespaces provide logical isolation but no resource quotas or per-namespace limits.

# 4. System Architecture

## 4.1 High-Level Design

Synapse Cache is a single-process TCP server. Each client connection spawns a goroutine that reads from the socket, hands the raw command to the parser, and writes the response back. The in-memory store is protected by a sync.RWMutex reads (GET, VSIMILARITY) take a read lock and run concurrently; writes (SET, VSET, DEL) take a write lock.

Similarity queries are dispatched to a worker pool (default: GOMAXPROCS workers) so a large VSIMILARITY scan does not block incoming client connections. Results are assembled and written back on the originating goroutine.

## 4.2 Component Breakdown

|**Component** | **Responsibility**|
|--- | ---|
|**cmd/server/main.go** | Entry point. Parses flags, initialises store, starts TCP listener, wires up graceful shutdown.|
|**internal/server/server.go** | Accept loop. Spawns one goroutine per connection. Owns the listener lifecycle.|
|**internal/protocol/parser.go** | Tokenizes raw bytes into Command structs. Handles pipelining (multiple commands in one read).|
|**internal/store/store.go** | Thread-safe in-memory store. Manages KV namespace (string values + TTL) and vector namespace ([]float32 + metadata map).|
|**internal/similarity/cosine.go** | Pure math. Computes cosine similarity between two []float32 slices. No dependencies. Benchmarked independently.|
|**internal/similarity/engine.go** | Orchestrates top-K search across all vectors in a namespace. Dispatches to worker pool. Returns ranked results.|
|**internal/persist/aof.go** | Append-only log writer/reader. Streams commands to disk. Replays on startup for warm restart.|
|**pkg/client/client.go** | Official Go client library. Implements the protocol. Exported for use in the RAG demo and external projects.|

## 4.3 Concurrency Model

Three layers of concurrency are used, each with a clear scope:

- **Connection goroutines** One goroutine per TCP connection. Each goroutine owns a bufio.Reader/Writer pair. No shared state except the store.
- **Store lock (sync.RWMutex)** Read operations (GET, VSIMILARITY scan setup) take RLock(). Write operations (SET, VSET, DEL, EXPIRE) take Lock(). The similarity compute itself happens outside the lock vectors are copied into a snapshot before the lock is released.
- **Similarity worker pool** A semaphore-bounded goroutine pool (default: runtime.GOMAXPROCS(0) workers) processes VSIMILARITY chunks in parallel using Go's sync.WaitGroup pattern.

## 4.4 Directory Structure

```
nearby/
├── cmd/
│ └── server/
│ └── main.go # Entry point
├── internal/
│ ├── server/
│ │ └── server.go # TCP accept loop
│ ├── protocol/
│ │ ├── parser.go # Command tokenizer
│ │ └── types.go # Command / Response types
│ ├── store/
│ │ └── store.go # In-memory store (KV + Vector)
│ ├── similarity/
│ │ ├── cosine.go # Core math
│ │ └── engine.go # Top-K search, worker pool
│ └── persist/
│ └── aof.go # Append-only log
├── pkg/
│ └── client/
│ └── client.go # Public Go client library
├── examples/
│ └── rag_demo/
│ └── main.go # OpenAI + Synapse RAG demo
├── bench/
│ └── bench_test.go # go test -bench benchmarks
├── Makefile
├── Dockerfile
└── README.md
```

# 5. Wire Protocol Specification

## 5.1 Design Principles

- Human-readable text protocol (like Redis RESP1, not binary)
- Each command is a single line terminated by \r\n
- Responses are prefixed by a single-character type byte: + (simple string), - (error), : (integer), $ (bulk string), * (array)
- Pipelining is supported: multiple commands may be sent before reading responses

## 5.2 Key-Value Commands

### SET

```redis
SET <key> <value>
SET <key> <value> EX <seconds>
 
Response: +OK
 
Examples:
 SET session:abc123 user_id_99
 SET cache:model_version gpt-4o EX 3600
```

### GET

```redis
GET <key>
 
Response: $<len>\r\n<value>\r\n (bulk string)
 $-1 (nil, key not found)
```

### DEL

```redis
DEL <key> [<key> ...]
 
Response: :<count> (number of keys deleted)
```

### EXPIRE

```text
EXPIRE <key> <seconds>
 
Response: :1 (key existed and TTL set)
 :0 (key not found)
```

### TTL

```text
TTL <key>
 
Response: :<seconds_remaining>
 :-1 (key exists but has no TTL)
 :-2 (key does not exist)
```

## 5.3 Vector Commands

### VSET Store a vector

```redis
VSET <namespace> <id> <dim> <f1> <f2> ... <fN> [META <k> <v> ...]
 
Response: +OK
 
Example store a 4-dim embedding with metadata:
 VSET docs chunk:42 4 0.1823 -0.4412 0.9901 0.0034 META source paper.pdf page 7
```

Notes: <dim> must match the number of floats provided. Synapse validates dimension on every VSET and rejects mismatches. Vectors are stored as raw []float32 no JSON, no base64.

### VGET Retrieve a stored vector

```redis
VGET <namespace> <id>
 
Response: *<N> (array of N floats, one per line)
 $-1 (not found)
```

### VSIMILARITY Top-K cosine similarity search

```redis
VSIMILARITY <namespace> <dim> <f1> ... <fN> TOP <k>
 
Response: *<k>
 *3
 $<len>\r\n<id>\r\n (vector ID)
 +<score> (cosine similarity, 4 decimal places)
 *<meta_count> (metadata pairs)
 ...
 
Example:
 VSIMILARITY docs 4 0.18 -0.44 0.99 0.00 TOP 3
```

### VDEL Delete a stored vector

```redis
VDEL <namespace> <id>
 
Response: :1 (deleted) or :0 (not found)
```

### VCOUNT Count vectors in a namespace

```text
VCOUNT <namespace>
 
Response: :<count>
```

## 5.4 Server Commands

### PING

```text
PING
PING <message>
 
Response: +PONG
 +<message>
```

### INFO

```text
INFO
 
Response: bulk string containing:
 # Server
 version:1.0.0
 uptime_seconds:3600
 # Memory
 used_memory_bytes:104857600
 # Keyspace
 kv_keys:1204
 # Vectors
 vector_namespaces:3
 total_vectors:48291
```

### BGSAVE

```text
BGSAVE
 
Response: +Background saving started
 
Triggers an async snapshot to synapse.aof on disk.
```

# 6. Functional Requirements

## 6.1 Core Store

|**ID** | **Requirement** | **Priority** | **Phase**|
|--- | --- | --- | ---|
|**FR-01** | Server MUST start on a configurable TCP port (default: 6379) with a --port flag. | P0 | v1.0|
|**FR-02** | Server MUST handle SET, GET, DEL, EXPIRE, TTL commands with correct semantics. | P0 | v1.0|
|**FR-03** | TTL expiry MUST be lazy (checked on GET) and active (background sweep every 100ms). | P0 | v1.0|
|**FR-04** | Server MUST handle VSET, VGET, VDEL, VCOUNT, VSIMILARITY commands. | P0 | v1.0|
|**FR-05** | VSET MUST reject vectors where provided float count differs from declared <dim>. | P0 | v1.0|
|**FR-06** | VSIMILARITY MUST return results sorted by descending cosine similarity. | P0 | v1.0|
|**FR-07** | VSIMILARITY MUST support TOP K where K is 1..1000. | P0 | v1.0|
|**FR-08** | Server MUST support vector metadata (arbitrary key-value pairs) stored alongside each vector. | P1 | v1.0|
|**FR-09** | Metadata MUST be returned alongside similarity results in VSIMILARITY responses. | P1 | v1.0|
|**FR-10** | Server MUST support multiple namespaces. Each namespace is an independent vector index. | P0 | v1.0|
|**FR-11** | INFO command MUST return server uptime, memory usage, KV key count, and per-namespace vector counts. | P1 | v1.0|
|**FR-12** | BGSAVE MUST write an append-only log to disk asynchronously. | P1 | v1.0|
|**FR-13** | Server MUST replay the AOF log on startup if a snapshot file exists (--aof flag). | P1 | v1.0|
|**FR-14** | Server MUST respond to PING with PONG within 1ms under no load. | P0 | v1.0|

## 6.2 Client Library

|**ID** | **Requirement** | **Priority** | **Phase**|
|--- | --- | --- | ---|
|**FR-15** | pkg/client must expose Set(), Get(), Del(), VSet(), VSimilarity() as typed Go methods. | P0 | v1.0|
|**FR-16** | Client MUST manage connection pooling transparently (min: 1, max: configurable). | P1 | v1.0|
|**FR-17** | VSimilarity() MUST return []SimilarityResult{ID, Score, Metadata} sorted by descending score. | P0 | v1.0|
|**FR-18** | Client MUST support context.Context cancellation on all blocking calls. | P1 | v1.0|

# 7. Non-Functional Requirements

## 7.1 Performance Targets

> [!NOTE]
> **How these numbers were derived**
> OpenAI text-embedding-3-small produces 1536-dim vectors. A developer caching a 10K-chunk corpus (a ~200-page book) should get sub-5ms similarity results. At 1536 floats × 4 bytes × 10K vectors = ~60MB in memory comfortably within a single server's RAM. The benchmark suite in bench/ will validate all targets below.

|**Metric** | **Target** | **Test condition**|
|--- | --- | ---|
|**VSIMILARITY latency (p50)** | < 5ms | 10K vectors, dim=1536, TOP 10, cold cache|
|**VSIMILARITY latency (p50)** | < 50ms | 100K vectors, dim=1536, TOP 10|
|**GET / SET throughput** | > 100K ops/sec | 100-byte values, pipelined, localhost|
|**Concurrent connections** | 1,000 simultaneous | No errors, no data races (go -race)|
|**VSET throughput** | > 10K ops/sec | dim=1536 vectors|
|**Server startup time** | < 200ms | Including AOF replay of 10K vectors|
|**Memory overhead per vector** | < 10KB | dim=1536 + 5 metadata pairs|

## 7.2 Reliability

- **No data races** All tests must pass under go test -race. This is a hard gate any race condition is a P0 bug.
- **Graceful shutdown** SIGTERM must trigger a final BGSAVE before closing connections. Clients must receive a connection-closing error, not a hard reset.
- **Panic recovery** A panic in a connection goroutine must be recovered and logged it must not crash the server process.
- **AOF integrity** If the AOF file is truncated or corrupt, the server must log a warning and start with whatever state it can reconstruct, not refuse to start.

## 7.3 Portability

- **Static binary** go build must produce a statically linked binary with no C dependencies.
- **Docker image** A Dockerfile using scratch base must produce an image under 15MB.
- **Platform support** linux/amd64 (primary), darwin/arm64 (local dev), linux/arm64 (Raspberry Pi / cheap cloud).

# 8. Implementation Plan

## 8.1 Phase Overview

|**Phase** | **Duration** | **Deliverables**|
|--- | --- | ---|
|**Phase 1** | Week 12 | TCP server skeleton, connection handler, PING/PONG, SET/GET/DEL. Passes basic integration tests.|
|**Phase 2** | Week 34 | TTL machinery, EXPIRE/TTL commands, background expiry sweep. Passes TTL correctness tests.|
|**Phase 3** | Week 56 | Vector namespace, VSET/VGET/VDEL/VCOUNT. Float parsing, dimension validation.|
|**Phase 4** | Week 78 | Similarity engine cosine.go, worker pool, VSIMILARITY. Correctness + benchmark tests passing.|
|**Phase 5** | Week 9 | Persistence layer AOF writer, AOF replay on startup, BGSAVE command.|
|**Phase 6** | Week 10 | Go client library in pkg/client. Full API coverage with context support.|
|**Phase 7** | Week 11 | RAG demo in examples/rag_demo. OpenAI embeddings, 50-chunk corpus, Q&A loop.|
|**Phase 8** | Week 12 | Polish: README, benchmarks, Docker image, GitHub Actions CI, final documentation.|

## 8.2 Phase 1 TCP Server Skeleton (Week 12)

1. Scaffold Go module: go mod init github.com/your-handle/nearby
2. Implement cmd/server/main.go with --port and --aof flags
3. Implement internal/server/server.go: net.Listen, Accept loop, per-connection goroutine
4. Implement internal/protocol/parser.go: tokenize by space, handle \r\n, return Command{Name, Args}
5. Implement internal/store/store.go: sync.RWMutex-protected map[string]string
6. Wire SET, GET, DEL, PING handlers
7. Write integration test: start server, connect, SET key, GET key, assert response

## 8.3 Phase 3 Vector Namespace (Week 56)

The vector namespace is a separate data structure from the KV store. Use the following schema:

```go
type VectorEntry struct {
 ID string
 Vector []float32
 Metadata map[string]string
}
 
type VectorNamespace struct {
 entries map[string]*VectorEntry // keyed by ID
 mu sync.RWMutex
}
 
type Store struct {
 kv map[string]kvEntry
 vectors map[string]*VectorNamespace // keyed by namespace
 mu sync.RWMutex
}
```

## 8.4 Phase 4 Similarity Engine (Week 78)

The cosine similarity between two unit-norm vectors A and B is their dot product. For non-normalized vectors:

```go
// cosine.go
func CosineSimilarity(a, b []float32) (float32, error) {
 if len(a) != len(b) {
 return 0, ErrDimensionMismatch
 }
 var dot, normA, normB float32
 for i := range a {
 dot += a[i] * b[i]
 normA += a[i] * a[i]
 normB += b[i] * b[i]
 }
 if normA == 0 || normB == 0 {
 return 0, ErrZeroVector
 }
 return dot / (float32(math.Sqrt(float64(normA))) *
 float32(math.Sqrt(float64(normB)))), nil
}
```

The engine.go Top-K search takes a snapshot of all vectors under RLock (copying the slice headers, not the float data), releases the lock, then fans out the similarity computation across a worker pool using channels. A min-heap of size K collects results.

## 8.5 Phase 7 RAG Demo

The RAG demo is the project's centrepiece showcase. It must be a complete, runnable program in examples/rag_demo/main.go that demonstrates end-to-end value:

```redis
// Workflow (pseudocode)
1. Load 50 text chunks from a sample document (included in the repo)
2. Call OpenAI /v1/embeddings to get 1536-dim embeddings
3. VSET each embedding into Synapse Cache namespace 'docs'
4. Enter a REPL loop:
 > Type a natural language question
 > Embed the question via OpenAI
 > VSIMILARITY docs <query_vec> TOP 3
 > Print the top 3 matching chunks with scores
 > (Optional) Pass context + question to GPT-4o for an answer
```

The demo should ship with a pre-embedded JSON file (chunks + embeddings) so the demo runs without an OpenAI key for basic testing. The OPENAI_API_KEY path enables live embedding and the optional answer generation.

# 9. Testing Strategy

## 9.1 Test Layers

|**Layer** | **Location** | **What it covers**|
|--- | --- | ---|
|**Unit** | internal/*/**_test.go | CosineSimilarity correctness, parser edge cases, TTL expiry timing, store read/write isolation|
|**Integration** | internal/server/*_test.go | Start real server on random port, send raw TCP commands, assert response bytes|
|**Race detection** | CI via go test -race | Every test run in CI executes under the race detector|
|**Benchmarks** | bench/bench_test.go | go test -bench for VSIMILARITY at 1K / 10K / 100K vectors; SET/GET throughput|
|**Fuzz** | internal/protocol/fuzz_test.go | go test -fuzz on the parser ensures no panics on malformed input|

## 9.2 Critical Test Cases

- **Cosine correctness** CosineSimilarity([1,0,0], [1,0,0]) == 1.0, CosineSimilarity([1,0,0], [-1,0,0]) == -1.0, CosineSimilarity([1,0,0], [0,1,0]) == 0.0
- **Dimension mismatch** VSET with 3 declared dims but 4 floats returns -ERR dimension mismatch
- **Race detector clean** go test -race ./... with 100 concurrent goroutines all hammering GET/SET/VSIMILARITY zero races
- **TTL expiry** SET key val EX 1, sleep 1.1s, GET key returns $-1
- **AOF round-trip** Load 1K vectors, BGSAVE, restart server with --aof, VCOUNT returns 1K
- **Empty namespace** VSIMILARITY on a namespace with 0 vectors returns *0
- **Fuzz parser** No panic on any arbitrary byte sequence up to 10KB

# 10. Go Client Library API

## 10.1 Connection

```text
import "github.com/your-handle/nearby/pkg/client"
 
c, err := client.New(client.Options{
 Addr: "localhost:6379",
 MaxConns: 10,
 DialTimeout: 5 * time.Second,
})
if err != nil {
 log.Fatal(err)
}
defer c.Close()
```

## 10.2 Key-Value

```text
// Set with optional TTL
err = c.Set(ctx, "key", "value", 0) // no expiry
err = c.Set(ctx, "key", "value", 1*time.Hour) // 1 hour TTL
 
// Get
val, err := c.Get(ctx, "key") // returns "", ErrNil if not found
 
// Delete
count, err := c.Del(ctx, "key1", "key2")
```

## 10.3 Vectors

```text
// Store a vector
err = c.VSet(ctx, client.VSetArgs{
 Namespace: "docs",
 ID: "chunk:42",
 Vector: []float32{0.18, -0.44, 0.99, 0.00},
 Metadata: map[string]string{"source": "paper.pdf", "page": "7"},
})
 
// Top-K similarity search
results, err := c.VSimilarity(ctx, client.VSimilarityArgs{
 Namespace: "docs",
 Vector: queryVector, // []float32
 TopK: 5,
})
 
// results is []SimilarityResult
for _, r := range results {
 fmt.Printf("ID: %s Score: %.4f Meta: %v\n", r.ID, r.Score, r.Metadata)
}
```

# 11. Benchmark Targets & README Story

## 11.1 Benchmark Suite

The bench/ directory must contain a go test -bench suite that produces the following table when run on the target hardware (any modern laptop). These numbers will appear in the README and are the primary technical marketing of the project.

|**Benchmark** | **Corpus size** | **Target p50** | **Comparison**|
|--- | --- | --- | ---|
|**BenchmarkVSimilarity1K** | 1,000 × dim-1536 | < 1ms | Redis JSON + Lua: ~40ms|
|**BenchmarkVSimilarity10K** | 10,000 × dim-1536 | < 5ms | Redis JSON + Lua: ~400ms|
|**BenchmarkVSimilarity100K** | 100,000 × dim-1536 | < 50ms | Redis JSON + Lua: ~4,000ms|
|**BenchmarkSetGet** | 100-byte values | > 100K ops/s | Redis: ~150K ops/s|
|**BenchmarkVSet1536** | dim-1536 vectors | > 10K ops/s | Baseline|

The Redis comparison numbers are derived from the known cost of JSON.parse() on a 1536-float array (~3.5KB) plus a Lua-side scan. They are conservative estimates actual Redis performance on this workload is worse.

## 11.2 README Structure

The README.md is a core deliverable. It must tell a story, not just list features. Recommended structure:

- **Hook (2 paragraphs)** The problem with caching embeddings in Redis. The insight. The sentence: 'Synapse stores raw []float32 slices and computes similarity in-place.'
- **Quick Start (5 steps)** Clone, build, start server, run demo, see results. Must work in under 10 minutes.
- **Architecture diagram** Mermaid diagram of the component layout.
- **Benchmark table** The table from §11.1 with actual measured numbers filled in.
- **Protocol reference** The wire protocol commands with examples enough for someone to telnet in and query.
- **RAG demo walkthrough** Screenshot/asciicast of the RAG demo in action.
- **Design decisions** Why brute-force over ANN. Why Go. Why a custom protocol over HTTP. This section makes senior engineers stop and read.

# 12. Risks & Mitigations

|**Risk** | **Severity** | **Mitigation**|
|--- | --- | ---|
|**Benchmark numbers don't beat Redis on KV ops** | Medium | Redis is optimized over 15 years. Frame benchmarks as 'within 2× Redis on KV, 80× faster on similarity' the story is similarity, not KV.|
|**Scope creep into ANN / clustering** | High | Explicit non-goals in this PRD. Any ANN implementation moves to a v2 milestone.|
|**Data race found in similarity engine** | High | Copy vector slice headers (not data) before releasing RLock. Write race detector tests in CI before shipping Phase 4.|
|**Memory blowup on 100K × dim-1536** | Medium | 1536 × 4 × 100K = 614MB. Document memory requirements clearly. Add --max-memory flag as a safety valve.|
|**AOF replay corrupts state on partial write** | Low | CRC32 checksum each AOF entry. Skip corrupted entries with a warning. Test with truncated AOF files.|
|**Project perceived as 'toy Redis clone'** | Medium | Lead with the vector story, not the KV story. The README hook must establish the AI/RAG angle in the first paragraph.|

# 13. Success Metrics

## 13.1 Technical Success

- **All tests pass under go test -race ./...** Zero data races detected.
- **All benchmark targets met** Numbers in §11.1 achieved on the test machine and committed to README.
- **RAG demo runs end-to-end** Startup to first similarity result in under 10 minutes from a fresh clone.
- **Docker image under 15MB** Achieved via scratch base image.
- **Zero external runtime dependencies** go.sum contains only test and tooling dependencies no production deps.

## 13.2 Portfolio Success

- **GitHub stars** 50+ stars within 30 days of launch (organic no paid promotion).
- **README clarity** A senior Go engineer unfamiliar with vector databases understands the project in under 5 minutes.
- **Interview leverage** At least 3 technical interviews in which the candidate demonstrates the benchmark table and the RAG demo live.
- **Hacker News / Reddit reception** A Show HN post receives substantive technical engagement (not just 'cool project').

# 14. Future Roadmap (v2+)

The following features are explicitly out of scope for v1.0 but are natural extensions that keep the project alive and growing:

- **v1.1 Token authentication** A --auth flag that requires AUTH <token> before any command. One day of work.
- **v1.2 Dot product similarity** Add VSIMILARITY ... METHOD DOT for use with pre-normalized embeddings (marginally faster).
- **v1.3 Batch VSET** VMSET command for bulk-loading vectors in a single round-trip. Critical for large corpus ingestion.
- **v2.0 Approximate Nearest Neighbor** Implement NSW (Navigable Small World) graph for sub-linear ANN queries at 1M+ vectors.
- **v2.1 Persistence v2** Write-ahead log with CRC32 checksums per entry and compaction support.
- **v3.0 Replication** Leader-follower replication using a Raft-lite protocol. At this point Synapse is a serious piece of infrastructure.

# Appendix A Glossary

|**Term** | **Definition**|
|--- | ---|
|**Embedding** | A dense vector of floats (typically 7683072 dimensions) that encodes the semantic meaning of a text chunk, image, or other data, as produced by a neural encoder model.|
|**RAG** | Retrieval-Augmented Generation. A pattern where a language model's response is grounded by relevant documents retrieved via embedding similarity search at query time.|
|**Cosine similarity** | A measure of similarity between two vectors equal to the cosine of the angle between them. Ranges from -1 (opposite) to 1 (identical direction). The standard metric for comparing text embeddings.|
|**Top-K search** | Finding the K vectors in a corpus most similar to a query vector. The core operation of any vector database.|
|**AOF** | Append-Only File. A persistence strategy where every write command is appended to a log file. The log is replayed on restart to restore state.|
|**Namespace** | A named partition within the vector store. Each namespace holds an independent set of vectors with its own ID space.|
|**dim** | Dimension count of a vector. All vectors in a namespace should share the same dimension (enforced by convention; Synapse validates on VSIMILARITY).|
|**RESP** | Redis Serialization Protocol. Synapse uses a simplified compatible subset.|
|**Worker pool** | A fixed set of goroutines that process work items from a shared queue, bounding peak parallelism.|

# Appendix B Environment Variables

|**Variable** | **Default** | **Description**|
|--- | --- | ---|
|**SYNAPSE_PORT** | 6379 | TCP port to listen on. Overridden by --port flag.|
|**SYNAPSE_AOF_PATH** | synapse.aof | Path to the append-only log file.|
|**SYNAPSE_MAX_MEMORY** | 0 (unlimited) | Soft memory limit in bytes. 0 = no limit.|
|**SYNAPSE_WORKERS** | GOMAXPROCS | Size of the similarity engine worker pool.|
|**SYNAPSE_LOG_LEVEL** | info | Log verbosity: debug, info, warn, error.|
|**OPENAI_API_KEY** | (none) | Required by the RAG demo only. Not used by the server.|
