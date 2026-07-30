# Nearby as an Agent Sidecar

> Deploy Nearby as a sub-millisecond, ephemeral vector memory grid for multi-agent AI pipelines.

## Overview

In multi-agent architectures (Researcher → Coder → Reviewer swarms), agents need to pass intermediate embeddings to each other constantly. Nearby serves as a **sidecar** — a colocated, in-memory vector store that eliminates network round-trips and serialization overhead.

```
┌─────────────────────────────────────────────┐
│  Pod / Docker Compose Service               │
│                                             │
│  ┌─────────────┐    localhost    ┌────────┐  │
│  │   Agent      │◄────:6379────►│ Nearby │  │
│  │  (Researcher │               │Sidecar │  │
│  │  / Coder /   │               │        │  │
│  │  Reviewer)   │               │  <15MB │  │
│  └─────────────┘                └────────┘  │
└─────────────────────────────────────────────┘
```

## Quick Start

### Docker Compose

```bash
docker-compose -f Memory-Grid-Multi-Agent-Architectures/docker-compose.sidecar.yml up
```

### Kubernetes

```bash
kubectl apply -f Memory-Grid-Multi-Agent-Architectures/k8s-sidecar.yaml
```

### Local Development

```bash
# Start Nearby
go run cmd/server/main.go -port 6379

# In your agent code, connect to localhost:6379
```

---

## TTL Configuration

### Per-Vector TTL

Set TTL on individual vectors using the `EX` flag:

```
VSET swarm:run-42 chunk:7 1536 <vector...> META stage researcher EX 300
```

The vector expires 300 seconds after being written.

### Per-Namespace TTL

Set a default TTL for an entire namespace — all existing and future vectors inherit it:

```
VEXPIRE swarm:run-42 300
```

**Priority rules:**
1. Explicit per-vector `EX` overrides namespace default
2. If no `EX` is specified, the namespace default applies
3. Use `VSetWithTTL(..., -1)` in Go to bypass namespace default for persistent vectors

---

## Namespace-Per-Run Convention

Use a consistent naming convention to scope vector memory to each pipeline run:

```
<scope>:<run-id>
```

**Examples:**
- `swarm:run-42` — agent swarm run #42
- `rerank:req-9182` — reranker request #9182
- `pipeline:batch-2024-01-15` — daily batch pipeline

### Lifecycle

```
# 1. Researcher agent writes embeddings
VSET swarm:run-42 chunk:1 1536 <vec> META stage researcher EX 300
VSET swarm:run-42 chunk:2 1536 <vec> META stage researcher EX 300

# 2. Coder agent reads them
VSIMILARITY swarm:run-42 1536 <query-vec> TOP 5

# 3. Orchestrator tears down when run completes
VNS DROP swarm:run-42
```

### Batch Ingestion

For bulk-loading (e.g., reranker staging with hundreds of candidates):

```
VMSET rerank:req-9182 1536 300 chunk:1 <f1..f1536> chunk:2 <f1..f1536> ...
```

---

## When to Promote to HNSW

Nearby defaults to **brute-force** cosine similarity, which is optimal for the typical agent scratchpad (hundreds to low thousands of vectors). 

**Promote to HNSW when:**
- A namespace grows past **~5,000 vectors** and query latency matters
- You're running a long-lived swarm where the scratchpad accumulates over time

```
# Create HNSW index for a namespace
VINDEX CREATE swarm:long-run M 16 EF_CONSTRUCTION 200

# All subsequent VSIMILARITY queries use HNSW automatically
VSIMILARITY swarm:long-run 1536 <query-vec> TOP 10
```

**Don't promote when:**
- Vectors live for seconds (handoff pattern) — index build overhead isn't worth it
- Namespace has < 1,000 vectors — brute-force is already sub-millisecond

---

## Observability

### INFO Command

```
INFO
```

Returns per-namespace stats:

```
# Server
version:1.1.0
# Keyspace
kv_keys:0
# Vectors
vector_namespaces:2
total_vectors:150
# Vector Namespaces
ns:swarm:run-42:vectors:100
ns:swarm:run-42:approx_memory_bytes:614400
ns:swarm:run-42:ttl:285
ns:rerank:req-9182:vectors:50
ns:rerank:req-9182:approx_memory_bytes:307200
ns:rerank:req-9182:ttl:-1
```

### VNS LIST

```
VNS LIST
```

Returns an array of `[name, vectorCount, approxMemory, ttlRemaining]` for each active namespace.

---

## Memory Sizing Guidelines

| Workload | Vectors | Dimensions | Approx RAM | Recommended Limit |
|---|---|---|---|---|
| Single agent handoff | 100–500 | 768 | ~1.5 MB | 128 MB |
| Reranker staging | 300–1,000 | 1536 | ~6 MB | 256 MB |
| Multi-agent swarm (5 concurrent runs) | 5,000 | 1536 | ~30 MB | 512 MB |
| Long-running pipeline | 10,000+ | 1536 | ~60 MB+ | 1 GB |

**Formula:** `approx_bytes = vector_count × dimensions × 4 + (metadata_overhead × vector_count)`

---

## Connection Pattern

Agents connect to Nearby on `localhost:6379` using any Redis-compatible TCP client. The wire protocol is text-based:

```python
# Python example
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 6379))

# Write embedding
sock.send(b'VSET swarm:run-42 chunk:1 3 0.1 0.2 0.3 META stage researcher EX 300\r\n')
response = sock.recv(1024)  # +OK\r\n

# Query
sock.send(b'VSIMILARITY swarm:run-42 3 0.1 0.2 0.3 TOP 5\r\n')
results = sock.recv(4096)
```

Or use the Go client at `pkg/client`.

---

## Command Reference (New in v1.1.0)

| Command | Description |
|---|---|
| `VSET <ns> <id> <dim> <floats...> [META k v ...] [EX <s>]` | Store vector with optional TTL |
| `VMSET <ns> <dim> <count> <id1> <floats...> <id2> <floats...> ...` | Batch store vectors |
| `VEXPIRE <ns> <seconds>` | Set default TTL for entire namespace |
| `VNS DROP <ns>` | Tear down namespace and all its vectors |
| `VNS LIST` | List active namespaces with stats |
| `INFO` | Extended server info with per-namespace stats |
