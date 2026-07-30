# PRD: Nearby as an Ephemeral Vector Memory Grid for Multi-Agent Architectures

**Status:** Draft
**Owner:** Saksham Goswami
**Component:** `nearby` (Hybrid-Vector-Cache-HNSW-Search-Engine)
**Related roadmap items:** `VMSET`, TTL on vector namespace, Prometheus metrics

---

## 1. Summary

Multi-agent AI systems (Researcher / Coder / Reviewer style swarms) need a shared scratchpad for intermediate embeddings that is faster and lighter than a networked vector database. This PRD defines the requirements to position and harden Nearby as that scratchpad: a sidecar-deployable, sub-millisecond, in-memory vector store for short-lived, agent-to-agent context sharing.

This is a positioning + capability PRD, not a rewrite. Nearby's existing wire protocol (`VSET` / `VSIMILARITY` / `VINDEX`) already covers the core mechanics. The gaps are in **ephemerality**, **multi-tenancy across agents**, **observability**, and **deployment ergonomics** for the sidecar pattern.

---

## 2. Problem Statement

In a multi-agent pipeline, agents pass intermediate state (partial retrievals, draft embeddings, reranker candidates) to each other constantly. Today, teams solve this with:

- A shared cloud vector DB (Pinecone/Weaviate) — adds network round-trip latency and per-call cost to what should be an in-process handoff.
- Ad-hoc JSON blobs in Redis — requires deserializing full vectors on every read, and Redis's own vector support is heavier than this use case needs.

Neither is designed for **short-lived, high-churn, agent-local** vector data. Most of this data is dead within seconds of being written — it's a handoff, not a corpus. Existing vector stores are built to persist a corpus indefinitely; they aren't optimized for "write it, one other agent reads it once, then discard."

**Core problem:** there's no lightweight primitive for sub-millisecond, ephemeral, shared vector memory scoped to a single agent swarm or pipeline run.

---

## 3. Goals

1. Let an upstream agent `VSET` an embedding and a downstream agent `VSIMILARITY`/`VGET` it with no network hop and no serialization overhead beyond the existing TCP protocol.
2. Make written vectors **expire automatically** once a pipeline run or agent task completes, so memory doesn't grow unbounded across long-running swarms.
3. Let multiple agents share a Nearby instance without one agent's writes/queries interfering with another's (namespace isolation, per-agent visibility where needed).
4. Make it trivial to deploy Nearby as a sidecar container next to an agent process (already close today via the <15MB Docker image; needs docs + a reference manifest).
5. Give operators enough visibility (metrics, `INFO`) to trust an in-memory cache is behaving correctly in a production swarm — since data loss here is data loss, not just cache-miss.

## Non-Goals

- Replacing long-term/persistent vector storage for a RAG corpus (brute-force + HNSW over a stable corpus is already served by Nearby's existing use case — not in scope here to redesign).
- Distributed/multi-node Nearby (this PRD is single-instance-per-sidecar; a clustered Nearby is a separate, larger effort).
- Building an agent orchestration framework. Nearby is the memory substrate, not the orchestrator.

---

## 4. Target Users

| Persona | Need |
|---|---|
| Engineer building an agent swarm (Researcher/Coder/Reviewer, etc.) | Fast, local handoff of embeddings between agent steps without standing up a DB |
| RAG pipeline engineer (e.g., reranker-heavy systems like the "Scholar Query" pipeline referenced in the use case) | A RAM-level cache to hold hundreds of candidate embeddings between retrieval and reranking |
| Platform/infra engineer | A sidecar they can deploy per-agent-instance with minimal operational surface (no cluster, no schema migration) |

---

## 5. Use Case Scenarios

### 5.1 Agent-to-agent handoff
A Researcher agent processes a document chunk, embeds it, and issues `VSET swarm:run-42 chunk:7 1536 <vector> META stage researcher`. A Coder agent, working the same task, issues `VSIMILARITY swarm:run-42 1536 <query-vector> TOP 5` and gets the Researcher's output back in under a millisecond, with no serialization round-trip through a shared JSON store.

### 5.2 Reranker staging cache
A hybrid search pipeline retrieves 300 candidate documents. Instead of holding them in application memory or re-embedding on the reranker call, it pushes all 300 embeddings into a per-request Nearby namespace (`VSET rerank:req-9182 ...`), lets the cross-encoder pull batches via `VSIMILARITY`/`VGET`, and lets the whole namespace expire once the request completes.

### 5.3 Sidecar deployment
An agent runtime (e.g., a Kubernetes pod running a single agent instance) has a Nearby container colocated on `localhost`. The agent talks to `localhost:6379` with no service discovery, no network policy changes, no external dependency.

---

## 6. Functional Requirements

### 6.1 Namespace-scoped TTL (new)
- Extend `VSET` to accept an optional `EX <seconds>` on vector entries (mirroring the existing KV `SET ... EX`), so a single embedding can auto-expire.
- Add `VEXPIRE <namespace> <seconds>` to set a default TTL for an entire namespace (e.g., "everything in `swarm:run-42` dies in 300s"), so callers aren't required to set TTL per-vector during high-throughput writes.
- Expired vectors must be excluded from `VSIMILARITY` results and reclaimed from memory (lazy-on-access plus a background sweep, consistent with how KV `EXPIRE` likely already works).

### 6.2 Namespace lifecycle commands (new)
- `VNS DROP <namespace>` — explicit teardown of an entire agent-run's memory (called by the orchestrator when a swarm run completes, rather than waiting on TTL).
- `VNS LIST` — enumerate active namespaces, for debugging/observability when many concurrent agent runs share one instance.

### 6.3 Batch ingestion (already on roadmap — reprioritize)
- `VMSET` for bulk-loading candidate embeddings in one round trip (directly serves the reranker-staging scenario in 5.2, where hundreds of vectors land at once).

### 6.4 Existing protocol reuse (no change required)
- `VSET` / `VGET` / `VDEL` / `VCOUNT` for basic agent read/write.
- `VSIMILARITY` for downstream agents pulling relevant context.
- `VINDEX CREATE`/`SET_EF` for cases where an agent's scratchpad grows large enough (rare, but possible in long-lived swarms) to benefit from HNSW rather than brute-force.

### 6.5 Observability (new)
- Extend `INFO` to report per-namespace vector counts and approximate memory footprint, so operators can see which agent/run is consuming the most scratchpad memory.
- Prometheus metrics endpoint (already on roadmap) should expose: vectors written/sec, vectors expired/sec, `VSIMILARITY` p50/p99 latency, and active namespace count — the metrics that matter for a shared-memory-under-churn workload, as opposed to a stable-corpus workload.

### 6.6 Deployment (new, docs + manifest, not code)
- Reference `docker-compose.yml` and a Kubernetes sidecar container spec showing Nearby colocated with an agent process on `localhost`.
- A short guide: "Nearby as an agent sidecar" — covering TTL configuration, namespace-per-run conventions, and when to promote to HNSW.

---

## 7. Non-Functional Requirements

- **Latency:** `VSET`/`VSIMILARITY` round trip must stay in the sub-millisecond to low-single-digit-millisecond range under the same conditions as existing benchmarks (this is the entire value proposition; regressions here undermine the use case).
- **Memory safety:** TTL/expiry must be reliable enough that a forgotten `VNS DROP` doesn't leak memory indefinitely in a long-running swarm — this is now a correctness requirement, not just a nice-to-have, because the workload is inherently high-churn.
- **Concurrency:** namespace-scoped TTL and `VNS DROP` must be implemented under the existing `sync.RWMutex` store discipline and pass `go test -race`, consistent with current testing practice.
- **No new external dependencies.** Stays a single static binary; sidecar image should stay in the same size class as today's <15MB image.

---

## 8. Success Metrics

- p50 agent-to-agent handoff latency (`VSET` → downstream `VSIMILARITY`) under 1ms at typical swarm scratchpad sizes (hundreds to low thousands of vectors per namespace).
- Memory usage returns to baseline after namespace TTL/`VNS DROP`, verified under a soak test simulating a long-running swarm with many short-lived runs.
- At least one reference integration (e.g., an update to `examples/rag_demo` or a new `examples/agent_swarm_demo`) demonstrating the handoff and reranker-staging patterns end-to-end.

---

## 9. Risks & Open Questions

- **Namespace collisions across concurrent agent runs:** need a convention (e.g., `swarm:<run-id>`) but should this be enforced by Nearby or left to caller discipline? Leaning toward caller discipline plus `VNS LIST` for visibility, to avoid adding schema/validation logic to a deliberately thin protocol.
- **TTL sweep cost:** a background sweeper needs to avoid contending with the similarity worker pool under load — needs benchmarking before landing.
- **Is per-vector TTL or per-namespace TTL the primary interface?** Per-namespace is likely the common case (whole run expires together) but per-vector TTL may matter for partially-stale scratchpads. Current plan is to support both, defaulting callers toward namespace-level.
- **Single-instance limit:** if a swarm scales past what one Nearby sidecar can hold in RAM, there's no clustering story yet. Out of scope here, but worth flagging as the natural next PRD.

---

## 10. Out of Scope / Future Work

- Multi-node/clustered Nearby for swarms too large for one sidecar's memory.
- Access control between agents sharing one instance (currently only the existing `--password` token auth, which is all-or-nothing).
- Dot-product similarity mode (`METHOD DOT`) — already on the general roadmap, orthogonal to this use case but would benefit reranker scenarios using non-normalized embeddings.