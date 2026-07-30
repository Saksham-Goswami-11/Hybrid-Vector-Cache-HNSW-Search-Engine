Act as a Principal AI Infrastructure Engineer.
Context: My root directory contains the open-source Microsoft AutoGen repo, plus my own project, Nearby an ultra-fast, in-memory Go vector cache exposing a custom Redis-RESP-style TCP text protocol on localhost:6379. (You may see it referenced elsewhere as sypnase-cache same engine, treat as identical.) Nearby has been stress-tested at 600 concurrent goroutines / 100k+ vectors with zero errors or race conditions TTL expiry (VSET ... EX), batch writes (VMSET), and namespace teardown (VNS DROP) are all confirmed working under load, not experimental.
Goal: Integrate Nearby into AutoGen as a drop-in replacement for its default ChromaDB-backed retrieval, then benchmark the swap on a real multi-agent run.
Step 0 Confirm exact protocol syntax before writing any client code.
Read nearby's actual source (internal/protocol/, internal/server/server.go) to nail down the precise request/response grammar for VSET (including EX), VMSET, VSIMILARITY, VGET, and VNS DROP. TTL and batch-write support are confirmed to exist this step is about matching exact syntax and reply framing, not verifying existence. If the README example differs from the actual source, source wins; tell me about the discrepancy.
Step 1 Vector store client (nearby_memory.py)
Write a NearbyVectorStore class using Python's native socket module (no external RESP libraries) that talks to localhost:6379.
Required behavior:
ingest(namespace, id, vector, metadata=None, ttl=None) → sends VSET with TTL.
ingest_batch(namespace, items, ttl=None) → sends VMSET for bulk loads; use this path for document ingestion in Step 2 rather than looping single VSET calls.
query(namespace, vector, top_k) → sends VSIMILARITY, parses the actual RESP-style reply into (id, score) tuples. Parse against real captured server output, not just the README example.
drop_namespace(namespace) → sends VNS DROP; used by the benchmark script to guarantee a clean slate between Run A and Run B.
Config via environment variables (NEARBY_HOST, NEARBY_PORT, NEARBY_TIMEOUT_MS).
Explicit socket timeout on connect and recv; typed NearbyConnectionError / NearbyProtocolError instead of raw socket exceptions.
Connection reuse (no open/close per call), retry-once-on-disconnect.
Step 2 AutoGen integration
Subclass AutoGen's RetrieveUserProxyAgent (confirm the exact hook methods for ingestion and retrieval in the installed AutoGen version these change across releases). Override ingestion to batch-embed text via sentence-transformers (default, offline) or OpenAI's text-embedding-3-small (if OPENAI_API_KEY is set), then push vectors through NearbyVectorStore.ingest_batch. Keep the embedding backend swappable behind one config flag.
Step 3 Benchmark script (benchmark_swarm.py)
Build a 3-agent AutoGen swarm (User Proxy, Researcher, Writer) and run the identical task twice:
Run A (baseline): default AutoGen + ChromaDB retrieval.
Run B (Nearby): same agents, same task, same embedding model, swapped retrieval backend only. Call drop_namespace before Run B starts so no state carries over from setup/warm-up.
Instrument both runs and report P50/P90/P99/Max, not just averages, for: (a) document ingestion, (b) per-query context retrieval, (c) total wall-clock run time mirroring the percentile methodology already used in the Nearby stress test, so the two result sets are comparable and you're not quietly hiding tail latency behind a good median.
Constraints:
Vector dimension must match the active embedding model (1536 for text-embedding-3-small; auto-adjust if using sentence-transformers).
Modular files, no single monolithic script.
Fail loudly and specifically on socket/connection errors no bare except: pass.
If AutoGen's retrieval hook signatures don't match what's assumed above, stop and report the actual signatures rather than guessing.