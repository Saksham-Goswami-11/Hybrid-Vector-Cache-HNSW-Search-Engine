"""
Benchmark Swarm Script — AutoGen v0.4 Vector Memory Comparison:
Run A1: AutoGen + ChromaDB (Single-Insert Loop)
Run A2: AutoGen + ChromaDB (Batch Insert)
Run B: AutoGen + NearbyVectorMemory (Single Ingest VSET)
Run C: AutoGen + NearbyVectorMemory (Batch Ingest VMSET)

Instruments & Reports Percentiles (P50, P90, P99, Max):
- Document Ingestion Latency (Single vs Batch for both ChromaDB and Nearby)
- Per-Query Context Retrieval Latency
- Total Wall-Clock Run Time
"""

import asyncio
import os
import time
import math
import numpy as np
from typing import List, Dict, Tuple

from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
)
from nearby_memory import NearbyVectorMemory, NearbyVectorMemoryConfig, NearbyVectorStore

# --- Fast Local Deterministic Embedding Function ---

class FastEmbeddingFunction:
    """128-dimensional fast deterministic embedding for zero-network-overhead benchmarking."""
    def __init__(self, dim: int = 128):
        self.dim = dim

    def _embed(self, input: List[str]) -> List[List[float]]:
        results = []
        for text in input:
            vec = [0.0] * self.dim
            for i, char in enumerate(text[:self.dim]):
                vec[i % self.dim] += float(ord(char))
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / norm for x in vec])
        return results

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

# --- Sample Dataset for Swarm Context (100 Documents) ---

CORPUS_DOCUMENTS = [
    f"Document Chunk #{i}: Nearby Vector Engine provides sub-millisecond in-memory vector storage for multi-agent swarms."
    for i in range(100)
]

TEST_QUERIES = [
    "How does batch ingestion work in Nearby?",
    "What is the TTL sweep interval for expiring vectors?",
    "How does Nearby achieve sub-millisecond latency?",
    "Explain namespace teardown and life-cycle management.",
    "What memory protocol does AutoGen use for RAG?",
]

def compute_percentiles(durations_ms: List[float]) -> Dict[str, float]:
    if not durations_ms:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.array(durations_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }

def print_metrics(title: str, stats: Dict[str, float]) -> None:
    print(f"  {title:<35} | P50: {stats['p50']:8.3f}ms | P90: {stats['p90']:8.3f}ms | P99: {stats['p99']:8.3f}ms | Max: {stats['max']:8.3f}ms")

async def run_benchmark():
    print("==========================================================================================")
    print("      🔥 AUTOGEN RETRIEVAL BACKEND BENCHMARK: CHROMADB VS NEARBY MEMORY GRID              ")
    print("==========================================================================================")

    embedder = FastEmbeddingFunction(dim=128)

    corpus_contents = [
        MemoryContent(content=text, mime_type=MemoryMimeType.TEXT, metadata={"doc_id": f"doc_{i}"})
        for i, text in enumerate(CORPUS_DOCUMENTS)
    ]

    import chromadb

    # --------------------------------------------------------------------------------------------
    # RUN A1: BASELINE (ChromaDB Single-Insert Loop)
    # --------------------------------------------------------------------------------------------
    print("\n------------------------------------------------------------------------------------------")
    print("🔵 RUN A1: AutoGen + ChromaDB (Single-Insert Loop Baseline)")
    print("------------------------------------------------------------------------------------------")

    chroma_client = chromadb.Client()
    col_single = chroma_client.create_collection(name="bench_chroma_single", embedding_function=embedder)

    chroma_ingest_times: List[float] = []
    t_start_a1 = time.perf_counter()

    for i, item in enumerate(corpus_contents):
        t0 = time.perf_counter()
        col_single.add(
            documents=[str(item.content)],
            metadatas=[item.metadata or {}],
            ids=[f"id_{i}"],
        )
        t1 = time.perf_counter()
        chroma_ingest_times.append((t1 - t0) * 1000.0)

    chroma_query_times: List[float] = []
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        results = col_single.query(query_texts=[q], n_results=3)
        t1 = time.perf_counter()
        chroma_query_times.append((t1 - t0) * 1000.0)
        assert len(results["documents"][0]) > 0

    t_end_a1 = time.perf_counter()
    total_time_a1 = (t_end_a1 - t_start_a1) * 1000.0

    chroma_ingest_stats = compute_percentiles(chroma_ingest_times)
    chroma_query_stats = compute_percentiles(chroma_query_times)

    print_metrics("ChromaDB Single Ingest (Loop)", chroma_ingest_stats)
    print_metrics("ChromaDB Context Retrieval", chroma_query_stats)
    print(f"  Total Wall-Clock Run Time          : {total_time_a1:.2f} ms")

    # --------------------------------------------------------------------------------------------
    # RUN A2: BASELINE (ChromaDB Native Batch Ingest)
    # --------------------------------------------------------------------------------------------
    print("\n------------------------------------------------------------------------------------------")
    print("🔵 RUN A2: AutoGen + ChromaDB (Native Batch Ingest)")
    print("------------------------------------------------------------------------------------------")

    col_batch = chroma_client.create_collection(name="bench_chroma_batch", embedding_function=embedder)

    t_start_a2 = time.perf_counter()
    t0_chroma_batch = time.perf_counter()
    col_batch.add(
        documents=[str(item.content) for item in corpus_contents],
        metadatas=[item.metadata or {} for item in corpus_contents],
        ids=[f"id_{i}" for i in range(len(corpus_contents))],
    )
    t1_chroma_batch = time.perf_counter()
    chroma_batch_time_ms = (t1_chroma_batch - t0_chroma_batch) * 1000.0

    chroma_batch_query_times: List[float] = []
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        results = col_batch.query(query_texts=[q], n_results=3)
        t1 = time.perf_counter()
        chroma_batch_query_times.append((t1 - t0) * 1000.0)
        assert len(results["documents"][0]) > 0

    t_end_a2 = time.perf_counter()
    total_time_a2 = (t_end_a2 - t_start_a2) * 1000.0
    chroma_batch_query_stats = compute_percentiles(chroma_batch_query_times)

    print(f"  ChromaDB Batch Ingest (100 docs)    | Total: {chroma_batch_time_ms:8.3f}ms | Avg/doc: {chroma_batch_time_ms/100:8.3f}ms")
    print_metrics("ChromaDB Batch Context Retrieval", chroma_batch_query_stats)
    print(f"  Total Wall-Clock Run Time          : {total_time_a2:.2f} ms")

    # --------------------------------------------------------------------------------------------
    # RUN B: NEARBY VECTOR CACHE (Single Ingest VSET)
    # --------------------------------------------------------------------------------------------
    print("\n------------------------------------------------------------------------------------------")
    print("⚡ RUN B: AutoGen + NearbyVectorMemory (Single Ingest VSET)")
    print("------------------------------------------------------------------------------------------")

    cleaner_store = NearbyVectorStore(host="localhost", port=6379)
    cleaner_store.drop_namespace("bench_nearby_single")
    cleaner_store._close()

    nearby_memory_single = NearbyVectorMemory(
        config=NearbyVectorMemoryConfig(
            host="localhost",
            port=6379,
            namespace="bench_nearby_single",
            k=3,
        )
    )
    nearby_memory_single._embedder = lambda text: embedder([text])[0]

    nearby_ingest_times: List[float] = []
    t_start_b = time.perf_counter()

    for item in corpus_contents:
        t0 = time.perf_counter()
        await nearby_memory_single.add(item)
        t1 = time.perf_counter()
        nearby_ingest_times.append((t1 - t0) * 1000.0)

    nearby_query_times: List[float] = []
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        res = await nearby_memory_single.query(q)
        t1 = time.perf_counter()
        nearby_query_times.append((t1 - t0) * 1000.0)
        assert len(res.results) > 0

    t_end_b = time.perf_counter()
    total_time_b = (t_end_b - t_start_b) * 1000.0

    await nearby_memory_single.close()

    nearby_ingest_stats = compute_percentiles(nearby_ingest_times)
    nearby_query_stats = compute_percentiles(nearby_query_times)

    print_metrics("Nearby Single Ingestion (VSET)", nearby_ingest_stats)
    print_metrics("Nearby Context Retrieval", nearby_query_stats)
    print(f"  Total Wall-Clock Run Time          : {total_time_b:.2f} ms")

    # --------------------------------------------------------------------------------------------
    # RUN C: NEARBY VECTOR CACHE (Batch Ingest VMSET)
    # --------------------------------------------------------------------------------------------
    print("\n------------------------------------------------------------------------------------------")
    print("🚀 RUN C: AutoGen + NearbyVectorMemory (Batch Ingest VMSET)")
    print("------------------------------------------------------------------------------------------")

    cleaner_store = NearbyVectorStore(host="localhost", port=6379)
    cleaner_store.drop_namespace("bench_nearby_batch")
    cleaner_store._close()

    nearby_memory_batch = NearbyVectorMemory(
        config=NearbyVectorMemoryConfig(
            host="localhost",
            port=6379,
            namespace="bench_nearby_batch",
            k=3,
        )
    )
    nearby_memory_batch._embedder = lambda text: embedder([text])[0]

    t_start_c = time.perf_counter()
    t0_batch = time.perf_counter()
    count = await nearby_memory_batch.add_batch(corpus_contents)
    t1_batch = time.perf_counter()
    nearby_batch_time_ms = (t1_batch - t0_batch) * 1000.0

    batch_query_times: List[float] = []
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        res = await nearby_memory_batch.query(q)
        t1 = time.perf_counter()
        batch_query_times.append((t1 - t0) * 1000.0)
        assert len(res.results) > 0

    t_end_c = time.perf_counter()
    total_time_c = (t_end_c - t_start_c) * 1000.0

    await nearby_memory_batch.close()

    batch_query_stats = compute_percentiles(batch_query_times)

    print(f"  Nearby Batch Ingest (100 docs)     | Total: {nearby_batch_time_ms:8.3f}ms | Avg/doc: {nearby_batch_time_ms/100:8.3f}ms")
    print_metrics("Nearby Batch Context Retrieval", batch_query_stats)
    print(f"  Total Wall-Clock Run Time          : {total_time_c:.2f} ms")

    # --------------------------------------------------------------------------------------------
    # SPEEDUP COMPARISON SUMMARY
    # --------------------------------------------------------------------------------------------
    print("\n==========================================================================================")
    print("                               🏆 BENCHMARK COMPARISON SUMMARY                            ")
    print("==========================================================================================")

    single_ingest_speedup = chroma_ingest_stats['p50'] / nearby_ingest_stats['p50'] if nearby_ingest_stats['p50'] > 0 else 1.0
    batch_vs_batch_speedup = chroma_batch_time_ms / nearby_batch_time_ms if nearby_batch_time_ms > 0 else 1.0
    query_speedup = chroma_query_stats['p50'] / nearby_query_stats['p50'] if nearby_query_stats['p50'] > 0 else 1.0

    print(f"  Single-vs-Single Ingest (P50):  {single_ingest_speedup:.2f}x faster (Nearby {nearby_ingest_stats['p50']:.3f}ms vs Chroma {chroma_ingest_stats['p50']:.3f}ms)")
    print(f"  Batch-vs-Batch Ingest (Total):  {batch_vs_batch_speedup:.2f}x faster (Nearby {nearby_batch_time_ms:.3f}ms vs Chroma {chroma_batch_time_ms:.3f}ms)")
    print(f"  Retrieval Query (P50):          {query_speedup:.2f}x faster (Nearby {nearby_query_stats['p50']:.3f}ms vs Chroma {chroma_query_stats['p50']:.3f}ms)")
    print(f"  Total Wall-Clock Speedup:       {total_time_a2 / total_time_c:.2f}x faster (Nearby Batch vs Chroma Batch Pipeline)")
    print("==========================================================================================")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
