import os
import hashlib
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from openai import OpenAI
from client import SynapseClient
from ingest import extract_text, chunk_text, KNOWLEDGE_BASE_DIR, NEARBY_PORT

openai_client = None
if os.environ.get("OPENAI_API_KEY"):
    openai_client = OpenAI()

class HybridRetriever:
    def __init__(self, directory: str = KNOWLEDGE_BASE_DIR, port: int = NEARBY_PORT):
        self.directory = os.path.abspath(directory)
        self.client = SynapseClient(port=port)
        self._corpus_cache: Dict[str, Dict[str, Any]] = {}  # filepath -> {mtime, chunks}
        self._bm25: Any = None
        self._keys: List[str] = []
        self._texts: List[str] = []

    def _rebuild_index_if_needed(self):
        files = []
        for root, _, filenames in os.walk(self.directory):
            for f in filenames:
                if not f.startswith('.') and f.lower().endswith(('.md', '.txt', '.pdf')):
                    files.append(os.path.join(root, f))
        
        rebuild_needed = False
        new_cache = {}
        for f in files:
            mtime = os.path.getmtime(f)
            if f not in self._corpus_cache or self._corpus_cache[f]['mtime'] != mtime:
                rebuild_needed = True
                text = extract_text(f)
                chunks = chunk_text(text)
                new_cache[f] = {'mtime': mtime, 'chunks': chunks}
            else:
                new_cache[f] = self._corpus_cache[f]

        if rebuild_needed or len(self._corpus_cache) != len(new_cache):
            self._corpus_cache = new_cache
            self._texts = []
            self._keys = []
            for filepath, data in self._corpus_cache.items():
                filename = os.path.basename(filepath)
                for idx, chunk in enumerate(data['chunks']):
                    self._texts.append(chunk)
                    self._keys.append(f"chunk:{filename}:{idx}")
            
            if self._texts:
                tokenized_corpus = [doc.lower().split() for doc in self._texts]
                self._bm25 = BM25Okapi(tokenized_corpus)
            else:
                self._bm25 = None

    def _embed_query(self, query: str) -> List[float]:
        if not openai_client:
            # Match the deterministic hashing used in ingest.py
            h = hashlib.sha256(query.encode()).digest()
            vec = []
            for idx in range(1536):
                val = float(h[idx % 32]) / 256.0
                vec.append(val)
            return vec

        resp = openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        return resp.data[0].embedding

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        self._rebuild_index_if_needed()

        # --- 1. Vector Search (Semantic) ---
        vector_results = []
        try:
            query_vector = self._embed_query(query)
            # Query top 10 semantic matches
            vector_results = self.client.vsimilarity("docs", query_vector, k=10)
        except Exception as e:
            print(f"Vector search failed: {e}")

        # --- 2. Keyword Search (BM25) ---
        bm25_results: List[Tuple[str, float]] = []
        if self._bm25 and query.strip():
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            
            scored_keys = list(zip(self._keys, scores))
            scored_keys.sort(key=lambda x: x[1], reverse=True)
            bm25_results = scored_keys[:10]

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        rrf_scores: Dict[str, float] = {}

        # Process vector ranks
        for rank, res in enumerate(vector_results, start=1):
            key = res["id"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60.0 + rank))

        # Process BM25 ranks
        for rank, (key, score) in enumerate(bm25_results, start=1):
            if score > 0:
                rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60.0 + rank))

        # Sort all candidates by RRF score descending
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        top_5_fused = fused_results[:5]

        # --- 4. Fetch payload from Synapse Cache ---
        final_results = []
        for key, rrf_score in top_5_fused:
            try:
                text = self.client.get(key)
                if not text:
                    continue
                
                parts = key.split(":")
                source = parts[1] if len(parts) > 1 else "unknown"
                
                final_results.append({
                    "key": key,
                    "text": text,
                    "source": source,
                    "rrf_score": rrf_score
                })
            except Exception as e:
                print(f"Failed to fetch chunk text for key {key}: {e}")

        return final_results
