"""
Nearby Vector Memory Integration for Microsoft AutoGen v0.4+

Provides:
1. NearbyVectorStore: Pure Python socket client (RESP-compatible) for Nearby vector engine.
2. NearbyVectorMemory: AutoGen Memory component subclass for seamless vector retrieval.
"""

from __future__ import annotations

import os
import socket
import sys
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from autogen_core import CancellationToken, Component
from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage

logger = logging.getLogger(__name__)

# --- Custom Exception Hierarchy ---

class NearbyError(Exception):
    """Base exception for Nearby vector store operations."""

class NearbyConnectionError(NearbyError):
    """Raised when connection to Nearby TCP server fails."""

class NearbyProtocolError(NearbyError):
    """Raised when protocol parsing or server error response occurs."""

def _format_token(val: Any) -> str:
    s = str(val)
    if " " in s or '"' in s or "\n" in s or "\t" in s:
        escaped = s.replace('"', '\\"')
        return f'"{escaped}"'
    return s

# --- Step 1: Low-Level Socket Client for Nearby ---

class NearbyVectorStore:
    """
    Pure-Python socket client for Nearby vector cache (localhost:6379).
    Implements RESP text protocol without external dependencies.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ) -> None:
        self.host = host or os.getenv("NEARBY_HOST", "localhost")
        self.port = port or int(os.getenv("NEARBY_PORT", "6379"))
        timeout_ms_val = timeout_ms or int(os.getenv("NEARBY_TIMEOUT_MS", "5000"))
        self.timeout_sec = timeout_ms_val / 1000.0
        self._sock: Optional[socket.socket] = None

    def _connect(self) -> socket.socket:
        """Establish or reuse TCP connection."""
        if self._sock is not None:
            return self._sock
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_sec)
            sock.connect((self.host, self.port))
            self._sock = sock
            return sock
        except Exception as e:
            self._sock = None
            raise NearbyConnectionError(f"Failed to connect to Nearby at {self.host}:{self.port}: {e}") from e

    def _close(self) -> None:
        """Close socket connection cleanly."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send_cmd(self, cmd_str: str) -> str:
        """Send command string and read full response with 1 retry on disconnect."""
        cmd = cmd_str.rstrip("\r\n") + "\r\n"
        data_bytes = cmd.encode("utf-8")

        for attempt in range(2):
            try:
                sock = self._connect()
                sock.sendall(data_bytes)
                return self._read_resp(sock)
            except (socket.error, NearbyConnectionError) as e:
                self._close()
                if attempt == 1:
                    raise NearbyConnectionError(f"Connection failed after retry: {e}") from e
            except Exception as e:
                if isinstance(e, NearbyError):
                    raise
                raise NearbyProtocolError(f"Protocol error executing '{cmd_str[:50]}...': {e}") from e

        raise NearbyConnectionError("Failed to communicate with Nearby server.")

    def _read_resp(self, sock: socket.socket) -> str:
        """Reads a complete RESP response from socket."""
        rfile = sock.makefile("rb")
        try:
            return self._parse_resp_value(rfile)
        finally:
            rfile.close()

    def _readline(self, rfile) -> str:
        line = rfile.readline()
        if not line:
            raise NearbyConnectionError("Server closed connection unexpectedly.")
        return line.decode("utf-8").rstrip("\r\n")

    def _parse_resp_value(self, rfile) -> str:
        """Parses RESP data into string representation or raw JSON-like string."""
        line = self._readline(rfile)
        if not line:
            raise NearbyProtocolError("Empty response line from Nearby server.")

        prefix = line[0]
        body = line[1:]

        if prefix == "+":  # SimpleString (+OK)
            return body
        elif prefix == "-":  # ErrorResponse (-ERR msg)
            raise NearbyProtocolError(f"Nearby Server Error: {body}")
        elif prefix == ":":  # IntegerResponse (:1)
            return body
        elif prefix == "$":  # BulkString ($len\r\nval\r\n)
            length = int(body)
            if length == -1:
                return ""
            data = rfile.read(length)
            rfile.read(2)  # consume \r\n
            return data.decode("utf-8")
        elif prefix == "*":  # ArrayResponse (*count\r\n...)
            count = int(body)
            if count == -1:
                return "[]"
            items = []
            for _ in range(count):
                items.append(self._parse_resp_value(rfile))
            return str(items)
        else:
            return line

    def ingest(
        self,
        namespace: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, str]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """Send VSET command with optional metadata and TTL."""
        dim = len(vector)
        parts = ["VSET", namespace, id, str(dim)]
        for f in vector:
            parts.append(f"{f:.6f}")

        if metadata:
            parts.append("META")
            for k, v in metadata.items():
                parts.extend([_format_token(k), _format_token(v)])

        if ttl is not None and ttl > 0:
            parts.extend(["EX", str(ttl)])

        cmd = " ".join(parts)
        resp = self._send_cmd(cmd)
        return resp == "OK"

    def ingest_batch(
        self,
        namespace: str,
        items: List[Tuple[str, List[float], Optional[Dict[str, str]]]],
        ttl: Optional[int] = None,
        chunk_size: int = 25,
    ) -> int:
        """Send VMSET command for bulk vector ingestion, automatically chunking large batches."""
        if not items:
            return 0

        total_stored = 0
        # Chunk items to prevent exceeding TCP line length limit (64KB)
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            dim = len(chunk[0][1])
            count = len(chunk)

            parts = ["VMSET", namespace, str(dim), str(count)]
            for item_id, vector, metadata in chunk:
                if len(vector) != dim:
                    raise NearbyProtocolError(f"Dimension mismatch in batch: expected {dim}, got {len(vector)}")
                parts.append(str(item_id))
                for f in vector:
                    parts.append(f"{f:.6f}")

                if metadata:
                    parts.append("META")
                    for k, v in metadata.items():
                        parts.extend([_format_token(k), _format_token(v)])

            cmd = " ".join(parts)
            resp = self._send_cmd(cmd)

            try:
                stored_count = int(resp)
            except ValueError:
                stored_count = count
            total_stored += stored_count

        # If namespace-level TTL is requested
        if ttl is not None and ttl > 0:
            self._send_cmd(f"VEXPIRE {namespace} {ttl}")

        return total_stored

    def query(
        self,
        namespace: str,
        vector: List[float],
        top_k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, str]]]:
        """Send VSIMILARITY command and parse reply into (id, score, metadata) list."""
        dim = len(vector)
        parts = ["VSIMILARITY", namespace, str(dim)]
        for f in vector:
            parts.append(f"{f:.6f}")
        parts.extend(["TOP", str(top_k)])

        cmd = " ".join(parts)

        # Parse raw socket response directly for exact structure
        sock = self._connect()
        try:
            sock.sendall((cmd + "\r\n").encode("utf-8"))
            rfile = sock.makefile("rb")
            try:
                line = self._readline(rfile)
                if not line.startswith("*"):
                    if line.startswith("-"):
                        raise NearbyProtocolError(f"Server error: {line[1:]}")
                    return []

                num_items = int(line[1:])
                if num_items <= 0:
                    return []

                results = []
                # Result items come in triplets: [ID (Bulk), Score (Simple), Metadata (Array)]
                for _ in range(num_items // 3):
                    # 1. ID
                    id_val = self._parse_resp_value(rfile)

                    # 2. Score
                    score_line = self._readline(rfile)
                    score_val = 0.0
                    if score_line.startswith("+"):
                        try:
                            score_val = float(score_line[1:])
                        except ValueError:
                            pass

                    # 3. Metadata Array
                    meta_line = self._readline(rfile)
                    metadata: Dict[str, str] = {}
                    if meta_line.startswith("*"):
                        meta_count = int(meta_line[1:])
                        for _ in range(meta_count // 2):
                            mk = self._parse_resp_value(rfile)
                            mv = self._parse_resp_value(rfile)
                            metadata[mk] = mv

                    results.append((id_val, score_val, metadata))

                return results
            finally:
                rfile.close()
        except NearbyError:
            raise
        except Exception as e:
            self._close()
            raise NearbyProtocolError(f"Error querying VSIMILARITY: {e}") from e

    def drop_namespace(self, namespace: str) -> bool:
        """Send VNS DROP to cleanly tear down namespace."""
        resp = self._send_cmd(f"VNS DROP {namespace}")
        return resp == "1"

# --- Step 2: AutoGen Memory Component Integration ---

class NearbyVectorMemoryConfig(BaseModel):
    """Configuration for Nearby-backed vector memory in AutoGen."""
    host: str = Field(default="localhost", description="Nearby server host")
    port: int = Field(default=6379, description="Nearby server port")
    namespace: str = Field(default="autogen_memory", description="Vector namespace")
    k: int = Field(default=5, description="Number of memories to retrieve")
    ttl_seconds: Optional[int] = Field(default=None, description="Optional TTL in seconds")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name (or 'openai/text-embedding-3-small')",
    )

class NearbyVectorMemory(Memory, Component[NearbyVectorMemoryConfig]):
    """
    Drop-in AutoGen v0.4 Memory implementation backed by Nearby vector cache.
    """

    component_config_schema = NearbyVectorMemoryConfig
    component_provider_override = "nearby_memory.NearbyVectorMemory"

    def __init__(self, config: Optional[NearbyVectorMemoryConfig] = None) -> None:
        self._config = config or NearbyVectorMemoryConfig()
        self._store = NearbyVectorStore(
            host=self._config.host,
            port=self._config.port,
        )
        self._embedder = None
        self._dim = 384  # Default for MiniLM-L6-v2

    def _get_embedder(self):
        """Lazy initialization of embedding function."""
        if self._embedder is not None:
            return self._embedder

        model_name = self._config.embedding_model
        if model_name.startswith("openai/") or os.getenv("OPENAI_API_KEY") and "text-embedding" in model_name:
            import openai
            self._dim = 1536
            real_model = model_name.replace("openai/", "")

            def _embed_openai(text: str) -> List[float]:
                client = openai.OpenAI()
                resp = client.embeddings.create(input=text, model=real_model)
                return resp.data[0].embedding

            self._embedder = _embed_openai
        else:
            try:
                from sentence_transformers import SentenceTransformer
                st_model = SentenceTransformer(model_name)
                self._dim = st_model.get_sentence_embedding_dimension()

                def _embed_st(text: str) -> List[float]:
                    vec = st_model.encode(text, convert_to_numpy=True)
                    return vec.tolist()

                self._embedder = _embed_st
            except ImportError:
                # Fallback simple deterministic embedder for testing if sentence-transformers unavailable
                self._dim = 128
                def _embed_fallback(text: str) -> List[float]:
                    import math
                    vec = [0.0] * 128
                    for i, char in enumerate(text[:128]):
                        vec[i % 128] += ord(char)
                    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                    return [x / norm for x in vec]

                self._embedder = _embed_fallback

        return self._embedder

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> None:
        embed_fn = self._get_embedder()
        text_str = str(content.content)
        vec = embed_fn(text_str)

        mem_id = str(uuid.uuid4())[:8]
        metadata = content.metadata or {}
        metadata["mime_type"] = str(content.mime_type)
        metadata["raw_content"] = text_str[:250]  # Store preview in metadata

        self._store.ingest(
            namespace=self._config.namespace,
            id=mem_id,
            vector=vec,
            metadata=metadata,
            ttl=self._config.ttl_seconds,
        )

    async def add_batch(self, contents: List[MemoryContent]) -> int:
        """Batch ingestion using Nearby's VMSET command."""
        if not contents:
            return 0
        embed_fn = self._get_embedder()

        batch_items = []
        for c in contents:
            text_str = str(c.content)
            vec = embed_fn(text_str)
            mem_id = str(uuid.uuid4())[:8]
            meta = c.metadata or {}
            meta["mime_type"] = str(c.mime_type)
            meta["raw_content"] = text_str[:250]
            batch_items.append((mem_id, vec, meta))

        return self._store.ingest_batch(
            namespace=self._config.namespace,
            items=batch_items,
            ttl=self._config.ttl_seconds,
        )

    async def query(
        self,
        query: Union[str, MemoryContent],
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        query_text = str(query.content) if isinstance(query, MemoryContent) else str(query)
        if not query_text.strip():
            return MemoryQueryResult(results=[])

        embed_fn = self._get_embedder()
        query_vec = embed_fn(query_text)

        top_k = kwargs.get("top_k", self._config.k)
        raw_results = self._store.query(
            namespace=self._config.namespace,
            vector=query_vec,
            top_k=top_k,
        )

        memories: List[MemoryContent] = []
        for item_id, score, meta in raw_results:
            raw_text = meta.get("raw_content", item_id)
            meta["score"] = str(score)
            meta["id"] = item_id

            memories.append(
                MemoryContent(
                    content=raw_text,
                    mime_type=MemoryMimeType.TEXT,
                    metadata=meta,
                )
            )

        return MemoryQueryResult(results=memories)

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last_msg = str(messages[-1].content)
        query_res = await self.query(last_msg)

        if query_res.results:
            ctx_text = "\n[Nearby Vector Memory Context]:\n" + "\n".join(
                f"• {m.content} (score: {m.metadata.get('score', 'N/A')})" for m in query_res.results
            )
            await model_context.add_message(SystemMessage(content=ctx_text))

        return UpdateContextResult(memories=query_res)

    async def clear(self) -> None:
        self._store.drop_namespace(self._config.namespace)

    async def close(self) -> None:
        self._store.drop_namespace(self._config.namespace)
        self._store._close()

    def _to_config(self) -> NearbyVectorMemoryConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: NearbyVectorMemoryConfig) -> NearbyVectorMemory:
        return cls(config=config)
