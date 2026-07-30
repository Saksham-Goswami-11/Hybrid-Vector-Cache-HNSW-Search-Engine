import time
import os
from nearby_memory import (
    NearbyVectorStore,
    NearbyVectorMemory,
    NearbyVectorMemoryConfig,
    NearbyConnectionError,
    NearbyProtocolError,
)
from autogen_core.memory import MemoryContent, MemoryMimeType

def test_nearby_vector_store_basic():
    store = NearbyVectorStore(host="localhost", port=6379)
    ns = "test_py_client"

    # Clean up before
    store.drop_namespace(ns)

    # Ingest single
    ok = store.ingest(ns, "v1", [0.1, 0.2, 0.3, 0.4], metadata={"doc": "manual"}, ttl=60)
    assert ok is True

    # Query
    results = store.query(ns, [0.1, 0.2, 0.3, 0.4], top_k=1)
    assert len(results) == 1
    assert results[0][0] == "v1"
    assert results[0][2]["doc"] == "manual"

    # Batch ingest
    batch_items = [
        ("v2", [0.0, 1.0, 0.0, 0.0], {"batch": "true"}),
        ("v3", [0.0, 0.0, 1.0, 0.0], {"batch": "true"}),
    ]
    stored = store.ingest_batch(ns, batch_items, ttl=60)
    assert stored == 2

    # Query batch
    results = store.query(ns, [0.0, 1.0, 0.0, 0.0], top_k=2)
    assert len(results) >= 1
    assert results[0][0] == "v2"

    # Drop namespace
    dropped = store.drop_namespace(ns)
    assert dropped is True

    # Query after drop
    results = store.query(ns, [0.1, 0.2, 0.3, 0.4], top_k=1)
    assert len(results) == 0
    print("✅ NearbyVectorStore socket test PASSED!")

if __name__ == "__main__":
    test_nearby_vector_store_basic()
