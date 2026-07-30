import time
import os
from nearby_memory import (
    NearbyVectorStore,
    NearbyVectorMemory,
    NearbyVectorMemoryConfig,
    NearbyConnectionError,
    NearbyProtocolError,
    NearbyAuthError,
    _format_token,
)
from autogen_core.memory import MemoryContent, MemoryMimeType

def test_format_token_injection_guard():
    """Verify that \\r and \\n in metadata values trigger ValueError."""
    try:
        _format_token("valid_key\r\nDEL important_key\r\n")
        assert False, "Should have raised ValueError on \\r\\n"
    except ValueError as e:
        assert "Metadata keys and values cannot contain newline characters" in str(e)

    try:
        _format_token("injected\nval")
        assert False, "Should have raised ValueError on \\n"
    except ValueError as e:
        assert "Metadata keys and values cannot contain newline characters" in str(e)

    try:
        _format_token("injected\rval")
        assert False, "Should have raised ValueError on \\r"
    except ValueError as e:
        assert "Metadata keys and values cannot contain newline characters" in str(e)

    assert _format_token("hello world") == '"hello world"'
    assert _format_token('quote"test') == '"quote\\"test"'
    assert _format_token("simple") == "simple"
    print("✅ _format_token injection guard test PASSED!")

def test_nearby_vector_store_basic():
    store = NearbyVectorStore(host="localhost", port=6379)
    ns = "test_py_client"

    store.drop_namespace(ns)

    ok = store.ingest(ns, "v1", [0.1, 0.2, 0.3, 0.4], metadata={"doc": "manual"}, ttl=60)
    assert ok is True

    results = store.query(ns, [0.1, 0.2, 0.3, 0.4], top_k=1)
    assert len(results) == 1
    assert results[0][0] == "v1"
    assert results[0][2]["doc"] == "manual"

    batch_items = [
        ("v2", [0.0, 1.0, 0.0, 0.0], {"batch": "true"}),
        ("v3", [0.0, 0.0, 1.0, 0.0], {"batch": "true"}),
    ]
    stored = store.ingest_batch(ns, batch_items, ttl=60)
    assert stored == 2

    results = store.query(ns, [0.0, 1.0, 0.0, 0.0], top_k=2)
    assert len(results) >= 1
    assert results[0][0] == "v2"

    dropped = store.drop_namespace(ns)
    assert dropped is True

    results = store.query(ns, [0.1, 0.2, 0.3, 0.4], top_k=1)
    assert len(results) == 0
    print("✅ NearbyVectorStore socket test PASSED!")

def test_store_content_preview_opt_in():
    """Verify that raw_content metadata field is omitted by default and present when opted-in."""
    cfg_default = NearbyVectorMemoryConfig(
        host="localhost",
        port=6379,
        namespace="test_preview_off",
        store_content_preview=False,
    )
    mem_default = NearbyVectorMemory(config=cfg_default)
    
    batch_items_off = [("v_off", [0.1, 0.2, 0.3, 0.4], {"mime_type": "text/plain"})]
    mem_default._store.ingest_batch("test_preview_off", batch_items_off)
    res_off = mem_default._store.query("test_preview_off", [0.1, 0.2, 0.3, 0.4], top_k=1)
    assert "raw_content" not in res_off[0][2]
    mem_default._store.drop_namespace("test_preview_off")

    cfg_on = NearbyVectorMemoryConfig(
        host="localhost",
        port=6379,
        namespace="test_preview_on",
        store_content_preview=True,
    )
    mem_on = NearbyVectorMemory(config=cfg_on)
    batch_items_on = [("v_on", [0.1, 0.2, 0.3, 0.4], {"mime_type": "text/plain", "raw_content": "Sample preview text"})]
    mem_on._store.ingest_batch("test_preview_on", batch_items_on)
    res_on = mem_on._store.query("test_preview_on", [0.1, 0.2, 0.3, 0.4], top_k=1)
    assert res_on[0][2].get("raw_content") == "Sample preview text"
    mem_on._store.drop_namespace("test_preview_on")

    print("✅ Opt-in store_content_preview test PASSED!")

def test_authenticated_server_handshake():
    """Verify AUTH handshake against authenticated Nearby server on port 6388."""
    # 1. Connect without password -> expect auth failure or error
    store_no_pass = NearbyVectorStore(host="localhost", port=6388)
    try:
        store_no_pass.ingest("auth_ns", "v1", [0.1, 0.2, 0.3, 0.4])
        assert False, "Should fail unauthenticated request"
    except (NearbyAuthError, NearbyProtocolError) as e:
        assert "NOAUTH" in str(e) or "Authentication" in str(e) or "invalid" in str(e)

    # 2. Connect with WRONG password -> expect NearbyAuthError
    store_wrong_pass = NearbyVectorStore(host="localhost", port=6388, password="wrongpassword")
    try:
        store_wrong_pass.ingest("auth_ns", "v1", [0.1, 0.2, 0.3, 0.4])
        assert False, "Should fail invalid password"
    except NearbyAuthError as e:
        assert "Authentication failed" in str(e) or "invalid password" in str(e)

    # 3. Connect with CORRECT password -> expect successful auth and operation
    store_correct_pass = NearbyVectorStore(host="localhost", port=6388, password="mysecretpass")
    ok = store_correct_pass.ingest("auth_ns", "v1", [0.1, 0.2, 0.3, 0.4], metadata={"auth": "ok"})
    assert ok is True

    res = store_correct_pass.query("auth_ns", [0.1, 0.2, 0.3, 0.4], top_k=1)
    assert len(res) == 1
    assert res[0][0] == "v1"
    assert res[0][2]["auth"] == "ok"
    store_correct_pass.drop_namespace("auth_ns")

    print("✅ Authenticated server AUTH handshake test (Correct / Wrong / No Pass) PASSED!")

if __name__ == "__main__":
    test_format_token_injection_guard()
    test_store_content_preview_opt_in()
    test_nearby_vector_store_basic()
    test_authenticated_server_handshake()
