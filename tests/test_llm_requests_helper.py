import os
import sys
import json
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.validation_debug import get_debug_db


def test_get_llm_requests_by_test_roundtrip(tmp_path):
    vdb = get_debug_db()
    test_id = "TEST-LLM-001"
    session_id = "sess-1"

    messages = [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "Hello"}]
    tools = [{"name": "search", "args": {"q": "Brandon"}}]

    # Log a single LLM request
    vdb.log_llm_request(
        system_prompt="System: diagnostics",
        messages=messages,
        tools=tools,
        provider="test-provider",
        model="test-model",
        test_id=test_id,
        session_id=session_id,
        request_id="req-1",
    )

    rows = vdb.get_llm_requests_by_test(test_id)
    assert len(rows) >= 1
    found = False
    for r in rows:
        if r["session_id"] == session_id and r["provider"] == "test-provider":
            assert isinstance(r["messages"], list)
            assert r["messages"][1]["content"] == "Hello"
            assert isinstance(r["tools"], list)
            found = True
            break
    assert found
