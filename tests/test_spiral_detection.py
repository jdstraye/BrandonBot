import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

import sqlite3
from backend.agent_orchestrator import detect_death_spiral
from backend.validation_debug import get_debug_db


def test_detect_death_spiral_by_intent_streak():
    metadata = {"validation_rejections": []}
    # Simulate three consecutive intent-checking rejections
    for i in range(3):
        metadata["validation_rejections"].append({
            "attempt": i + 1,
            "failed_checks": [{"safeguard": "intent_checking", "score": 4, "explanation": "mostly unrelated"}],
        })

    debug_db = get_debug_db()
    detected, reason, details = detect_death_spiral(metadata, test_id=None, session_id=None, debug_db=debug_db)
    assert detected is True
    assert "intent_rejection_streak" in reason


def test_detect_death_spiral_by_repetition():
    test_id = "SPIRAL-REP-000"
    debug_db = get_debug_db()
    # Ensure some recent identical sanitized responses are present
    for i in range(4):
        debug_db.log_ov_attempt(
            attempt_num=i,
            ov_results={"intent_checking": {"score": 4}},
            final_status="rejected",
            original_response="I apologize, but I'm having trouble completing this request.",
            sanitized_response="I apologize, but I'm having trouble completing this request.",
            aggregate_score=4,
            test_id=test_id,
            session_id=f"session-{i}",
            request_id=None,
        )

    metadata = {"validation_rejections": []}
    detected, reason, details = detect_death_spiral(metadata, test_id=test_id, session_id=None, debug_db=debug_db)
    assert detected is True
    assert "repetition_detected" in reason or "repetition_detected_unique_count" in reason


def test_log_spiral_event_persists():
    debug_db = get_debug_db()
    test_id = "SPIRAL-LOG-000"
    recent = [{"attempt": 1, "reason": "intent"}, {"attempt": 2, "reason": "intent"}]
    debug_db.log_spiral_event(test_id=test_id, session_id="s1", request_id="r1", reason="test_spiral", recent_rejections=recent)

    # Verify via direct DB query
    conn = __import__('sqlite3').connect(debug_db.db_path)
    cur = conn.cursor()
    cur.execute('SELECT test_id, reason, recent_rejections FROM spiral_events WHERE test_id = ? ORDER BY id DESC LIMIT 1', (test_id,))
    row = cur.fetchone()
    conn.close()
    assert row is not None
    assert row[0] == test_id
    assert 'test_spiral' in row[1]
    assert 'attempt' in row[2]
