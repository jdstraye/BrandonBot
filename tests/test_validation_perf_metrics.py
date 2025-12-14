import sqlite3
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.validation_debug import ValidationDebugDB


def test_log_perf_metric_respects_config(monkeypatch, tmp_path):
    # Create a temp DB path to avoid touching repo DB
    db_path = tmp_path / "debug.db"
    vdb = ValidationDebugDB(db_path=db_path)

    # Monkeypatch load_config to enable perf monitor
    class FakeCfg:
        validation = {"perf_monitor": True}

    monkeypatch.setattr('backend.config_loader.load_config', lambda: FakeCfg())

    # Call log_perf_metric (should write a row)
    vdb.log_perf_metric(step="test_step", duration_ms=123, test_id="T-001")

    # Verify perf_metrics has an entry
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM perf_metrics WHERE test_id = ?", ("T-001",))
    count = cur.fetchone()[0]
    conn.close()
    assert count >= 1


def test_log_perf_metric_disabled_by_config(monkeypatch, tmp_path):
    db_path = tmp_path / "debug.db"
    vdb = ValidationDebugDB(db_path=db_path)

    class FakeCfg:
        validation = {"perf_monitor": False}

    monkeypatch.setattr('backend.config_loader.load_config', lambda: FakeCfg())

    vdb.log_perf_metric(step="test_step", duration_ms=10, test_id="T-002")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM perf_metrics WHERE test_id = ?", ("T-002",))
    count = cur.fetchone()[0]
    conn.close()
    assert count == 0
