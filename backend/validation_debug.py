"""
Validation Debug Database for BrandonBot

Simple SQLite-based telemetry for debugging Output Validator rejections.
Keeps the validation CSV clean as a scorecard while storing detailed
rejection info for investigation.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "validation" / "debug.db"


@dataclass
class OVRejectionRecord:
    """A single OV rejection event."""
    timestamp: str
    test_id: str
    session_id: str
    request_id: str
    query: str
    original_response: str
    safeguard: str
    score: int
    explanation: str
    all_results: str  # JSON of all safeguard results


class ValidationDebugDB:
    """
    SQLite database for storing OV rejection telemetry.
    
    Usage:
        debug_db = ValidationDebugDB()
        debug_db.log_ov_rejection(
            test_id="A_VAGUE-002",
            session_id="sess_123",
            request_id="req_456",
            query="Hi Brandon, How are you today?",
            original_response="I think you should...",
            safeguard="ethics_morality",
            score=5,
            explanation="Response contained problematic content",
            all_results={"intent_checking": {"score": 2, ...}, ...}
        )
        
        # Query rejections
        rejections = debug_db.get_rejections_by_test("A_VAGUE-002")
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ov_rejections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    test_id TEXT,
                    session_id TEXT,
                    request_id TEXT,
                    query TEXT NOT NULL,
                    original_response TEXT NOT NULL,
                    safeguard TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    explanation TEXT,
                    all_results TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ov_rejections_test_id 
                ON ov_rejections(test_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ov_rejections_safeguard 
                ON ov_rejections(safeguard)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ov_rejections_score 
                ON ov_rejections(score)
            """)
            
            conn.commit()
        
        logger.info(f"ValidationDebugDB initialized at {self.db_path}")
    
    def log_ov_rejection(
        self,
        query: str,
        original_response: str,
        safeguard: str,
        score: int,
        explanation: str,
        test_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        all_results: Optional[Dict[str, Any]] = None
    ):
        """Log an OV rejection event."""
        timestamp = datetime.utcnow().isoformat()
        all_results_json = json.dumps(all_results) if all_results else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO ov_rejections 
                (timestamp, test_id, session_id, request_id, query, 
                 original_response, safeguard, score, explanation, all_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, test_id, session_id, request_id, query,
                original_response, safeguard, score, explanation, all_results_json
            ))
            conn.commit()
        
        logger.debug(f"Logged OV rejection: {safeguard}={score} for query '{query[:50]}...'")
    
    def log_all_ov_failures(
        self,
        query: str,
        original_response: str,
        validation_result: Any,  # OVValidationResult
        test_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        Log all failing safeguards from an OVValidationResult.
        
        Args:
            validation_result: The OVValidationResult object with .results dict
        """
        all_results = {}
        for safeguard, result in validation_result.results.items():
            all_results[safeguard.value] = {
                "score": result.score,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "method": result.method
            }
        
        for safeguard, result in validation_result.results.items():
            if result.score >= 4:  # Log score 4 and 5 (hard fails)
                self.log_ov_rejection(
                    query=query,
                    original_response=original_response,
                    safeguard=safeguard.value,
                    score=result.score,
                    explanation=result.explanation,
                    test_id=test_id,
                    session_id=session_id,
                    request_id=request_id,
                    all_results=all_results
                )
    
    def get_rejections_by_test(self, test_id: str) -> List[OVRejectionRecord]:
        """Get all rejections for a specific test ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM ov_rejections 
                WHERE test_id = ? 
                ORDER BY timestamp DESC
            """, (test_id,))
            
            return [OVRejectionRecord(
                timestamp=row["timestamp"],
                test_id=row["test_id"],
                session_id=row["session_id"],
                request_id=row["request_id"],
                query=row["query"],
                original_response=row["original_response"],
                safeguard=row["safeguard"],
                score=row["score"],
                explanation=row["explanation"],
                all_results=row["all_results"]
            ) for row in cursor.fetchall()]
    
    def get_rejections_by_safeguard(self, safeguard: str) -> List[OVRejectionRecord]:
        """Get all rejections for a specific safeguard."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM ov_rejections 
                WHERE safeguard = ? 
                ORDER BY timestamp DESC
            """, (safeguard,))
            
            return [OVRejectionRecord(
                timestamp=row["timestamp"],
                test_id=row["test_id"],
                session_id=row["session_id"],
                request_id=row["request_id"],
                query=row["query"],
                original_response=row["original_response"],
                safeguard=row["safeguard"],
                score=row["score"],
                explanation=row["explanation"],
                all_results=row["all_results"]
            ) for row in cursor.fetchall()]
    
    def get_recent_rejections(self, limit: int = 20) -> List[OVRejectionRecord]:
        """Get the most recent rejections."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM ov_rejections 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            return [OVRejectionRecord(
                timestamp=row["timestamp"],
                test_id=row["test_id"],
                session_id=row["session_id"],
                request_id=row["request_id"],
                query=row["query"],
                original_response=row["original_response"],
                safeguard=row["safeguard"],
                score=row["score"],
                explanation=row["explanation"],
                all_results=row["all_results"]
            ) for row in cursor.fetchall()]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of rejections."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM ov_rejections").fetchone()[0]
            
            by_safeguard = {}
            cursor = conn.execute("""
                SELECT safeguard, COUNT(*) as count, AVG(score) as avg_score
                FROM ov_rejections 
                GROUP BY safeguard
                ORDER BY count DESC
            """)
            for row in cursor.fetchall():
                by_safeguard[row[0]] = {"count": row[1], "avg_score": round(row[2], 2)}
            
            by_score = {}
            cursor = conn.execute("""
                SELECT score, COUNT(*) as count
                FROM ov_rejections 
                GROUP BY score
                ORDER BY score
            """)
            for row in cursor.fetchall():
                by_score[row[0]] = row[1]
            
            return {
                "total_rejections": total,
                "by_safeguard": by_safeguard,
                "by_score": by_score
            }
    
    def clear_all(self):
        """Clear all rejection records (for testing)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM ov_rejections")
            conn.commit()
        logger.info("Cleared all OV rejection records")


# Global singleton instance
_debug_db: Optional[ValidationDebugDB] = None


def get_debug_db() -> ValidationDebugDB:
    """Get the global ValidationDebugDB instance."""
    global _debug_db
    if _debug_db is None:
        _debug_db = ValidationDebugDB()
    return _debug_db
