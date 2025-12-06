"""
Validation Debug Database for BrandonBot

SQLite-based telemetry for debugging validation runs.
Keeps the validation CSV clean as a user-facing scorecard while storing:
- OV rejection events
- Tool calls and results
- PQ (Prequalifier) decisions
- Raw LLM responses and internal reasoning

The CSV should only contain clean user-facing dialog.
All internal agent/orchestrator communication goes here.
"""

import sqlite3
import json
import logging
import re
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


@dataclass
class ToolCallRecord:
    """A tool call event."""
    timestamp: str
    test_id: str
    session_id: str
    tool_name: str
    arguments: str  # JSON
    result: str
    success: bool
    duration_ms: int


@dataclass
class PQDecisionRecord:
    """A Prequalifier decision event."""
    timestamp: str
    test_id: str
    session_id: str
    query: str
    frustration_decision: str
    vagueness_decision: str
    pattern_flags: str  # JSON
    explanation: str


@dataclass
class RawLLMRecord:
    """A raw LLM response (before sanitization)."""
    timestamp: str
    test_id: str
    session_id: str
    query: str
    raw_response: str
    sanitized_response: str
    model: str
    tokens_used: int


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
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    test_id TEXT,
                    session_id TEXT,
                    tool_name TEXT NOT NULL,
                    arguments TEXT,
                    result TEXT,
                    success BOOLEAN NOT NULL,
                    duration_ms INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_calls_test_id 
                ON tool_calls(test_id)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pq_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    test_id TEXT,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    frustration_decision TEXT NOT NULL,
                    vagueness_decision TEXT NOT NULL,
                    pattern_flags TEXT,
                    explanation TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pq_decisions_test_id 
                ON pq_decisions(test_id)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_llm_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    test_id TEXT,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    sanitized_response TEXT,
                    model TEXT,
                    tokens_used INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_llm_test_id 
                ON raw_llm_responses(test_id)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_reasoning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    request_id TEXT,
                    reasoning TEXT NOT NULL,
                    parse_method TEXT NOT NULL,
                    raw_response TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_reasoning_session 
                ON llm_reasoning(session_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_reasoning_request 
                ON llm_reasoning(request_id)
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
    
    def log_reasoning(
        self,
        session_id: Optional[str],
        request_id: Optional[str],
        reasoning: str,
        parse_method: str,
        raw_response: Optional[str] = None
    ):
        """
        Log LLM reasoning that was extracted from structured output.
        
        This separates internal chain-of-thought from user-facing responses,
        keeping debug info in SQLite while CSV only shows clean output.
        
        Args:
            session_id: The session ID
            request_id: The request ID
            reasoning: The extracted reasoning/chain-of-thought
            parse_method: How the response was parsed (json, delimiter, chatter_stripped)
            raw_response: Optional full raw LLM response
        """
        timestamp = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO llm_reasoning 
                (timestamp, session_id, request_id, reasoning, parse_method, raw_response)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                timestamp, session_id, request_id, reasoning, parse_method, raw_response
            ))
            conn.commit()
        
        logger.debug(f"Logged LLM reasoning ({parse_method}): {reasoning[:100]}...")
    
    def get_reasoning_by_request(self, request_id: str) -> List[Dict[str, Any]]:
        """Get all reasoning logs for a specific request ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM llm_reasoning 
                WHERE request_id = ? 
                ORDER BY timestamp ASC
            """, (request_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
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
            conn.execute("DELETE FROM tool_calls")
            conn.execute("DELETE FROM pq_decisions")
            conn.execute("DELETE FROM raw_llm_responses")
            conn.commit()
        logger.info("Cleared all debug records")
    
    def log_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        success: bool,
        duration_ms: int = 0,
        test_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log a tool call event."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        arguments_json = json.dumps(arguments) if arguments else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO tool_calls 
                (timestamp, test_id, session_id, tool_name, arguments, result, success, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, test_id, session_id, tool_name, arguments_json, result, success, duration_ms))
            conn.commit()
    
    def log_pq_decision(
        self,
        query: str,
        frustration_decision: str,
        vagueness_decision: str,
        pattern_flags: Optional[Dict[str, Any]] = None,
        explanation: str = "",
        test_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log a Prequalifier decision."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        flags_json = json.dumps(pattern_flags) if pattern_flags else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pq_decisions 
                (timestamp, test_id, session_id, query, frustration_decision, vagueness_decision, pattern_flags, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, test_id, session_id, query, frustration_decision, vagueness_decision, flags_json, explanation))
            conn.commit()
    
    def log_raw_llm_response(
        self,
        query: str,
        raw_response: str,
        sanitized_response: str,
        model: str = "",
        tokens_used: int = 0,
        test_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log a raw LLM response before sanitization."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO raw_llm_responses 
                (timestamp, test_id, session_id, query, raw_response, sanitized_response, model, tokens_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, test_id, session_id, query, raw_response, sanitized_response, model, tokens_used))
            conn.commit()


# Global singleton instance
_debug_db: Optional[ValidationDebugDB] = None


def get_debug_db() -> ValidationDebugDB:
    """Get the global ValidationDebugDB instance."""
    global _debug_db
    if _debug_db is None:
        _debug_db = ValidationDebugDB()
    return _debug_db


def sanitize_bot_response(raw_response: str) -> str:
    """
    Sanitize bot response by removing internal LLM reasoning and tool mentions.
    
    The CSV should only contain clean, user-facing dialog.
    Internal reasoning patterns to remove:
    - "I apologize for the error in my previous response"
    - "I should have used the search_brandon_positions tool"
    - "Let me correct that..."
    - "To verify my answer, I would use..."
    - Self-corrections and meta-commentary
    - Tool call references
    - Code blocks with function calls
    
    Returns:
        Clean, user-facing response text
    """
    if not raw_response:
        return ""
    
    text = raw_response.strip()
    
    patterns_to_remove = [
        r"I apologize for (?:the )?(?:error|technical issue|mistake)[^.]*\.",
        r"I should have (?:used|stated|said)[^.]*\.",
        r"Let me (?:correct|fix|try)[^.]*\.",
        r"To (?:verify|correct|check)[^.]*(?:I would|I'll)[^.]*\.",
        r"You're right,? I should have[^.]*\.",
        r"Let's start fresh\.",
        r"Since I'm a (?:large )?language model[^.]*\.",
        r"The correct response would be[^.]*\.",
        r"If I were to respond[^.]*\.",
        r"```[^`]*```",
        r"I'll make sure to use the \w+ (?:tool|function)[^.]*\.",
        r"(?:search_brandon_positions|search_party_platform|web_search|rag_search|register_volunteer|request_callback)\([^)]*\)",
        r"Unfortunately, I was unable to retrieve[^.]*\.",
        r"It seems that I didn't provide[^.]*\.",
        r"Thought:.*?(?=\n(?:Action:|Observation:|$))",
        r"Action:.*?(?=\n(?:Thought:|Observation:|$))",
        r"Observation:.*?(?=\n(?:Thought:|Action:|$))",
        r"\{\"tool\":\s*\"[^\"]+\",\s*\"args\":.*?\}",
        r"\{\"function\":\s*\"[^\"]+\",\s*\"arguments\":.*?\}",
        r"<tool_call>.*?</tool_call>",
        r"<analysis>.*?</analysis>",
        r"<internal>.*?</internal>",
        r"<reasoning>.*?</reasoning>",
        r"\[Internal\].*?(?=\n\n|\Z)",
        r"\[TOOL RESULT\].*?(?=\n\n|\Z)",
        r"I need to use the \w+ (?:tool|function)[^.]*\.",
        r"Based on the tool results?,?\s*",
        r"The \w+ tool (?:returned|shows|indicates)[^.]*\.",
        
        r"You're right,? I should verify[^.]*\. Here's the corrected response:?",
        r"Here's (?:the|my) corrected response:?",
        r"Here's (?:the|a) (?:more )?(?:accurate|better|proper) response:?",
        r"I'll make sure to verify[^.]*\.",
        r"I'll proceed to[^.]*\.",
        r"^#+\s*Step \d+:?[^\n]*\n?",
        r"\n#+\s*Step \d+:?[^\n]*",
        r"No (?:search|tool|verification) is needed[^.]*[.,]",
        r"(?:The|My) response is:?",
        r"I'll (?:provide|give) (?:a|the) (?:friendly|helpful)[^.]*\.",
        r"(?:So )?I'll proceed[^.]*\.",
        r"Let me (?:provide|give)[^.]*\.",
        r"I(?:'ll| will) verify[^.]*\.",
        r"After (?:searching|checking|verifying)[^,.:]*,?\s*",
        r"According to (?:my search|the search|my verification)[^,.:]*,?\s*",
        r"^\"",
        r"\"$",
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    
    if len(text) < 20 or text.lower().startswith("i'm not aware") or "don't have enough information" in text.lower():
        clean_sentences = []
        sentences = raw_response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            skip = False
            skip_keywords = [
                "should have", "apologize", "error in my", "let me correct",
                "to verify", "search_", "function", "tool", "language model",
                "correct response would be", "here's the corrected", "step 1:",
                "step 2:", "step 3:", "i'll proceed", "no search is needed",
                "i'll verify", "i'll make sure", "the response is:"
            ]
            for kw in skip_keywords:
                if kw.lower() in sentence.lower():
                    skip = True
                    break
            if not skip:
                clean_sentences.append(sentence)
        
        if clean_sentences:
            text = '. '.join(clean_sentences)
            if not text.endswith(('.', '?', '!')):
                text += '.'
    
    return text
