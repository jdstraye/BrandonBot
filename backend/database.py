import aiosqlite
import logging
import os
from datetime import datetime
import json
from typing import Optional, List

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS user_consent (
                    user_id TEXT PRIMARY KEY,
                    consent_given BOOLEAN,
                    consent_date TEXT,
                    updated_at TEXT
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    response TEXT,
                    confidence REAL,
                    sources TEXT,
                    timestamp TEXT,
                    consent_given BOOLEAN,
                    model_used TEXT
                )
            ''')
            
            try:
                async with db.execute("PRAGMA table_info(interactions)") as cursor:
                    columns = [row[1] for row in await cursor.fetchall()]
                if 'model_used' not in columns:
                    await db.execute('ALTER TABLE interactions ADD COLUMN model_used TEXT')
            except Exception:
                pass
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS callback_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    name TEXT,
                    phone TEXT,
                    email TEXT,
                    question TEXT,
                    timestamp TEXT,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS new_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT UNIQUE,
                    count INTEGER DEFAULT 1,
                    first_asked TEXT,
                    last_asked TEXT
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT,
                    model_used TEXT,
                    timestamp TEXT NOT NULL,
                    consent_given BOOLEAN DEFAULT FALSE
                )
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversation_session 
                ON conversation_history(session_id)
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    session_id TEXT,
                    user_id TEXT,
                    query TEXT,
                    response TEXT,
                    model_used TEXT,
                    tool_calls TEXT,
                    total_tokens INTEGER,
                    duration_ms INTEGER,
                    timestamp TEXT NOT NULL,
                    error TEXT
                )
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_request_logs_session 
                ON request_logs(session_id)
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    session_id TEXT,
                    request_id TEXT,
                    success BOOLEAN NOT NULL,
                    latency_ms INTEGER,
                    tokens_used INTEGER,
                    error TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_performance_provider 
                ON model_performance(provider, model)
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS volunteers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT,
                    zip_code TEXT,
                    interests TEXT,
                    availability TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT DEFAULT 'new',
                    email_sent BOOLEAN DEFAULT FALSE
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS donation_interests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT,
                    message TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT DEFAULT 'new',
                    email_sent BOOLEAN DEFAULT FALSE
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS compliance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    severity TEXT DEFAULT 'info',
                    timestamp TEXT NOT NULL
                )
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_compliance_event_type 
                ON compliance_log(event_type)
            ''')
            
            await db.commit()
            logger.info("Database initialized successfully")
    
    async def update_consent(self, user_id: str, consent_given: bool):
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            await db.execute('''
                INSERT INTO user_consent (user_id, consent_given, consent_date, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    consent_given = excluded.consent_given,
                    updated_at = excluded.updated_at
            ''', (user_id, consent_given, now, now))
            await db.commit()
    
    async def get_consent(self, user_id: str) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                'SELECT consent_given FROM user_consent WHERE user_id = ?',
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else False
    
    async def log_interaction(self, user_id: str, query: str, response: str, 
                             confidence: float, sources: list, consent_given: bool,
                             model_used: Optional[str] = None):
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            await db.execute('''
                INSERT INTO interactions (user_id, query, response, confidence, sources, timestamp, consent_given, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, query, response, confidence, json.dumps(sources), now, consent_given, model_used))
            await db.commit()
            
            await self._track_new_question(query)
    
    async def _track_new_question(self, question: str):
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            await db.execute('''
                INSERT INTO new_questions (question, count, first_asked, last_asked)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(question) DO UPDATE SET
                    count = count + 1,
                    last_asked = excluded.last_asked
            ''', (question, now, now))
            await db.commit()
    
    async def log_callback_request(self, user_id: str, name: str, phone: str, 
                                   email: Optional[str], question: str):
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            await db.execute('''
                INSERT INTO callback_requests (user_id, name, phone, email, question, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, name, phone, email, question, now))
            await db.commit()
    
    async def log_conversation_turn(self, session_id: str, role: str, content: str,
                                     user_id: Optional[str] = None, tool_calls: Optional[list] = None,
                                     model_used: Optional[str] = None, consent_given: bool = False):
        """Log a single conversation turn with consent flag"""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            await db.execute('''
                INSERT INTO conversation_history 
                (session_id, user_id, role, content, tool_calls, model_used, timestamp, consent_given)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, user_id, role, content, 
                  json.dumps(tool_calls) if tool_calls else None,
                  model_used, now, consent_given))
            await db.commit()
    
    async def get_conversation_history(self, session_id: str, limit: int = 20) -> list:
        """Retrieve conversation history for a session"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('''
                SELECT role, content, tool_calls, model_used, timestamp
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit)) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "role": row[0],
                        "content": row[1],
                        "tool_calls": json.loads(row[2]) if row[2] else None,
                        "model_used": row[3],
                        "timestamp": row[4]
                    }
                    for row in reversed(rows)
                ]
    
    async def log_request(self, request_id: str, session_id: str, query: str,
                          response: str, model_used: Optional[str] = None,
                          tool_calls: Optional[list] = None, total_tokens: int = 0,
                          duration_ms: int = 0, user_id: Optional[str] = None,
                          error: Optional[str] = None):
        """Log a complete request with timing and tool traces"""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            await db.execute('''
                INSERT INTO request_logs
                (request_id, session_id, user_id, query, response, model_used, 
                 tool_calls, total_tokens, duration_ms, timestamp, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (request_id, session_id, user_id, query, response, model_used,
                  json.dumps(tool_calls) if tool_calls else None,
                  total_tokens, duration_ms, now, error))
            await db.commit()
    
    async def log_model_performance(self, provider: str, model: str, 
                                     success: bool, latency_ms: int = 0,
                                     tokens_used: int = 0, error: Optional[str] = None,
                                     session_id: Optional[str] = None,
                                     request_id: Optional[str] = None):
        """Log model performance for evaluation"""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            await db.execute('''
                INSERT INTO model_performance
                (provider, model, session_id, request_id, success, latency_ms, 
                 tokens_used, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (provider, model, session_id, request_id, success, latency_ms,
                  tokens_used, error, now))
            await db.commit()
    
    async def get_model_stats(self) -> dict:
        """Get aggregated model performance statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('''
                SELECT provider, model, 
                       COUNT(*) as total_calls,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_calls,
                       AVG(latency_ms) as avg_latency,
                       SUM(tokens_used) as total_tokens
                FROM model_performance
                GROUP BY provider, model
                ORDER BY total_calls DESC
            ''') as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "provider": row[0],
                        "model": row[1],
                        "total_calls": row[2],
                        "successful_calls": row[3],
                        "success_rate": row[3] / row[2] if row[2] > 0 else 0,
                        "avg_latency_ms": round(row[4] or 0, 2),
                        "total_tokens": row[5] or 0
                    }
                    for row in rows
                ]
    
    async def get_request_logs(self, session_id: Optional[str] = None, 
                                limit: int = 100) -> list:
        """Retrieve request logs, optionally filtered by session"""
        async with aiosqlite.connect(self.db_path) as db:
            if session_id:
                query = '''
                    SELECT request_id, session_id, query, response, model_used, 
                           tool_calls, total_tokens, duration_ms, timestamp, error
                    FROM request_logs WHERE session_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                '''
                params = (session_id, limit)
            else:
                query = '''
                    SELECT request_id, session_id, query, response, model_used, 
                           tool_calls, total_tokens, duration_ms, timestamp, error
                    FROM request_logs
                    ORDER BY timestamp DESC LIMIT ?
                '''
                params = (limit,)
            
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "request_id": row[0],
                        "session_id": row[1],
                        "query": row[2],
                        "response": row[3],
                        "model_used": row[4],
                        "tool_calls": json.loads(row[5]) if row[5] else None,
                        "total_tokens": row[6],
                        "duration_ms": row[7],
                        "timestamp": row[8],
                        "error": row[9]
                    }
                    for row in rows
                ]
    
    async def get_stats(self):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('SELECT COUNT(*) FROM interactions') as cursor:
                total_interactions = (await cursor.fetchone())[0]
            
            async with db.execute('SELECT COUNT(*) FROM callback_requests WHERE status = "pending"') as cursor:
                pending_callbacks = (await cursor.fetchone())[0]
            
            async with db.execute(
                'SELECT question, count FROM new_questions ORDER BY count DESC LIMIT 10'
            ) as cursor:
                top_questions = await cursor.fetchall()
            
            async with db.execute('SELECT COUNT(*) FROM request_logs') as cursor:
                total_requests = (await cursor.fetchone())[0]
            
            async with db.execute('SELECT AVG(duration_ms) FROM request_logs WHERE duration_ms > 0') as cursor:
                avg_duration = (await cursor.fetchone())[0] or 0
            
            return {
                "total_interactions": total_interactions,
                "pending_callbacks": pending_callbacks,
                "top_questions": [{"question": q[0], "count": q[1]} for q in top_questions],
                "total_requests": total_requests,
                "avg_response_time_ms": round(avg_duration, 2)
            }
    
    async def log_volunteer(self, name: str, email: str, phone: str = "",
                            zip_code: str = "", interests: List[str] = None,
                            availability: str = "flexible"):
        """Log volunteer registration"""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            await db.execute('''
                INSERT INTO volunteers 
                (name, email, phone, zip_code, interests, availability, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, email, phone, zip_code, 
                  json.dumps(interests) if interests else "[]",
                  availability, now))
            await db.commit()
            logger.info(f"Volunteer logged: {name} ({email})")
    
    async def log_donation_interest(self, name: str, email: str, 
                                     phone: str = "", message: str = ""):
        """Log donation interest (NOT actual donation - for FEC compliance)"""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.utcnow().isoformat()
            await db.execute('''
                INSERT INTO donation_interests 
                (name, email, phone, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, email, phone, message, now))
            await db.commit()
            logger.info(f"Donation interest logged: {name} ({email})")
    
    async def log_compliance_event(self, event_type: str, event_data: dict = None,
                                    session_id: str = None, severity: str = "info"):
        """Log compliance-related events for audit trail"""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.utcnow().isoformat()
            await db.execute('''
                INSERT INTO compliance_log 
                (session_id, event_type, event_data, severity, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, event_type, 
                  json.dumps(event_data) if event_data else None,
                  severity, now))
            await db.commit()
    
    async def get_volunteers(self, status: str = None, limit: int = 100) -> list:
        """Get volunteer registrations"""
        async with aiosqlite.connect(self.db_path) as db:
            if status:
                query = '''
                    SELECT id, name, email, phone, zip_code, interests, 
                           availability, timestamp, status
                    FROM volunteers WHERE status = ?
                    ORDER BY timestamp DESC LIMIT ?
                '''
                params = (status, limit)
            else:
                query = '''
                    SELECT id, name, email, phone, zip_code, interests, 
                           availability, timestamp, status
                    FROM volunteers
                    ORDER BY timestamp DESC LIMIT ?
                '''
                params = (limit,)
            
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "phone": row[3],
                        "zip_code": row[4],
                        "interests": json.loads(row[5]) if row[5] else [],
                        "availability": row[6],
                        "timestamp": row[7],
                        "status": row[8]
                    }
                    for row in rows
                ]
    
    async def get_donation_interests(self, status: str = None, limit: int = 100) -> list:
        """Get donation interest registrations"""
        async with aiosqlite.connect(self.db_path) as db:
            if status:
                query = '''
                    SELECT id, name, email, phone, message, timestamp, status
                    FROM donation_interests WHERE status = ?
                    ORDER BY timestamp DESC LIMIT ?
                '''
                params = (status, limit)
            else:
                query = '''
                    SELECT id, name, email, phone, message, timestamp, status
                    FROM donation_interests
                    ORDER BY timestamp DESC LIMIT ?
                '''
                params = (limit,)
            
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "phone": row[3],
                        "message": row[4],
                        "timestamp": row[5],
                        "status": row[6]
                    }
                    for row in rows
                ]
    
    async def close(self):
        pass
