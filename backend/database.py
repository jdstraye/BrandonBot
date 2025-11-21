import aiosqlite
import logging
import os
from datetime import datetime
import json
from typing import Optional

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
                    consent_given BOOLEAN
                )
            ''')
            
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
            
            await db.commit()
            logger.info("Database initialized successfully")
    
    async def update_consent(self, user_id: str, consent_given: bool):
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.utcnow().isoformat()
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
                             confidence: float, sources: list, consent_given: bool):
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.utcnow().isoformat()
            await db.execute('''
                INSERT INTO interactions (user_id, query, response, confidence, sources, timestamp, consent_given)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, query, response, confidence, json.dumps(sources), now, consent_given))
            await db.commit()
            
            await self._track_new_question(query)
    
    async def _track_new_question(self, question: str):
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.utcnow().isoformat()
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
            now = datetime.utcnow().isoformat()
            await db.execute('''
                INSERT INTO callback_requests (user_id, name, phone, email, question, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, name, phone, email, question, now))
            await db.commit()
    
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
            
            return {
                "total_interactions": total_interactions,
                "pending_callbacks": pending_callbacks,
                "top_questions": [{"question": q[0], "count": q[1]} for q in top_questions]
            }
    
    async def close(self):
        pass
