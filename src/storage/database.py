"""SQLite database connection and schema management."""

import sqlite3
import json
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class Database:
    """SQLite database manager with connection pooling."""

    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    context_tags TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    preference TEXT NOT NULL,
                    strength REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.5,
                    examples TEXT,
                    exceptions TEXT,
                    source TEXT DEFAULT 'learned',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rules (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    action TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    exceptions TEXT,
                    source TEXT DEFAULT 'explicit',
                    created_at TEXT NOT NULL
                )
            """)

            # Feedback events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    data TEXT,
                    context TEXT,
                    emotional_tone TEXT DEFAULT 'neutral',
                    processed INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT,
                    response TEXT,
                    context TEXT,
                    duration_seconds REAL DEFAULT 0.0,
                    engagement_score REAL DEFAULT 0.5,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Learning updates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_updates (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    priority TEXT DEFAULT 'medium',
                    update_data TEXT,
                    affected_memories TEXT,
                    source TEXT DEFAULT 'implicit',
                    applied INTEGER DEFAULT 0,
                    impact_score REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Personality profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_profiles (
                    user_id TEXT PRIMARY KEY,
                    assertiveness REAL DEFAULT 0.5,
                    warmth REAL DEFAULT 0.5,
                    detail_orientation REAL DEFAULT 0.5,
                    formality REAL DEFAULT 0.5,
                    humor_level REAL DEFAULT 0.3,
                    pace REAL DEFAULT 0.5,
                    directness REAL DEFAULT 0.5,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_preferences_user ON preferences(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback_events(user_id)")

            conn.commit()

    @contextmanager
    def get_connection(self):
        """Get a database connection with context management."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = ()):
        """Execute a query and return results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchall()

    def execute_many(self, query: str, params_list: list):
        """Execute a query with multiple parameter sets."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
