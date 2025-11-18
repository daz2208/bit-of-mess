"""Data access repositories for all entity types."""

import json
import numpy as np
from datetime import datetime
from typing import List, Optional

from .database import Database
from ..models.memory import MemoryEntry, PreferenceNode, Rule
from ..models.feedback import FeedbackEvent, InteractionEvent, FeedbackType, EmotionalTone
from ..models.learning import LearningUpdate, UpdateType
from ..models.action import PersonalityProfile
from ..models.base import Priority, MemoryType, Context


class MemoryRepository:
    """Repository for memory entries."""

    def __init__(self, db: Database):
        self.db = db

    def save(self, memory: MemoryEntry) -> str:
        """Save a memory entry."""
        embedding_blob = memory.embedding.tobytes() if memory.embedding is not None else None

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memories
                (id, user_id, memory_type, content, embedding, importance,
                 access_count, last_accessed, context_tags, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id, memory.user_id, memory.memory_type.value, memory.content,
                embedding_blob, memory.importance, memory.access_count,
                memory.last_accessed.isoformat(), json.dumps(memory.context_tags),
                json.dumps(memory.metadata), memory.created_at.isoformat(),
                datetime.utcnow().isoformat()
            ))
            conn.commit()
        return memory.id

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()

        if row:
            return self._row_to_memory(row)
        return None

    def get_by_user(self, user_id: str, memory_type: Optional[MemoryType] = None,
                    limit: int = 100) -> List[MemoryEntry]:
        """Get memories for a user, optionally filtered by type."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if memory_type:
                cursor.execute("""
                    SELECT * FROM memories
                    WHERE user_id = ? AND memory_type = ?
                    ORDER BY last_accessed DESC LIMIT ?
                """, (user_id, memory_type.value, limit))
            else:
                cursor.execute("""
                    SELECT * FROM memories
                    WHERE user_id = ?
                    ORDER BY last_accessed DESC LIMIT ?
                """, (user_id, limit))

            return [self._row_to_memory(row) for row in cursor.fetchall()]

    def update_access(self, memory_id: str):
        """Update access count and timestamp."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), memory_id))
            conn.commit()

    def delete(self, memory_id: str):
        """Delete a memory."""
        self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def _row_to_memory(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        embedding = None
        if row['embedding']:
            embedding = np.frombuffer(row['embedding'], dtype=np.float32)

        return MemoryEntry(
            id=row['id'],
            user_id=row['user_id'],
            memory_type=MemoryType(row['memory_type']),
            content=row['content'],
            embedding=embedding,
            importance=row['importance'],
            access_count=row['access_count'],
            last_accessed=datetime.fromisoformat(row['last_accessed']),
            context_tags=json.loads(row['context_tags']) if row['context_tags'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )


class PreferenceRepository:
    """Repository for user preferences."""

    def __init__(self, db: Database):
        self.db = db

    def save(self, pref: PreferenceNode) -> str:
        """Save a preference."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO preferences
                (id, user_id, category, preference, strength, confidence,
                 examples, exceptions, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pref.id, pref.user_id, pref.category, pref.preference,
                pref.strength, pref.confidence, json.dumps(pref.examples),
                json.dumps(pref.exceptions), pref.source,
                pref.created_at.isoformat(), datetime.utcnow().isoformat()
            ))
            conn.commit()
        return pref.id

    def get_by_user(self, user_id: str, category: Optional[str] = None) -> List[PreferenceNode]:
        """Get preferences for a user."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute("""
                    SELECT * FROM preferences
                    WHERE user_id = ? AND category = ?
                    ORDER BY strength DESC
                """, (user_id, category))
            else:
                cursor.execute("""
                    SELECT * FROM preferences
                    WHERE user_id = ?
                    ORDER BY strength DESC
                """, (user_id,))

            return [self._row_to_preference(row) for row in cursor.fetchall()]

    def update_strength(self, pref_id: str, new_strength: float):
        """Update preference strength."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE preferences SET strength = ?, updated_at = ?
                WHERE id = ?
            """, (new_strength, datetime.utcnow().isoformat(), pref_id))
            conn.commit()

    def _row_to_preference(self, row) -> PreferenceNode:
        """Convert row to PreferenceNode."""
        return PreferenceNode(
            id=row['id'],
            user_id=row['user_id'],
            category=row['category'],
            preference=row['preference'],
            strength=row['strength'],
            confidence=row['confidence'],
            examples=json.loads(row['examples']) if row['examples'] else [],
            exceptions=json.loads(row['exceptions']) if row['exceptions'] else [],
            source=row['source'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )


class RuleRepository:
    """Repository for explicit rules."""

    def __init__(self, db: Database):
        self.db = db

    def save(self, rule: Rule) -> str:
        """Save a rule."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO rules
                (id, user_id, condition, action, strength, exceptions, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.id, rule.user_id, rule.condition, rule.action,
                rule.strength, json.dumps(rule.exceptions), rule.source,
                rule.created_at.isoformat()
            ))
            conn.commit()
        return rule.id

    def get_by_user(self, user_id: str) -> List[Rule]:
        """Get all rules for a user."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM rules WHERE user_id = ? ORDER BY strength DESC
            """, (user_id,))

            return [self._row_to_rule(row) for row in cursor.fetchall()]

    def _row_to_rule(self, row) -> Rule:
        """Convert row to Rule."""
        return Rule(
            id=row['id'],
            user_id=row['user_id'],
            condition=row['condition'],
            action=row['action'],
            strength=row['strength'],
            exceptions=json.loads(row['exceptions']) if row['exceptions'] else [],
            source=row['source'],
            created_at=datetime.fromisoformat(row['created_at'])
        )


class FeedbackRepository:
    """Repository for feedback events."""

    def __init__(self, db: Database):
        self.db = db

    def save(self, event: FeedbackEvent) -> str:
        """Save a feedback event."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO feedback_events
                (id, user_id, type, data, context, emotional_tone, processed, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id, event.user_id, event.type.value, json.dumps(event.data),
                json.dumps(event.context.to_dict()) if event.context else None,
                event.emotional_tone.value, 1 if event.processed else 0,
                event.created_at.isoformat(), datetime.utcnow().isoformat()
            ))
            conn.commit()
        return event.id

    def get_unprocessed(self, user_id: str) -> List[FeedbackEvent]:
        """Get unprocessed feedback events."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM feedback_events
                WHERE user_id = ? AND processed = 0
                ORDER BY created_at ASC
            """, (user_id,))

            return [self._row_to_feedback(row) for row in cursor.fetchall()]

    def mark_processed(self, event_id: str):
        """Mark event as processed."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE feedback_events SET processed = 1, updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), event_id))
            conn.commit()

    def _row_to_feedback(self, row) -> FeedbackEvent:
        """Convert row to FeedbackEvent."""
        context = None
        if row['context']:
            ctx_data = json.loads(row['context'])
            context = Context(
                user_id=ctx_data.get('user_id', ''),
                timestamp=datetime.fromisoformat(ctx_data['timestamp']) if 'timestamp' in ctx_data else datetime.utcnow(),
                time_of_day=ctx_data.get('time_of_day', ''),
                day_type=ctx_data.get('day_type', ''),
                user_state=ctx_data.get('user_state', {}),
                environment=ctx_data.get('environment', {}),
                metadata=ctx_data.get('metadata', {})
            )

        return FeedbackEvent(
            id=row['id'],
            user_id=row['user_id'],
            type=FeedbackType(row['type']),
            data=json.loads(row['data']) if row['data'] else {},
            context=context,
            emotional_tone=EmotionalTone(row['emotional_tone']),
            processed=bool(row['processed']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )


class InteractionRepository:
    """Repository for interaction events."""

    def __init__(self, db: Database):
        self.db = db

    def save(self, interaction: InteractionEvent) -> str:
        """Save an interaction."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO interactions
                (id, user_id, event_type, content, response, context,
                 duration_seconds, engagement_score, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction.id, interaction.user_id, interaction.event_type,
                interaction.content, interaction.response,
                json.dumps(interaction.context.to_dict()) if interaction.context else None,
                interaction.duration_seconds, interaction.engagement_score,
                json.dumps(interaction.metadata), interaction.created_at.isoformat(),
                datetime.utcnow().isoformat()
            ))
            conn.commit()
        return interaction.id

    def get_recent(self, user_id: str, limit: int = 100) -> List[InteractionEvent]:
        """Get recent interactions."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM interactions
                WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?
            """, (user_id, limit))

            return [self._row_to_interaction(row) for row in cursor.fetchall()]

    def _row_to_interaction(self, row) -> InteractionEvent:
        """Convert row to InteractionEvent."""
        context = None
        if row['context']:
            ctx_data = json.loads(row['context'])
            context = Context(
                user_id=ctx_data.get('user_id', ''),
                timestamp=datetime.fromisoformat(ctx_data['timestamp']) if 'timestamp' in ctx_data else datetime.utcnow(),
                time_of_day=ctx_data.get('time_of_day', ''),
                day_type=ctx_data.get('day_type', ''),
                user_state=ctx_data.get('user_state', {}),
                environment=ctx_data.get('environment', {}),
                metadata=ctx_data.get('metadata', {})
            )

        return InteractionEvent(
            id=row['id'],
            user_id=row['user_id'],
            event_type=row['event_type'],
            content=row['content'] or '',
            response=row['response'] or '',
            context=context,
            duration_seconds=row['duration_seconds'],
            engagement_score=row['engagement_score'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )


class PersonalityRepository:
    """Repository for personality profiles."""

    def __init__(self, db: Database):
        self.db = db

    def save(self, profile: PersonalityProfile):
        """Save or update personality profile."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO personality_profiles
                (user_id, assertiveness, warmth, detail_orientation, formality,
                 humor_level, pace, directness, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id, profile.assertiveness, profile.warmth,
                profile.detail_orientation, profile.formality, profile.humor_level,
                profile.pace, profile.directness, datetime.utcnow().isoformat()
            ))
            conn.commit()

    def get(self, user_id: str) -> PersonalityProfile:
        """Get personality profile, creating default if not exists."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM personality_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

        if row:
            return PersonalityProfile(
                user_id=row['user_id'],
                assertiveness=row['assertiveness'],
                warmth=row['warmth'],
                detail_orientation=row['detail_orientation'],
                formality=row['formality'],
                humor_level=row['humor_level'],
                pace=row['pace'],
                directness=row['directness']
            )

        # Return default profile
        return PersonalityProfile(user_id=user_id)
