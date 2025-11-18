"""Learning-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

from .base import BaseModel, Priority, MemoryType


class UpdateType(str, Enum):
    BEHAVIOR_CORRECTION = "behavior_correction"
    PREFERENCE_REFINEMENT = "preference_refinement"
    EXPLICIT_RULE = "explicit_rule"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    KNOWLEDGE_UPDATE = "knowledge_update"


@dataclass
class LearningUpdate(BaseModel):
    """An update to be applied to the knowledge system."""
    user_id: str = ""
    type: UpdateType = UpdateType.KNOWLEDGE_UPDATE
    confidence: float = 0.5
    priority: Priority = Priority.MEDIUM
    update_data: dict = field(default_factory=dict)
    affected_memories: list = field(default_factory=list)
    source: str = "implicit"
    applied: bool = False
    impact_score: float = 0.0

    def to_dict(self) -> dict:
        result = super().to_dict()
        result['type'] = self.type.value
        result['priority'] = self.priority.value
        return result


@dataclass
class PatternShift:
    """Represents a detected shift in behavioral patterns."""
    pattern_type: str
    old_pattern: dict
    new_pattern: dict
    confidence: float
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type,
            "old_pattern": self.old_pattern,
            "new_pattern": self.new_pattern,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class KnowledgeProtection:
    """Tracks protected knowledge to prevent catastrophic forgetting."""
    knowledge_id: str
    importance: float
    last_rehearsed: datetime = field(default_factory=datetime.utcnow)
    rehearsal_count: int = 0
    protection_level: str = "normal"  # normal, elevated, critical

    def to_dict(self) -> dict:
        return {
            "knowledge_id": self.knowledge_id,
            "importance": self.importance,
            "last_rehearsed": self.last_rehearsed.isoformat(),
            "rehearsal_count": self.rehearsal_count,
            "protection_level": self.protection_level
        }
