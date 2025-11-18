"""Base data models for the agent system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum
import uuid


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryType(str, Enum):
    EPISODIC = "episodic"      # Specific events
    SEMANTIC = "semantic"       # Concepts & knowledge
    PROCEDURAL = "procedural"   # How-to knowledge
    PREFERENCE = "preference"   # Tastes & values


class ActionType(str, Enum):
    AUTONOMOUS_EXECUTE = "autonomous_execute"
    SUGGEST_WITH_RATIONALE = "suggest_with_rationale"
    ASK_CLARIFICATION = "ask_clarification"
    SILENT_LEARN = "silent_learn"


@dataclass
class BaseModel:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if hasattr(item, 'to_dict') else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


@dataclass
class Context:
    """Contextual information for any operation."""
    user_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    time_of_day: str = ""
    day_type: str = ""  # workday, weekend, holiday
    user_state: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.time_of_day:
            hour = self.timestamp.hour
            if 5 <= hour < 12:
                self.time_of_day = "morning"
            elif 12 <= hour < 17:
                self.time_of_day = "afternoon"
            elif 17 <= hour < 21:
                self.time_of_day = "evening"
            else:
                self.time_of_day = "night"

        if not self.day_type:
            self.day_type = "weekend" if self.timestamp.weekday() >= 5 else "workday"

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "time_of_day": self.time_of_day,
            "day_type": self.day_type,
            "user_state": self.user_state,
            "environment": self.environment,
            "metadata": self.metadata
        }


@dataclass
class Stimulus:
    """Input stimulus to the agent."""
    type: str
    data: dict
    context: Context
    source: str = "user"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "data": self.data,
            "context": self.context.to_dict(),
            "source": self.source
        }
