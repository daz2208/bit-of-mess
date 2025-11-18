"""Memory-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .base import BaseModel, MemoryType


@dataclass
class MemoryEntry(BaseModel):
    """A single memory entry in the system."""
    user_id: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    content: str = ""
    embedding: np.ndarray | None = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    context_tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = super().to_dict()
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        result['memory_type'] = self.memory_type.value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryEntry':
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        if 'memory_type' in data:
            data['memory_type'] = MemoryType(data['memory_type'])
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'last_accessed' in data and isinstance(data['last_accessed'], str):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


@dataclass
class PreferenceNode:
    """A node in the preference graph."""
    id: str = ""
    user_id: str = ""
    category: str = ""  # e.g., "communication", "scheduling", "content"
    preference: str = ""
    strength: float = 0.5  # 0-1, how strong the preference is
    confidence: float = 0.5  # 0-1, how confident we are
    examples: list = field(default_factory=list)
    exceptions: list = field(default_factory=list)
    source: str = "learned"  # "explicit", "learned", "inferred"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "category": self.category,
            "preference": self.preference,
            "strength": self.strength,
            "confidence": self.confidence,
            "examples": self.examples,
            "exceptions": self.exceptions,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class Rule:
    """An explicit or learned rule."""
    id: str = ""
    user_id: str = ""
    condition: str = ""
    action: str = ""
    strength: float = 1.0
    exceptions: list = field(default_factory=list)
    source: str = "explicit"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "condition": self.condition,
            "action": self.action,
            "strength": self.strength,
            "exceptions": self.exceptions,
            "source": self.source,
            "created_at": self.created_at.isoformat()
        }
