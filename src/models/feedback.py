"""Feedback-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

from .base import BaseModel, Context


class FeedbackType(str, Enum):
    DIRECT_CORRECTION = "direct_correction"
    PRAISE = "praise"
    CRITICISM = "criticism"
    RULE_DEFINITION = "rule_definition"
    PREFERENCE_STATEMENT = "preference_statement"
    SUGGESTION_IGNORED = "suggestion_ignored"
    SUGGESTION_MODIFIED = "suggestion_modified"
    SUGGESTION_ACCEPTED = "suggestion_accepted"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class EmotionalTone(str, Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"
    PLEASED = "pleased"
    CONFUSED = "confused"


@dataclass
class FeedbackEvent(BaseModel):
    """An event representing user feedback."""
    user_id: str = ""
    type: FeedbackType = FeedbackType.DIRECT_CORRECTION
    data: dict = field(default_factory=dict)
    context: Optional[Context] = None
    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    processed: bool = False

    def to_dict(self) -> dict:
        result = super().to_dict()
        result['type'] = self.type.value
        result['emotional_tone'] = self.emotional_tone.value
        if self.context:
            result['context'] = self.context.to_dict()
        return result


@dataclass
class LearningSignal:
    """A signal extracted from feedback that can be used for learning."""
    type: str
    pattern: dict
    confidence: float
    learning_rate: float = 0.3
    source: str = "implicit"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "pattern": self.pattern,
            "confidence": self.confidence,
            "learning_rate": self.learning_rate,
            "source": self.source
        }


@dataclass
class InteractionEvent(BaseModel):
    """A record of an interaction for pattern analysis."""
    user_id: str = ""
    event_type: str = ""
    content: str = ""
    response: str = ""
    context: Optional[Context] = None
    duration_seconds: float = 0.0
    engagement_score: float = 0.5
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = super().to_dict()
        if self.context:
            result['context'] = self.context.to_dict()
        return result
