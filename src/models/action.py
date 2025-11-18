"""Action and decision-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .base import BaseModel, ActionType


@dataclass
class ActionPlan:
    """A planned action to be executed."""
    description: str
    action_type: ActionType
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "action_type": self.action_type.value,
            "parameters": self.parameters
        }


@dataclass
class Decision(BaseModel):
    """A decision made by the reasoning engine."""
    user_id: str = ""
    action_type: ActionType = ActionType.ASK_CLARIFICATION
    action_plan: Optional[ActionPlan] = None
    confidence: float = 0.5
    intrusion_score: float = 0.5
    value_score: float = 0.5
    reasoning_chain: list = field(default_factory=list)
    transparency_level: str = "summary"  # full, summary, on_request

    def to_dict(self) -> dict:
        result = super().to_dict()
        result['action_type'] = self.action_type.value
        if self.action_plan:
            result['action_plan'] = self.action_plan.to_dict()
        return result


@dataclass
class ActionResult(BaseModel):
    """Result of executing an action."""
    user_id: str = ""
    decision_id: str = ""
    success: bool = True
    result_data: dict = field(default_factory=dict)
    explanation: str = ""
    confidence: float = 0.5
    user_feedback: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "decision_id": self.decision_id,
            "success": self.success,
            "result_data": self.result_data,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "user_feedback": self.user_feedback,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class PersonalityProfile:
    """User's personality profile for adaptation."""
    user_id: str = ""
    assertiveness: float = 0.5
    warmth: float = 0.5
    detail_orientation: float = 0.5
    formality: float = 0.5
    humor_level: float = 0.3
    pace: float = 0.5  # slow vs fast
    directness: float = 0.5

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "assertiveness": self.assertiveness,
            "warmth": self.warmth,
            "detail_orientation": self.detail_orientation,
            "formality": self.formality,
            "humor_level": self.humor_level,
            "pace": self.pace,
            "directness": self.directness
        }

    def get_trait(self, trait: str) -> float:
        return getattr(self, trait, 0.5)

    def set_trait(self, trait: str, value: float):
        if hasattr(self, trait):
            setattr(self, trait, max(0.0, min(1.0, value)))
