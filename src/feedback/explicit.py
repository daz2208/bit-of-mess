"""Process explicit user feedback like corrections and rule definitions."""

from typing import Optional
from datetime import datetime
import uuid

from ..models.feedback import FeedbackEvent, FeedbackType, LearningSignal
from ..models.learning import LearningUpdate, UpdateType
from ..models.base import Priority


class ExplicitFeedbackProcessor:
    """Process explicit user feedback into learning updates."""

    def __init__(self):
        self.feedback_handlers = {
            FeedbackType.DIRECT_CORRECTION: self._handle_direct_correction,
            FeedbackType.PRAISE: self._handle_praise,
            FeedbackType.CRITICISM: self._handle_criticism,
            FeedbackType.RULE_DEFINITION: self._handle_rule_definition,
            FeedbackType.PREFERENCE_STATEMENT: self._handle_preference_statement
        }

    async def process(self, event: FeedbackEvent) -> Optional[LearningUpdate]:
        """Process a feedback event into a learning update."""

        handler = self.feedback_handlers.get(event.type)
        if not handler:
            return None

        return await handler(event)

    async def _handle_direct_correction(self, event: FeedbackEvent) -> LearningUpdate:
        """Handle when user directly corrects agent behavior."""

        wrong = event.data.get("wrong_behavior", "")
        correct = event.data.get("correct_behavior", "")

        return LearningUpdate(
            id=str(uuid.uuid4()),
            user_id=event.user_id,
            type=UpdateType.BEHAVIOR_CORRECTION,
            confidence=0.95,  # High confidence for direct corrections
            priority=Priority.HIGH,
            update_data={
                "incorrect_action": wrong,
                "correct_action": correct,
                "context": event.context.to_dict() if event.context else {},
                "emotional_tone": event.emotional_tone.value,
                "rule": {
                    "condition": f"When {wrong.lower()} is attempted",
                    "action": correct,
                    "strength": 1.0
                }
            },
            affected_memories=["procedural", "preference"],
            source="explicit"
        )

    async def _handle_praise(self, event: FeedbackEvent) -> LearningUpdate:
        """Handle positive feedback to reinforce behavior."""

        praised_behavior = event.data.get("behavior", "")

        return LearningUpdate(
            id=str(uuid.uuid4()),
            user_id=event.user_id,
            type=UpdateType.PREFERENCE_REFINEMENT,
            confidence=0.8,
            priority=Priority.MEDIUM,
            update_data={
                "behavior": praised_behavior,
                "reinforcement": "positive",
                "strength_boost": 0.2,
                "context": event.context.to_dict() if event.context else {}
            },
            affected_memories=["preference"],
            source="explicit"
        )

    async def _handle_criticism(self, event: FeedbackEvent) -> LearningUpdate:
        """Handle negative feedback to discourage behavior."""

        criticized_behavior = event.data.get("behavior", "")

        return LearningUpdate(
            id=str(uuid.uuid4()),
            user_id=event.user_id,
            type=UpdateType.PREFERENCE_REFINEMENT,
            confidence=0.85,
            priority=Priority.HIGH,
            update_data={
                "behavior": criticized_behavior,
                "reinforcement": "negative",
                "strength_reduction": 0.3,
                "context": event.context.to_dict() if event.context else {}
            },
            affected_memories=["preference", "procedural"],
            source="explicit"
        )

    async def _handle_rule_definition(self, event: FeedbackEvent) -> LearningUpdate:
        """Handle explicit rule definition from user."""

        return LearningUpdate(
            id=str(uuid.uuid4()),
            user_id=event.user_id,
            type=UpdateType.EXPLICIT_RULE,
            confidence=0.99,  # User-defined rules are highest confidence
            priority=Priority.CRITICAL,
            update_data={
                "condition": event.data.get("condition", ""),
                "action": event.data.get("preferred_action", ""),
                "strength": 1.0,
                "exceptions": [],
                "source": "user_explicit"
            },
            affected_memories=["preference"],
            source="explicit"
        )

    async def _handle_preference_statement(self, event: FeedbackEvent) -> LearningUpdate:
        """Handle explicit preference statements."""

        return LearningUpdate(
            id=str(uuid.uuid4()),
            user_id=event.user_id,
            type=UpdateType.PREFERENCE_REFINEMENT,
            confidence=0.9,
            priority=Priority.HIGH,
            update_data={
                "category": event.data.get("category", "general"),
                "preference": event.data.get("preference", ""),
                "strength": 0.9,
                "context": event.context.to_dict() if event.context else {}
            },
            affected_memories=["preference"],
            source="explicit"
        )
