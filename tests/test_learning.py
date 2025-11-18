"""Tests for learning components."""

import pytest
import os
from src.learning.multi_modal import MultiModalLearner
from src.learning.integrator import LearningIntegrator
from src.models.feedback import FeedbackEvent, FeedbackType, EmotionalTone
from src.models.base import Context


class TestMultiModalLearner:
    """Tests for MultiModalLearner."""

    def setup_method(self):
        self.learner = MultiModalLearner()

    @pytest.mark.asyncio
    async def test_process_feedback(self):
        context = Context(user_id="test_user")
        event = FeedbackEvent(
            user_id="test_user",
            type=FeedbackType.DIRECT_CORRECTION,
            data={
                "wrong_behavior": "scheduled at 8am",
                "correct_behavior": "schedule after 9am"
            },
            context=context,
            emotional_tone=EmotionalTone.FRUSTRATED
        )

        updates = await self.learner.process_feedback(event)
        assert len(updates) > 0
        assert updates[0].confidence > 0.8  # High confidence for explicit correction

    @pytest.mark.asyncio
    async def test_process_rule_definition(self):
        context = Context(user_id="test_user")
        event = FeedbackEvent(
            user_id="test_user",
            type=FeedbackType.RULE_DEFINITION,
            data={
                "condition": "before 9am",
                "preferred_action": "decline meetings"
            },
            context=context
        )

        updates = await self.learner.process_feedback(event)
        assert len(updates) > 0
        assert updates[0].confidence > 0.95  # Very high for explicit rules


class TestLearningIntegrator:
    """Tests for LearningIntegrator."""

    def setup_method(self):
        self.integrator = LearningIntegrator()

    @pytest.mark.asyncio
    async def test_integrate_updates(self):
        from src.models.learning import LearningUpdate, UpdateType
        from src.models.base import Priority

        updates = [
            LearningUpdate(
                user_id="test_user",
                type=UpdateType.PREFERENCE_REFINEMENT,
                confidence=0.8,
                priority=Priority.MEDIUM,
                update_data={"test": "data1"}
            ),
            LearningUpdate(
                user_id="test_user",
                type=UpdateType.BEHAVIOR_CORRECTION,
                confidence=0.9,
                priority=Priority.HIGH,
                update_data={"test": "data2"}
            )
        ]

        integrated = await self.integrator.integrate(updates)
        assert len(integrated) == 2
        # Should be sorted by priority
        assert integrated[0].priority == Priority.HIGH

    @pytest.mark.asyncio
    async def test_conflict_resolution(self):
        from src.models.learning import LearningUpdate, UpdateType
        from src.models.base import Priority

        # Create conflicting updates
        explicit = LearningUpdate(
            user_id="test_user",
            type=UpdateType.EXPLICIT_RULE,
            confidence=0.99,
            priority=Priority.CRITICAL,
            update_data={"rule": "never before 9am"},
            affected_memories=["preference"],
            source="explicit"
        )

        implicit = LearningUpdate(
            user_id="test_user",
            type=UpdateType.PREFERENCE_REFINEMENT,
            confidence=0.6,
            priority=Priority.MEDIUM,
            update_data={"pattern": "usually schedules at 8am"},
            affected_memories=["preference"],
            source="implicit"
        )

        integrated = await self.integrator.integrate([explicit, implicit])
        # Explicit should take priority
        assert integrated[0].source == "explicit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
