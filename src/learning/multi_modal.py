"""Multi-modal learner that processes all types of feedback."""

from typing import List, Optional, Dict
import asyncio

from ..models.feedback import FeedbackEvent, InteractionEvent
from ..models.learning import LearningUpdate
from ..feedback.explicit import ExplicitFeedbackProcessor
from ..feedback.implicit import ImplicitSignalExtractor
from ..feedback.behavioral import BehavioralPatternAnalyzer


class MultiModalLearner:
    """Process feedback from multiple modalities simultaneously."""

    def __init__(self):
        self.explicit_processor = ExplicitFeedbackProcessor()
        self.implicit_extractor = ImplicitSignalExtractor()
        self.behavioral_analyzer = BehavioralPatternAnalyzer()

        self.learning_rates = {
            "immediate": 0.8,    # Direct corrections
            "short_term": 0.3,   # Pattern changes
            "long_term": 0.1     # Gradual preference shifts
        }

    async def process_feedback(self, event: FeedbackEvent) -> List[LearningUpdate]:
        """Process a feedback event through all relevant processors."""

        updates = []

        # Process through explicit processor
        explicit_update = await self.explicit_processor.process(event)
        if explicit_update:
            explicit_update.update_data["learning_rate"] = self.learning_rates["immediate"]
            updates.append(explicit_update)

        # Process through implicit extractor
        implicit_update = await self.implicit_extractor.process(event)
        if implicit_update:
            implicit_update.update_data["learning_rate"] = self.learning_rates["short_term"]
            updates.append(implicit_update)

        return updates

    async def process_interaction(self, event: InteractionEvent) -> Optional[LearningUpdate]:
        """Process an interaction event for behavioral patterns."""

        update = await self.behavioral_analyzer.process(event)
        if update:
            update.update_data["learning_rate"] = self.learning_rates["long_term"]

        return update

    async def process_batch(
        self,
        feedback_events: List[FeedbackEvent],
        interaction_events: List[InteractionEvent]
    ) -> List[LearningUpdate]:
        """Process multiple events in batch."""

        all_updates = []

        # Process feedback events
        for event in feedback_events:
            updates = await self.process_feedback(event)
            all_updates.extend(updates)

        # Process interaction events
        for event in interaction_events:
            update = await self.process_interaction(event)
            if update:
                all_updates.append(update)

        return all_updates

    def get_adoption_rate(self, user_id: str, suggestion_type: str) -> float:
        """Get adoption rate for a specific suggestion type."""
        return self.implicit_extractor.get_adoption_rate(user_id, suggestion_type)
