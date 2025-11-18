"""Central Nervous System - Main orchestrator for the adaptive agent."""

from typing import Dict, List, Optional
from datetime import datetime

from ..models.base import Context, Stimulus, ActionType
from ..models.action import ActionPlan, Decision, ActionResult
from ..models.feedback import FeedbackEvent, InteractionEvent, FeedbackType, EmotionalTone
from ..models.learning import LearningUpdate

from ..storage.database import Database
from ..storage.repositories import (
    MemoryRepository, PreferenceRepository, RuleRepository,
    FeedbackRepository, InteractionRepository, PersonalityRepository
)

from ..memory.hierarchical import HierarchicalMemory
from ..personality.adaptive import AdaptivePersonality
from ..reasoning.meta_reasoning import MetaReasoningEngine
from ..alignment.value_alignment import ValueAlignmentSystem
from ..execution.transparent import TransparentExecutor

from ..learning.multi_modal import MultiModalLearner
from ..learning.integrator import LearningIntegrator
from ..learning.knowledge_updater import KnowledgeUpdater
from ..learning.forgetting_prevention import ForgettingPreventionSystem


class CentralNervousSystem:
    """
    Central Nervous System - The main orchestrator for the adaptive AI agent.

    Coordinates:
    - Memory retrieval and storage
    - Personality adaptation
    - Meta-reasoning and decision making
    - Value alignment validation
    - Transparent action execution
    - Continuous learning from feedback
    """

    def __init__(self, user_id: str, db_path: str = "agent_memory.db"):
        self.user_id = user_id

        # Initialize database and repositories
        self.db = Database(db_path)
        self.memory_repo = MemoryRepository(self.db)
        self.pref_repo = PreferenceRepository(self.db)
        self.rule_repo = RuleRepository(self.db)
        self.feedback_repo = FeedbackRepository(self.db)
        self.interaction_repo = InteractionRepository(self.db)
        self.personality_repo = PersonalityRepository(self.db)

        # Initialize core systems
        self.memory = HierarchicalMemory(self.memory_repo, self.pref_repo)
        self.personality = AdaptivePersonality(self.personality_repo)
        self.reasoning = MetaReasoningEngine()
        self.alignment = ValueAlignmentSystem(self.rule_repo, self.pref_repo)
        self.executor = TransparentExecutor()

        # Initialize learning systems
        self.learner = MultiModalLearner()
        self.integrator = LearningIntegrator()
        self.knowledge_updater = KnowledgeUpdater(
            self.memory_repo, self.pref_repo, self.rule_repo
        )
        self.forgetting_prevention = ForgettingPreventionSystem(self.memory_repo)

    async def process_stimulus(
        self,
        stimulus: Stimulus,
        proposed_action: Optional[ActionPlan] = None
    ) -> ActionResult:
        """
        Main entry point - process a stimulus and produce a response.

        This is the core loop:
        1. Enrich context with memories
        2. Apply personality filters
        3. Meta-reasoning to decide action type
        4. Value alignment validation
        5. Execute with transparency
        """

        # 1. Enrich context
        enriched_context = await self._enrich_context(stimulus)

        # 2. Get personality tone
        tone = self.personality.generate_response_tone(
            self.user_id,
            enriched_context
        )

        # 3. Get relevant preferences and rules
        preferences = await self.memory.preference_graph.get_preferences(
            self.user_id, min_strength=0.3
        )
        rules = self.rule_repo.get_by_user(self.user_id)

        # 4. Meta-reasoning
        decision = await self.reasoning.deliberate(
            context=enriched_context,
            preferences=preferences,
            rules=rules,
            proposed_action=proposed_action
        )

        # 5. Value alignment
        validated_decision = await self.alignment.validate(self.user_id, decision)

        # 6. Execute with transparency
        result = await self.executor.execute(validated_decision)

        # 7. Record interaction for learning
        await self._record_interaction(stimulus, result)

        return result

    async def _enrich_context(self, stimulus: Stimulus) -> Dict:
        """Enrich the stimulus context with relevant memories and preferences."""

        # Build query from stimulus
        query = f"{stimulus.type} {' '.join(str(v) for v in stimulus.data.values())}"

        # Retrieve relevant context
        retrieved = await self.memory.retrieve_relevant_context(
            user_id=self.user_id,
            query=query,
            context=stimulus.context
        )

        # Build enriched context
        enriched = {
            "user_id": self.user_id,
            "stimulus_type": stimulus.type,
            "stimulus_data": stimulus.data,
            "time_of_day": stimulus.context.time_of_day,
            "day_type": stimulus.context.day_type,
            "user_state": stimulus.context.user_state,
            "relevant_memories": [
                {"content": m.content, "score": s}
                for m, s in retrieved["memories"][:5]
            ],
            "relevant_preferences": [
                {"preference": p.preference, "strength": p.strength}
                for p in retrieved["preferences"][:5]
            ],
            "historical_success": self.executor.get_success_rate(self.user_id)
        }

        return enriched

    async def _record_interaction(
        self,
        stimulus: Stimulus,
        result: ActionResult
    ):
        """Record the interaction for learning."""

        interaction = InteractionEvent(
            user_id=self.user_id,
            event_type=stimulus.type,
            content=str(stimulus.data),
            response=result.explanation,
            context=stimulus.context,
            engagement_score=result.confidence
        )

        self.interaction_repo.save(interaction)

        # Process for behavioral patterns
        update = await self.learner.process_interaction(interaction)
        if update:
            await self._apply_learning([update])

    async def provide_feedback(
        self,
        feedback_type: FeedbackType,
        data: Dict,
        emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    ):
        """Process explicit user feedback."""

        context = Context(
            user_id=self.user_id,
            user_state=data.get("user_state", {})
        )

        event = FeedbackEvent(
            user_id=self.user_id,
            type=feedback_type,
            data=data,
            context=context,
            emotional_tone=emotional_tone
        )

        # Save feedback
        self.feedback_repo.save(event)

        # Process through learner
        updates = await self.learner.process_feedback(event)

        # Apply learning
        if updates:
            await self._apply_learning(updates)

        # Adapt personality if relevant
        if "interaction" in data:
            interaction = InteractionEvent(
                user_id=self.user_id,
                event_type="feedback",
                content=str(data),
                context=context
            )
            await self.personality.adapt_from_interaction(self.user_id, interaction)

    async def _apply_learning(self, updates: List[LearningUpdate]):
        """Apply learning updates to knowledge systems."""

        if not updates:
            return

        # Integrate updates
        integrated = await self.integrator.integrate(updates)

        # Prevent catastrophic forgetting
        for update in integrated:
            learning_rate = await self.forgetting_prevention.protect_before_update(
                self.user_id, update
            )
            update.update_data["learning_rate"] = learning_rate

        # Apply to knowledge stores
        await self.knowledge_updater.apply_updates(integrated)

    async def add_rule(self, condition: str, action: str):
        """Add an explicit rule."""

        await self.provide_feedback(
            FeedbackType.RULE_DEFINITION,
            {
                "condition": condition,
                "preferred_action": action
            }
        )

    async def add_preference(self, category: str, preference: str, strength: float = 0.8):
        """Add an explicit preference."""

        await self.provide_feedback(
            FeedbackType.PREFERENCE_STATEMENT,
            {
                "category": category,
                "preference": preference,
                "strength": strength
            }
        )

    async def correct_behavior(
        self,
        wrong_behavior: str,
        correct_behavior: str,
        emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    ):
        """Correct agent behavior."""

        await self.provide_feedback(
            FeedbackType.DIRECT_CORRECTION,
            {
                "wrong_behavior": wrong_behavior,
                "correct_behavior": correct_behavior
            },
            emotional_tone=emotional_tone
        )

    async def get_profile_summary(self) -> Dict:
        """Get a summary of the user's profile and learned patterns."""

        profile = self.personality.get_profile(self.user_id)
        preferences = self.pref_repo.get_by_user(self.user_id)
        rules = self.rule_repo.get_by_user(self.user_id)
        protection_status = await self.forgetting_prevention.get_protection_status(
            self.user_id
        )

        return {
            "personality": {
                "warmth": profile.warmth,
                "formality": profile.formality,
                "detail_orientation": profile.detail_orientation,
                "pace": profile.pace,
                "directness": profile.directness
            },
            "preferences_count": len(preferences),
            "top_preferences": [
                {"preference": p.preference, "strength": p.strength}
                for p in preferences[:5]
            ],
            "rules_count": len(rules),
            "knowledge_protection": protection_status,
            "success_rate": self.executor.get_success_rate(self.user_id)
        }

    async def consolidate_memory(self):
        """Run memory consolidation (like sleep)."""
        await self.memory.consolidate(self.user_id)
        await self.forgetting_prevention.perform_rehearsals(self.user_id)

    async def store_knowledge(
        self,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
        tags: List[str] = None
    ):
        """Store new knowledge."""

        from ..models.base import MemoryType
        mem_type = MemoryType(memory_type)

        await self.memory.store(
            user_id=self.user_id,
            content=content,
            memory_type=mem_type,
            importance=importance,
            context_tags=tags or []
        )
