"""Apply learning updates to knowledge systems."""

from typing import List
from datetime import datetime
import uuid

from ..models.learning import LearningUpdate, UpdateType
from ..models.memory import MemoryEntry, PreferenceNode, Rule
from ..models.base import Priority, MemoryType
from ..storage.repositories import MemoryRepository, PreferenceRepository, RuleRepository


class KnowledgeUpdater:
    """Apply learning updates to persistent knowledge stores."""

    def __init__(
        self,
        memory_repo: MemoryRepository,
        pref_repo: PreferenceRepository,
        rule_repo: RuleRepository
    ):
        self.memory_repo = memory_repo
        self.pref_repo = pref_repo
        self.rule_repo = rule_repo

    async def apply_updates(self, updates: List[LearningUpdate]) -> List[str]:
        """Apply a list of learning updates, returning IDs of applied updates."""

        applied = []

        # Sort by priority
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3
        }
        updates.sort(key=lambda u: priority_order.get(u.priority, 4))

        for update in updates:
            try:
                await self._apply_single_update(update)
                applied.append(update.id)
            except Exception as e:
                # Log error but continue with other updates
                print(f"Error applying update {update.id}: {e}")

        return applied

    async def _apply_single_update(self, update: LearningUpdate):
        """Apply a single learning update."""

        if update.type == UpdateType.BEHAVIOR_CORRECTION:
            await self._apply_behavior_correction(update)
        elif update.type == UpdateType.PREFERENCE_REFINEMENT:
            await self._apply_preference_refinement(update)
        elif update.type == UpdateType.EXPLICIT_RULE:
            await self._apply_explicit_rule(update)
        elif update.type == UpdateType.BEHAVIORAL_PATTERN:
            await self._apply_behavioral_pattern(update)
        elif update.type == UpdateType.KNOWLEDGE_UPDATE:
            await self._apply_knowledge_update(update)

    async def _apply_behavior_correction(self, update: LearningUpdate):
        """Apply a behavior correction."""

        data = update.update_data
        user_id = update.user_id

        # Create a procedural memory for the correction
        memory = MemoryEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            memory_type=MemoryType.PROCEDURAL,
            content=f"When {data.get('incorrect_action', '')}: Instead do {data.get('correct_action', '')}",
            importance=0.9,  # High importance for corrections
            context_tags=["correction", "behavior"],
            metadata={
                "update_id": update.id,
                "context": data.get("context", {})
            }
        )
        self.memory_repo.save(memory)

        # Create or update rule
        if "rule" in data:
            rule_data = data["rule"]
            rule = Rule(
                id=str(uuid.uuid4()),
                user_id=user_id,
                condition=rule_data.get("condition", ""),
                action=rule_data.get("action", ""),
                strength=rule_data.get("strength", 1.0),
                source="learned_correction"
            )
            self.rule_repo.save(rule)

    async def _apply_preference_refinement(self, update: LearningUpdate):
        """Apply a preference refinement."""

        data = update.update_data
        user_id = update.user_id

        # Check for reinforcement
        if data.get("reinforcement") == "positive":
            # Strengthen existing similar preferences
            category = data.get("category", "general")
            prefs = self.pref_repo.get_by_user(user_id, category)

            behavior = data.get("behavior", "")
            for pref in prefs:
                if self._text_similarity(pref.preference, behavior) > 0.5:
                    new_strength = min(1.0, pref.strength + data.get("strength_boost", 0.1))
                    self.pref_repo.update_strength(pref.id, new_strength)

        elif data.get("reinforcement") == "negative":
            # Weaken preferences
            category = data.get("category", "general")
            prefs = self.pref_repo.get_by_user(user_id, category)

            behavior = data.get("behavior", "")
            for pref in prefs:
                if self._text_similarity(pref.preference, behavior) > 0.5:
                    new_strength = max(0.0, pref.strength - data.get("strength_reduction", 0.2))
                    self.pref_repo.update_strength(pref.id, new_strength)

        else:
            # Create new preference
            pref = PreferenceNode(
                id=str(uuid.uuid4()),
                user_id=user_id,
                category=data.get("category", "general"),
                preference=data.get("preference", data.get("inferred_preference", "")),
                strength=data.get("strength", update.confidence),
                confidence=update.confidence,
                source=update.source
            )
            self.pref_repo.save(pref)

    async def _apply_explicit_rule(self, update: LearningUpdate):
        """Apply an explicit rule definition."""

        data = update.update_data
        user_id = update.user_id

        rule = Rule(
            id=str(uuid.uuid4()),
            user_id=user_id,
            condition=data.get("condition", ""),
            action=data.get("action", ""),
            strength=data.get("strength", 1.0),
            exceptions=data.get("exceptions", []),
            source="user_explicit"
        )
        self.rule_repo.save(rule)

        # Also store as preference
        pref = PreferenceNode(
            id=str(uuid.uuid4()),
            user_id=user_id,
            category="rules",
            preference=f"{data.get('condition', '')} -> {data.get('action', '')}",
            strength=1.0,
            confidence=0.99,
            source="explicit"
        )
        self.pref_repo.save(pref)

    async def _apply_behavioral_pattern(self, update: LearningUpdate):
        """Apply behavioral pattern learnings."""

        data = update.update_data
        user_id = update.user_id

        # Store pattern shifts as memories
        for shift in data.get("pattern_shifts", []):
            memory = MemoryEntry(
                id=str(uuid.uuid4()),
                user_id=user_id,
                memory_type=MemoryType.SEMANTIC,
                content=f"Pattern shift: {shift.get('pattern_type', '')} changed from {shift.get('old_pattern', {})} to {shift.get('new_pattern', {})}",
                importance=shift.get("confidence", 0.5),
                context_tags=["pattern", "behavioral"],
                metadata=shift
            )
            self.memory_repo.save(memory)

        # Store inferred preferences
        for pref_data in data.get("inferred_preferences", []):
            pref = PreferenceNode(
                id=str(uuid.uuid4()),
                user_id=user_id,
                category=pref_data.get("category", "behavioral"),
                preference=pref_data.get("preference", ""),
                strength=pref_data.get("confidence", 0.5),
                confidence=pref_data.get("confidence", 0.5),
                source="inferred"
            )
            self.pref_repo.save(pref)

    async def _apply_knowledge_update(self, update: LearningUpdate):
        """Apply general knowledge update."""

        data = update.update_data
        user_id = update.user_id

        memory = MemoryEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            memory_type=MemoryType(data.get("memory_type", "semantic")),
            content=data.get("content", ""),
            importance=update.confidence,
            context_tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
        self.memory_repo.save(memory)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        return overlap / total if total > 0 else 0.0
