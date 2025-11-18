"""Prevent catastrophic forgetting of important knowledge."""

from typing import List, Dict
from datetime import datetime, timedelta
import uuid

from ..models.learning import LearningUpdate, KnowledgeProtection
from ..models.memory import MemoryEntry
from ..storage.repositories import MemoryRepository


class ForgettingPreventionSystem:
    """System to prevent catastrophic forgetting of important knowledge."""

    def __init__(self, memory_repo: MemoryRepository):
        self.memory_repo = memory_repo
        self.protected_knowledge: Dict[str, KnowledgeProtection] = {}
        self.rehearsal_schedule: List[Dict] = []

    async def protect_before_update(
        self,
        user_id: str,
        new_knowledge: LearningUpdate
    ) -> float:
        """
        Protect existing knowledge before applying new updates.
        Returns adaptive learning rate.
        """

        # Find affected knowledge
        affected = await self._find_affected_knowledge(user_id, new_knowledge)

        if not affected:
            return 0.3  # Default learning rate

        # Calculate importance of affected knowledge
        importance_scores = self._calculate_importance(affected)

        # Protect high-importance knowledge
        max_importance = 0.0
        for memory in affected:
            score = importance_scores.get(memory.id, 0.0)
            if score > 0.7:
                await self._protect_memory(memory, score)
                max_importance = max(max_importance, score)

        # Calculate adaptive learning rate
        learning_rate = self._calculate_adaptive_rate(max_importance)

        return learning_rate

    async def _find_affected_knowledge(
        self,
        user_id: str,
        new_knowledge: LearningUpdate
    ) -> List[MemoryEntry]:
        """Find existing knowledge that might be affected by new knowledge."""

        affected = []

        # Get existing memories
        memories = self.memory_repo.get_by_user(user_id, limit=1000)

        # Extract key terms from new knowledge
        new_content = str(new_knowledge.update_data)
        new_terms = set(new_content.lower().split())

        for memory in memories:
            # Check for overlap
            memory_terms = set(memory.content.lower().split())
            overlap = new_terms & memory_terms

            # If significant overlap, might be affected
            if len(overlap) > 3 or (len(overlap) > 0 and len(overlap) / len(memory_terms) > 0.3):
                affected.append(memory)

        return affected

    def _calculate_importance(self, memories: List[MemoryEntry]) -> Dict[str, float]:
        """Calculate importance scores for memories."""

        scores = {}

        for memory in memories:
            # Base importance
            score = memory.importance

            # Boost for access frequency
            if memory.access_count > 10:
                score = min(1.0, score + 0.1)
            elif memory.access_count > 5:
                score = min(1.0, score + 0.05)

            # Boost for recency
            days_old = (datetime.utcnow() - memory.last_accessed).days
            if days_old < 7:
                score = min(1.0, score + 0.1)
            elif days_old < 30:
                score = min(1.0, score + 0.05)

            scores[memory.id] = score

        return scores

    async def _protect_memory(self, memory: MemoryEntry, importance: float):
        """Add memory to protection list."""

        protection = KnowledgeProtection(
            knowledge_id=memory.id,
            importance=importance,
            protection_level="elevated" if importance > 0.9 else "normal"
        )

        self.protected_knowledge[memory.id] = protection

        # Schedule rehearsal
        await self._schedule_rehearsal(memory, importance)

    async def _schedule_rehearsal(self, memory: MemoryEntry, importance: float):
        """Schedule rehearsal of protected memory."""

        # More important = more frequent rehearsal
        if importance > 0.9:
            interval_days = 1
        elif importance > 0.7:
            interval_days = 3
        else:
            interval_days = 7

        self.rehearsal_schedule.append({
            "memory_id": memory.id,
            "user_id": memory.user_id,
            "next_rehearsal": datetime.utcnow() + timedelta(days=interval_days),
            "importance": importance
        })

    def _calculate_adaptive_rate(self, max_importance: float) -> float:
        """Calculate learning rate based on affected knowledge importance."""

        base_rate = 0.3

        if max_importance > 0.9:
            # Drastically reduce for critical knowledge
            return base_rate * 0.1
        elif max_importance > 0.7:
            # Moderate reduction
            return base_rate * 0.5
        elif max_importance > 0.5:
            # Slight reduction
            return base_rate * 0.8
        else:
            return base_rate

    async def perform_rehearsals(self, user_id: str):
        """Perform scheduled rehearsals for a user."""

        now = datetime.utcnow()
        to_rehearse = []

        for item in self.rehearsal_schedule:
            if item["user_id"] == user_id and item["next_rehearsal"] <= now:
                to_rehearse.append(item)

        for item in to_rehearse:
            memory_id = item["memory_id"]

            # Update access to reinforce memory
            self.memory_repo.update_access(memory_id)

            # Update protection record
            if memory_id in self.protected_knowledge:
                protection = self.protected_knowledge[memory_id]
                protection.last_rehearsed = now
                protection.rehearsal_count += 1

            # Reschedule based on rehearsal count (spaced repetition)
            protection = self.protected_knowledge.get(memory_id)
            if protection:
                # Increase interval with each rehearsal
                interval = min(30, 2 ** protection.rehearsal_count)
                item["next_rehearsal"] = now + timedelta(days=interval)

    async def get_protection_status(self, user_id: str) -> Dict:
        """Get protection status for a user's knowledge."""

        user_protections = {
            k: v for k, v in self.protected_knowledge.items()
            if self._get_memory_user(k) == user_id
        }

        return {
            "protected_count": len(user_protections),
            "critical_count": sum(1 for p in user_protections.values() if p.protection_level == "critical"),
            "scheduled_rehearsals": sum(1 for s in self.rehearsal_schedule if s["user_id"] == user_id)
        }

    def _get_memory_user(self, memory_id: str) -> str:
        """Get user ID for a memory."""
        memory = self.memory_repo.get(memory_id)
        return memory.user_id if memory else ""
