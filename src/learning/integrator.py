"""Integrate learning updates with conflict resolution."""

from typing import List, Dict, Tuple
from datetime import datetime

from ..models.learning import LearningUpdate, UpdateType
from ..models.base import Priority


class LearningIntegrator:
    """Intelligently integrate multiple learning updates with conflict resolution."""

    def __init__(self):
        self.priority_order = {
            Priority.CRITICAL: 4,
            Priority.HIGH: 3,
            Priority.MEDIUM: 2,
            Priority.LOW: 1
        }

    async def integrate(self, updates: List[LearningUpdate]) -> List[LearningUpdate]:
        """Integrate updates with conflict resolution and consistency checking."""

        if not updates:
            return []

        # Detect and resolve conflicts
        resolved = await self._resolve_conflicts(updates)

        # Check consistency
        consistent = await self._check_consistency(resolved)

        # Sort by priority and confidence
        sorted_updates = self._sort_by_priority(consistent)

        return sorted_updates

    async def _resolve_conflicts(self, updates: List[LearningUpdate]) -> List[LearningUpdate]:
        """Resolve conflicts between updates."""

        conflicts = self._detect_conflicts(updates)

        for conflict in conflicts:
            resolution = self._resolve_conflict(conflict)
            updates = self._apply_resolution(updates, conflict, resolution)

        return updates

    def _detect_conflicts(self, updates: List[LearningUpdate]) -> List[Dict]:
        """Detect conflicts between updates."""
        conflicts = []

        for i, u1 in enumerate(updates):
            for u2 in updates[i+1:]:
                conflict_type = self._check_conflict(u1, u2)
                if conflict_type:
                    conflicts.append({
                        "type": conflict_type,
                        "updates": [u1, u2]
                    })

        return conflicts

    def _check_conflict(self, u1: LearningUpdate, u2: LearningUpdate) -> Optional[str]:
        """Check if two updates conflict."""

        # Same user, same affected memories
        if u1.user_id != u2.user_id:
            return None

        shared_memories = set(u1.affected_memories) & set(u2.affected_memories)
        if not shared_memories:
            return None

        # Check for contradictory updates
        if u1.source == "explicit" and u2.source == "implicit":
            return "explicit_vs_implicit"

        # Check for recency conflict
        time_diff = abs((u1.created_at - u2.created_at).total_seconds())
        if time_diff < 3600:  # Within 1 hour
            if u1.type == u2.type:
                return "recent_vs_historical"

        # Check for specificity conflict
        if self._is_more_specific(u1, u2) or self._is_more_specific(u2, u1):
            return "specific_vs_general"

        return None

    def _is_more_specific(self, u1: LearningUpdate, u2: LearningUpdate) -> bool:
        """Check if u1 is more specific than u2."""
        # Compare based on context depth
        ctx1 = u1.update_data.get("context", {})
        ctx2 = u2.update_data.get("context", {})

        return len(ctx1) > len(ctx2) * 1.5

    def _resolve_conflict(self, conflict: Dict) -> Dict:
        """Resolve a conflict between updates."""

        u1, u2 = conflict["updates"]
        conflict_type = conflict["type"]

        if conflict_type == "explicit_vs_implicit":
            # Explicit always wins
            winner = u1 if u1.source == "explicit" else u2
            loser = u2 if u1.source == "explicit" else u1
            return {
                "winner": winner,
                "loser": loser,
                "action": "discard_loser",
                "reason": "explicit_feedback_priority"
            }

        elif conflict_type == "recent_vs_historical":
            # Balance recency with consistency
            if u1.confidence > u2.confidence + 0.2:
                return {
                    "winner": u1,
                    "loser": u2,
                    "action": "reduce_loser_confidence",
                    "reason": "confidence_priority"
                }
            else:
                # More recent wins slightly
                winner = u1 if u1.created_at > u2.created_at else u2
                loser = u2 if u1.created_at > u2.created_at else u1
                return {
                    "winner": winner,
                    "loser": loser,
                    "action": "merge_with_recency_weight",
                    "reason": "recency_priority"
                }

        elif conflict_type == "specific_vs_general":
            # Specific rules override general
            winner = u1 if self._is_more_specific(u1, u2) else u2
            loser = u2 if self._is_more_specific(u1, u2) else u1
            return {
                "winner": winner,
                "loser": loser,
                "action": "add_exception_to_general",
                "reason": "specificity_priority"
            }

        return {"action": "keep_both"}

    def _apply_resolution(
        self,
        updates: List[LearningUpdate],
        conflict: Dict,
        resolution: Dict
    ) -> List[LearningUpdate]:
        """Apply conflict resolution to updates."""

        action = resolution.get("action", "keep_both")

        if action == "discard_loser":
            loser = resolution["loser"]
            updates = [u for u in updates if u.id != loser.id]

        elif action == "reduce_loser_confidence":
            loser = resolution["loser"]
            for u in updates:
                if u.id == loser.id:
                    u.confidence *= 0.5

        elif action == "merge_with_recency_weight":
            # Keep both but adjust confidence
            loser = resolution["loser"]
            for u in updates:
                if u.id == loser.id:
                    u.confidence *= 0.7

        elif action == "add_exception_to_general":
            # Mark the general rule as having an exception
            loser = resolution["loser"]
            winner = resolution["winner"]
            for u in updates:
                if u.id == loser.id:
                    if "exceptions" not in u.update_data:
                        u.update_data["exceptions"] = []
                    u.update_data["exceptions"].append({
                        "context": winner.update_data.get("context", {}),
                        "override_id": winner.id
                    })

        return updates

    async def _check_consistency(self, updates: List[LearningUpdate]) -> List[LearningUpdate]:
        """Check updates for internal consistency."""

        consistent = []

        for update in updates:
            # Basic consistency checks
            if update.confidence < 0 or update.confidence > 1:
                update.confidence = max(0, min(1, update.confidence))

            if not update.update_data:
                continue

            consistent.append(update)

        return consistent

    def _sort_by_priority(self, updates: List[LearningUpdate]) -> List[LearningUpdate]:
        """Sort updates by priority and confidence."""

        def sort_key(u: LearningUpdate) -> Tuple:
            return (
                self.priority_order.get(u.priority, 0),
                u.confidence,
                u.created_at.timestamp()
            )

        return sorted(updates, key=sort_key, reverse=True)


# Helper for type hints
from typing import Optional
