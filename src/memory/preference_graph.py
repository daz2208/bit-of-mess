"""Preference graph for tracking user preferences and their relationships."""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import uuid

from ..models.memory import PreferenceNode
from ..storage.repositories import PreferenceRepository


class PreferenceGraph:
    """Graph-based preference system for tracking and querying user preferences."""

    def __init__(self, pref_repo: PreferenceRepository):
        self.pref_repo = pref_repo

    async def add_preference(
        self,
        user_id: str,
        category: str,
        preference: str,
        strength: float = 0.5,
        confidence: float = 0.5,
        source: str = "learned",
        examples: List = None
    ) -> str:
        """Add or update a preference."""

        # Check if similar preference exists
        existing = await self.find_similar_preference(user_id, category, preference)

        if existing:
            # Update existing preference
            existing.strength = min(1.0, existing.strength + (strength * 0.2))
            existing.confidence = min(1.0, (existing.confidence + confidence) / 2)
            if examples:
                existing.examples.extend(examples)
            existing.updated_at = datetime.utcnow()
            return self.pref_repo.save(existing)

        # Create new preference
        pref = PreferenceNode(
            id=str(uuid.uuid4()),
            user_id=user_id,
            category=category,
            preference=preference,
            strength=strength,
            confidence=confidence,
            examples=examples or [],
            source=source
        )
        return self.pref_repo.save(pref)

    async def find_similar_preference(
        self,
        user_id: str,
        category: str,
        preference: str
    ) -> Optional[PreferenceNode]:
        """Find a similar existing preference."""
        prefs = self.pref_repo.get_by_user(user_id, category)

        # Simple keyword matching
        pref_words = set(preference.lower().split())

        for p in prefs:
            existing_words = set(p.preference.lower().split())
            overlap = len(pref_words & existing_words)
            if overlap >= len(pref_words) * 0.6:  # 60% overlap
                return p

        return None

    async def get_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
        min_strength: float = 0.0
    ) -> List[PreferenceNode]:
        """Get user preferences filtered by criteria."""
        prefs = self.pref_repo.get_by_user(user_id, category)
        return [p for p in prefs if p.strength >= min_strength]

    async def query_relevant_preferences(
        self,
        user_id: str,
        context: Dict
    ) -> List[PreferenceNode]:
        """Get preferences relevant to a given context."""
        all_prefs = self.pref_repo.get_by_user(user_id)

        # Score preferences by relevance to context
        scored_prefs = []
        context_text = " ".join(str(v) for v in context.values()).lower()

        for pref in all_prefs:
            relevance = 0.0

            # Check category match
            if pref.category.lower() in context_text:
                relevance += 0.3

            # Check preference content match
            pref_words = pref.preference.lower().split()
            matches = sum(1 for w in pref_words if w in context_text)
            if pref_words:
                relevance += 0.5 * (matches / len(pref_words))

            # Weight by strength and confidence
            final_score = relevance * pref.strength * pref.confidence

            if final_score > 0.1:
                scored_prefs.append((pref, final_score))

        # Sort by score
        scored_prefs.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored_prefs[:10]]

    async def weaken_preference(self, pref_id: str, factor: float = 0.7):
        """Weaken a preference (when contradicted)."""
        prefs = self.pref_repo.get_by_user("")  # Get all to find by ID
        for pref in prefs:
            if pref.id == pref_id:
                new_strength = pref.strength * factor
                self.pref_repo.update_strength(pref_id, new_strength)
                break

    async def add_exception(
        self,
        pref_id: str,
        exception_context: Dict,
        correct_behavior: str
    ):
        """Add an exception to a preference."""
        prefs = self.pref_repo.get_by_user("")
        for pref in prefs:
            if pref.id == pref_id:
                pref.exceptions.append({
                    "context": exception_context,
                    "correct_behavior": correct_behavior,
                    "added_at": datetime.utcnow().isoformat()
                })
                self.pref_repo.save(pref)
                break

    async def predict_discomfort(
        self,
        user_id: str,
        proposed_action: Dict
    ) -> List[Dict]:
        """Predict if an action might violate user preferences."""
        violations = []
        prefs = self.pref_repo.get_by_user(user_id)

        action_text = " ".join(str(v) for v in proposed_action.values()).lower()

        for pref in prefs:
            # Check for potential violations
            pref_words = pref.preference.lower().split()

            # Look for negation patterns
            negations = ['not', 'never', 'avoid', 'don\'t', 'without']
            is_negative_pref = any(neg in pref.preference.lower() for neg in negations)

            if is_negative_pref:
                # Check if action contains what should be avoided
                for word in pref_words:
                    if word not in negations and word in action_text:
                        violations.append({
                            "preference_id": pref.id,
                            "preference": pref.preference,
                            "confidence": pref.confidence * pref.strength,
                            "type": "potential_violation"
                        })
                        break

        return violations
