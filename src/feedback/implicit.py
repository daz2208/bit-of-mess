"""Extract learning signals from implicit user behavior."""

from typing import Optional, List, Dict
from datetime import datetime
import uuid

from ..models.feedback import FeedbackEvent, FeedbackType, LearningSignal
from ..models.learning import LearningUpdate, UpdateType
from ..models.base import Priority


class ImplicitSignalExtractor:
    """Extract learning signals from implicit user behavior."""

    def __init__(self):
        self.rejection_patterns: Dict[str, List] = {}  # user_id -> patterns
        self.adoption_rates: Dict[str, Dict] = {}  # user_id -> rates by type
        self.engagement_history: Dict[str, List] = {}  # user_id -> scores

    async def process(self, event: FeedbackEvent) -> Optional[LearningUpdate]:
        """Process implicit feedback signals."""

        if event.type == FeedbackType.SUGGESTION_IGNORED:
            return await self._handle_suggestion_ignored(event)
        elif event.type == FeedbackType.SUGGESTION_MODIFIED:
            return await self._handle_suggestion_modified(event)
        elif event.type == FeedbackType.SUGGESTION_ACCEPTED:
            return await self._handle_suggestion_accepted(event)

        return None

    async def _handle_suggestion_ignored(self, event: FeedbackEvent) -> LearningUpdate:
        """Handle when a suggestion is ignored."""

        user_id = event.user_id
        suggestion_type = event.data.get("suggestion_type", "unknown")
        content = event.data.get("content", "")

        # Track rejection pattern
        if user_id not in self.rejection_patterns:
            self.rejection_patterns[user_id] = []

        pattern = {
            "suggestion_type": suggestion_type,
            "content_features": self._extract_content_features(content),
            "context": event.context.to_dict() if event.context else {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.rejection_patterns[user_id].append(pattern)

        # Update adoption rate
        self._update_adoption_rate(user_id, suggestion_type, accepted=False)

        # Create learning update if pattern is strong enough
        rejection_count = sum(
            1 for p in self.rejection_patterns[user_id]
            if p["suggestion_type"] == suggestion_type
        )

        confidence = min(0.9, 0.5 + (rejection_count * 0.1))

        return LearningUpdate(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=UpdateType.PREFERENCE_REFINEMENT,
            confidence=confidence,
            priority=Priority.MEDIUM if rejection_count > 2 else Priority.LOW,
            update_data={
                "pattern_type": "suggestion_rejection",
                "suggestion_type": suggestion_type,
                "rejection_count": rejection_count,
                "content_features": pattern["content_features"],
                "inferred_preference": f"Avoid {suggestion_type} suggestions"
            },
            affected_memories=["preference"],
            source="implicit"
        )

    async def _handle_suggestion_modified(self, event: FeedbackEvent) -> LearningUpdate:
        """Handle when a suggestion is modified - partial acceptance."""

        user_id = event.user_id
        original = event.data.get("original", "")
        modified = event.data.get("modified", "")
        suggestion_type = event.data.get("suggestion_type", "unknown")

        # Analyze modifications
        modifications = self._analyze_modifications(original, modified)

        return LearningUpdate(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=UpdateType.PREFERENCE_REFINEMENT,
            confidence=0.75,  # Good confidence for modified suggestions
            priority=Priority.MEDIUM,
            update_data={
                "pattern_type": "suggestion_modification",
                "suggestion_type": suggestion_type,
                "kept_aspects": modifications["kept"],
                "rejected_aspects": modifications["rejected"],
                "added_aspects": modifications["added"],
                "modification_pattern": modifications["pattern"]
            },
            affected_memories=["preference", "procedural"],
            source="implicit"
        )

    async def _handle_suggestion_accepted(self, event: FeedbackEvent) -> LearningUpdate:
        """Handle when a suggestion is fully accepted."""

        user_id = event.user_id
        suggestion_type = event.data.get("suggestion_type", "unknown")
        content = event.data.get("content", "")

        # Update adoption rate
        self._update_adoption_rate(user_id, suggestion_type, accepted=True)

        return LearningUpdate(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=UpdateType.PREFERENCE_REFINEMENT,
            confidence=0.7,
            priority=Priority.LOW,  # Positive signals are lower priority
            update_data={
                "pattern_type": "suggestion_acceptance",
                "suggestion_type": suggestion_type,
                "content_features": self._extract_content_features(content),
                "inferred_preference": f"Continue {suggestion_type} suggestions"
            },
            affected_memories=["preference"],
            source="implicit"
        )

    def _extract_content_features(self, content: str) -> Dict:
        """Extract features from content for pattern analysis."""
        words = content.lower().split()

        return {
            "length": len(content),
            "word_count": len(words),
            "has_numbers": any(c.isdigit() for c in content),
            "has_time": any(w in words for w in ["am", "pm", "morning", "afternoon", "evening"]),
            "has_date": any(w in words for w in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow"]),
            "formality": self._estimate_formality(content),
            "key_terms": list(set(w for w in words if len(w) > 4))[:10]
        }

    def _estimate_formality(self, content: str) -> float:
        """Estimate formality level of content (0=casual, 1=formal)."""
        formal_indicators = ["please", "kindly", "would", "shall", "regarding", "therefore"]
        casual_indicators = ["hey", "yeah", "gonna", "wanna", "btw", "asap"]

        content_lower = content.lower()
        formal_count = sum(1 for w in formal_indicators if w in content_lower)
        casual_count = sum(1 for w in casual_indicators if w in content_lower)

        if formal_count + casual_count == 0:
            return 0.5

        return formal_count / (formal_count + casual_count)

    def _analyze_modifications(self, original: str, modified: str) -> Dict:
        """Analyze what was changed between original and modified."""
        orig_words = set(original.lower().split())
        mod_words = set(modified.lower().split())

        kept = orig_words & mod_words
        rejected = orig_words - mod_words
        added = mod_words - orig_words

        # Determine modification pattern
        if len(rejected) > len(kept):
            pattern = "major_change"
        elif len(added) > len(kept):
            pattern = "expansion"
        elif len(rejected) > 0:
            pattern = "refinement"
        else:
            pattern = "minor_addition"

        return {
            "kept": list(kept),
            "rejected": list(rejected),
            "added": list(added),
            "pattern": pattern
        }

    def _update_adoption_rate(self, user_id: str, suggestion_type: str, accepted: bool):
        """Update adoption rate tracking."""
        if user_id not in self.adoption_rates:
            self.adoption_rates[user_id] = {}

        if suggestion_type not in self.adoption_rates[user_id]:
            self.adoption_rates[user_id][suggestion_type] = {
                "accepted": 0,
                "total": 0
            }

        self.adoption_rates[user_id][suggestion_type]["total"] += 1
        if accepted:
            self.adoption_rates[user_id][suggestion_type]["accepted"] += 1

    def get_adoption_rate(self, user_id: str, suggestion_type: str) -> float:
        """Get adoption rate for a suggestion type."""
        if user_id not in self.adoption_rates:
            return 0.5  # Default

        if suggestion_type not in self.adoption_rates[user_id]:
            return 0.5

        rates = self.adoption_rates[user_id][suggestion_type]
        if rates["total"] == 0:
            return 0.5

        return rates["accepted"] / rates["total"]
