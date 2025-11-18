"""Analyze behavioral patterns over time."""

from typing import Optional, List, Dict
from datetime import datetime, timedelta
from collections import deque
import uuid
import statistics

from ..models.feedback import FeedbackEvent, InteractionEvent
from ..models.learning import LearningUpdate, UpdateType, PatternShift
from ..models.base import Priority


class BehavioralPatternAnalyzer:
    """Analyze behavioral patterns to detect shifts and infer preferences."""

    def __init__(self):
        # Pattern buffers per user
        self.short_term: Dict[str, deque] = {}   # Last ~20 interactions
        self.medium_term: Dict[str, deque] = {}  # Last ~100 interactions
        self.long_term: Dict[str, List] = {}     # Significant patterns

    async def process(self, event: InteractionEvent) -> Optional[LearningUpdate]:
        """Process an interaction and detect patterns."""

        user_id = event.user_id

        # Initialize buffers if needed
        if user_id not in self.short_term:
            self.short_term[user_id] = deque(maxlen=20)
            self.medium_term[user_id] = deque(maxlen=100)
            self.long_term[user_id] = []

        # Add to buffers
        interaction_data = {
            "id": event.id,
            "type": event.event_type,
            "timestamp": event.created_at,
            "engagement": event.engagement_score,
            "duration": event.duration_seconds,
            "context": event.context.to_dict() if event.context else {},
            "content_length": len(event.content)
        }

        self.short_term[user_id].append(interaction_data)
        self.medium_term[user_id].append(interaction_data)

        # Only analyze if we have enough data
        if len(self.short_term[user_id]) < 5:
            return None

        # Detect pattern shifts
        shifts = await self._detect_pattern_shifts(user_id)

        # Infer preferences
        preferences = await self._infer_preferences(user_id)

        if not shifts and not preferences:
            return None

        confidence = self._calculate_confidence(user_id, shifts, preferences)

        return LearningUpdate(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=UpdateType.BEHAVIORAL_PATTERN,
            confidence=confidence,
            priority=Priority.MEDIUM if confidence > 0.7 else Priority.LOW,
            update_data={
                "pattern_shifts": [s.to_dict() for s in shifts],
                "inferred_preferences": preferences,
                "analysis_window": {
                    "short_term_count": len(self.short_term[user_id]),
                    "medium_term_count": len(self.medium_term[user_id])
                }
            },
            affected_memories=["preference", "procedural"],
            source="behavioral"
        )

    async def _detect_pattern_shifts(self, user_id: str) -> List[PatternShift]:
        """Detect significant shifts between short and medium term patterns."""
        shifts = []

        short = list(self.short_term[user_id])
        medium = list(self.medium_term[user_id])

        if len(short) < 5 or len(medium) < 20:
            return shifts

        # Analyze engagement shift
        short_engagement = statistics.mean([i["engagement"] for i in short])
        medium_engagement = statistics.mean([i["engagement"] for i in medium])

        if abs(short_engagement - medium_engagement) > 0.2:
            shifts.append(PatternShift(
                pattern_type="engagement",
                old_pattern={"mean": medium_engagement},
                new_pattern={"mean": short_engagement},
                confidence=min(0.9, abs(short_engagement - medium_engagement) * 2)
            ))

        # Analyze time-of-day shift
        short_hours = [i["context"].get("time_of_day", "") for i in short if i.get("context")]
        medium_hours = [i["context"].get("time_of_day", "") for i in medium if i.get("context")]

        if short_hours and medium_hours:
            short_dominant = max(set(short_hours), key=short_hours.count)
            medium_dominant = max(set(medium_hours), key=medium_hours.count)

            if short_dominant != medium_dominant:
                shifts.append(PatternShift(
                    pattern_type="time_of_day",
                    old_pattern={"dominant": medium_dominant},
                    new_pattern={"dominant": short_dominant},
                    confidence=0.7
                ))

        # Analyze interaction duration shift
        short_duration = statistics.mean([i["duration"] for i in short])
        medium_duration = statistics.mean([i["duration"] for i in medium])

        if medium_duration > 0 and abs(short_duration - medium_duration) / medium_duration > 0.3:
            shifts.append(PatternShift(
                pattern_type="interaction_duration",
                old_pattern={"mean": medium_duration},
                new_pattern={"mean": short_duration},
                confidence=0.6
            ))

        return shifts

    async def _infer_preferences(self, user_id: str) -> List[Dict]:
        """Infer preferences from consistent behavioral patterns."""
        preferences = []

        interactions = list(self.medium_term[user_id])
        if len(interactions) < 10:
            return preferences

        # Time preference
        times = [i["context"].get("time_of_day", "") for i in interactions if i.get("context")]
        if times:
            time_counts = {}
            for t in times:
                time_counts[t] = time_counts.get(t, 0) + 1

            dominant_time = max(time_counts, key=time_counts.get)
            time_ratio = time_counts[dominant_time] / len(times)

            if time_ratio > 0.6:
                preferences.append({
                    "category": "temporal",
                    "preference": f"Prefers {dominant_time} interactions",
                    "confidence": time_ratio,
                    "evidence_count": time_counts[dominant_time]
                })

        # Engagement pattern
        high_engagement = [i for i in interactions if i["engagement"] > 0.7]
        if len(high_engagement) > 5:
            # Find what's common in high engagement interactions
            common_types = {}
            for i in high_engagement:
                t = i["type"]
                common_types[t] = common_types.get(t, 0) + 1

            if common_types:
                top_type = max(common_types, key=common_types.get)
                preferences.append({
                    "category": "content",
                    "preference": f"High engagement with {top_type}",
                    "confidence": common_types[top_type] / len(high_engagement),
                    "evidence_count": common_types[top_type]
                })

        # Pace preference
        durations = [i["duration"] for i in interactions]
        avg_duration = statistics.mean(durations)
        std_duration = statistics.stdev(durations) if len(durations) > 1 else 0

        if std_duration < avg_duration * 0.3:  # Consistent duration
            pace = "quick" if avg_duration < 30 else "moderate" if avg_duration < 120 else "detailed"
            preferences.append({
                "category": "pace",
                "preference": f"Prefers {pace} interactions",
                "confidence": 1 - (std_duration / avg_duration) if avg_duration > 0 else 0.5,
                "evidence_count": len(durations)
            })

        return preferences

    def _calculate_confidence(
        self,
        user_id: str,
        shifts: List[PatternShift],
        preferences: List[Dict]
    ) -> float:
        """Calculate overall confidence in the pattern analysis."""

        if not shifts and not preferences:
            return 0.0

        confidences = []

        for shift in shifts:
            confidences.append(shift.confidence)

        for pref in preferences:
            confidences.append(pref.get("confidence", 0.5))

        if not confidences:
            return 0.5

        # Weight by amount of data
        data_factor = min(1.0, len(self.medium_term[user_id]) / 50)

        return statistics.mean(confidences) * data_factor
