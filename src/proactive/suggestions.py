"""Proactive suggestion engine - anticipate user needs."""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics


@dataclass
class Suggestion:
    """A proactive suggestion."""
    id: str
    type: str
    content: str
    confidence: float
    reasoning: str
    context_triggers: List[str]
    priority: int  # 1-10


class ProactiveSuggestionEngine:
    """
    Generate proactive suggestions based on patterns and context.

    Analyzes:
    - Time patterns
    - Behavioral sequences
    - Unfinished tasks
    - Upcoming needs
    """

    def __init__(self):
        # Track user patterns
        self.time_patterns: Dict[str, Dict] = {}
        self.sequence_patterns: Dict[str, List] = {}
        self.task_history: Dict[str, List] = {}

    def analyze_and_suggest(
        self,
        user_id: str,
        current_context: Dict,
        preferences: List[Dict],
        recent_interactions: List[Dict]
    ) -> List[Suggestion]:
        """Generate suggestions based on analysis."""
        suggestions = []

        # Time-based suggestions
        time_suggestions = self._suggest_from_time_patterns(
            user_id, current_context
        )
        suggestions.extend(time_suggestions)

        # Sequence-based suggestions
        seq_suggestions = self._suggest_from_sequences(
            user_id, recent_interactions
        )
        suggestions.extend(seq_suggestions)

        # Preference-based suggestions
        pref_suggestions = self._suggest_from_preferences(
            user_id, current_context, preferences
        )
        suggestions.extend(pref_suggestions)

        # Unfinished task reminders
        task_suggestions = self._suggest_unfinished_tasks(user_id)
        suggestions.extend(task_suggestions)

        # Sort by priority and confidence
        suggestions.sort(key=lambda s: (s.priority, s.confidence), reverse=True)

        return suggestions[:5]  # Top 5

    def learn_time_pattern(
        self,
        user_id: str,
        action: str,
        timestamp: datetime
    ):
        """Learn time patterns from user actions."""
        if user_id not in self.time_patterns:
            self.time_patterns[user_id] = {}

        if action not in self.time_patterns[user_id]:
            self.time_patterns[user_id][action] = {
                "hours": [],
                "weekdays": [],
                "count": 0
            }

        pattern = self.time_patterns[user_id][action]
        pattern["hours"].append(timestamp.hour)
        pattern["weekdays"].append(timestamp.weekday())
        pattern["count"] += 1

        # Keep last 100
        if len(pattern["hours"]) > 100:
            pattern["hours"] = pattern["hours"][-100:]
            pattern["weekdays"] = pattern["weekdays"][-100:]

    def learn_sequence(
        self,
        user_id: str,
        actions: List[str]
    ):
        """Learn action sequences."""
        if user_id not in self.sequence_patterns:
            self.sequence_patterns[user_id] = []

        self.sequence_patterns[user_id].append(actions)

        # Keep last 50 sequences
        if len(self.sequence_patterns[user_id]) > 50:
            self.sequence_patterns[user_id] = self.sequence_patterns[user_id][-50:]

    def track_task(
        self,
        user_id: str,
        task: str,
        status: str  # "started", "completed", "abandoned"
    ):
        """Track task for unfinished task suggestions."""
        if user_id not in self.task_history:
            self.task_history[user_id] = []

        self.task_history[user_id].append({
            "task": task,
            "status": status,
            "timestamp": datetime.utcnow()
        })

    def _suggest_from_time_patterns(
        self,
        user_id: str,
        context: Dict
    ) -> List[Suggestion]:
        """Generate suggestions based on time patterns."""
        suggestions = []

        if user_id not in self.time_patterns:
            return suggestions

        current_hour = datetime.utcnow().hour
        current_weekday = datetime.utcnow().weekday()

        for action, pattern in self.time_patterns[user_id].items():
            if pattern["count"] < 3:
                continue

            # Calculate if this is a typical time for this action
            if pattern["hours"]:
                avg_hour = statistics.mean(pattern["hours"])
                std_hour = statistics.stdev(pattern["hours"]) if len(pattern["hours"]) > 1 else 2

                # Within 1 std of typical time
                if abs(current_hour - avg_hour) <= std_hour:
                    confidence = 1 - (abs(current_hour - avg_hour) / (std_hour + 1))

                    suggestions.append(Suggestion(
                        id=f"time_{action}_{current_hour}",
                        type="time_pattern",
                        content=f"Around this time you usually: {action}",
                        confidence=min(0.9, confidence),
                        reasoning=f"Based on {pattern['count']} similar occurrences",
                        context_triggers=["time_of_day"],
                        priority=6
                    ))

        return suggestions

    def _suggest_from_sequences(
        self,
        user_id: str,
        recent_interactions: List[Dict]
    ) -> List[Suggestion]:
        """Suggest next action based on sequences."""
        suggestions = []

        if user_id not in self.sequence_patterns:
            return suggestions

        if not recent_interactions:
            return suggestions

        # Get recent action sequence
        recent_actions = [i.get("intent", "") for i in recent_interactions[-3:]]

        # Find matching patterns
        matches = []
        for seq in self.sequence_patterns[user_id]:
            if len(seq) > len(recent_actions):
                # Check if recent matches start of sequence
                if seq[:len(recent_actions)] == recent_actions:
                    next_action = seq[len(recent_actions)]
                    matches.append(next_action)

        if matches:
            # Most common next action
            from collections import Counter
            most_common = Counter(matches).most_common(1)[0]
            next_action, count = most_common

            confidence = min(0.85, count / len(self.sequence_patterns[user_id]))

            suggestions.append(Suggestion(
                id=f"seq_{next_action}",
                type="sequence_prediction",
                content=f"Based on your pattern, you might want to: {next_action}",
                confidence=confidence,
                reasoning=f"This follows your typical sequence {count} times",
                context_triggers=["action_sequence"],
                priority=7
            ))

        return suggestions

    def _suggest_from_preferences(
        self,
        user_id: str,
        context: Dict,
        preferences: List[Dict]
    ) -> List[Suggestion]:
        """Generate suggestions from preferences."""
        suggestions = []

        time_of_day = context.get("time_of_day", "")

        for pref in preferences:
            pref_text = pref.get("preference", "").lower()
            strength = pref.get("strength", 0.5)

            # Morning preferences
            if time_of_day == "morning" and "morning" in pref_text:
                suggestions.append(Suggestion(
                    id=f"pref_morning_{hash(pref_text)}",
                    type="preference_reminder",
                    content=f"Reminder: {pref.get('preference', '')}",
                    confidence=strength,
                    reasoning="Matches your morning preferences",
                    context_triggers=["morning"],
                    priority=5
                ))

            # Focus time preferences
            if "focus" in pref_text or "quiet" in pref_text:
                suggestions.append(Suggestion(
                    id=f"pref_focus_{hash(pref_text)}",
                    type="preference_reminder",
                    content=f"Consider: {pref.get('preference', '')}",
                    confidence=strength * 0.8,
                    reasoning="You have focus time preferences",
                    context_triggers=["focus_related"],
                    priority=4
                ))

        return suggestions

    def _suggest_unfinished_tasks(self, user_id: str) -> List[Suggestion]:
        """Remind about unfinished tasks."""
        suggestions = []

        if user_id not in self.task_history:
            return suggestions

        now = datetime.utcnow()

        # Find started but not completed tasks
        unfinished = []
        for task_entry in self.task_history[user_id]:
            if task_entry["status"] == "started":
                # Check if not completed later
                task_name = task_entry["task"]
                completed = any(
                    t["task"] == task_name and t["status"] == "completed"
                    for t in self.task_history[user_id]
                    if t["timestamp"] > task_entry["timestamp"]
                )
                if not completed:
                    age = (now - task_entry["timestamp"]).total_seconds() / 3600
                    unfinished.append((task_name, age))

        # Sort by age (oldest first)
        unfinished.sort(key=lambda x: x[1], reverse=True)

        for task_name, age_hours in unfinished[:3]:
            priority = 8 if age_hours > 24 else 5

            suggestions.append(Suggestion(
                id=f"unfinished_{hash(task_name)}",
                type="unfinished_task",
                content=f"You started but didn't complete: {task_name}",
                confidence=min(0.9, 0.5 + (age_hours / 48)),
                reasoning=f"Started {age_hours:.1f} hours ago",
                context_triggers=["task_reminder"],
                priority=priority
            ))

        return suggestions

    def get_anticipations(
        self,
        user_id: str,
        context: Dict
    ) -> List[Dict]:
        """Get anticipated needs based on patterns."""
        anticipations = []

        # Check what usually happens at this time
        if user_id in self.time_patterns:
            current_hour = datetime.utcnow().hour

            for action, pattern in self.time_patterns[user_id].items():
                if pattern["hours"] and len(pattern["hours"]) >= 5:
                    avg_hour = statistics.mean(pattern["hours"])
                    # If action typically happens in next hour
                    if 0 <= avg_hour - current_hour <= 1:
                        anticipations.append({
                            "action": action,
                            "expected_in": "next hour",
                            "confidence": min(0.8, pattern["count"] / 10)
                        })

        return anticipations
