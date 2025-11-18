"""Analytics engine for insights and learning progress."""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from collections import Counter
import statistics


class AnalyticsEngine:
    """
    Generate analytics and insights about user patterns and agent learning.

    Tracks:
    - Learning velocity
    - Accuracy over time
    - Pattern discovery
    - User engagement
    """

    def __init__(self):
        self.interaction_log: Dict[str, List] = {}
        self.prediction_log: Dict[str, List] = {}
        self.learning_events: Dict[str, List] = {}

    def log_interaction(
        self,
        user_id: str,
        interaction_type: str,
        intent: str,
        success: bool,
        confidence: float,
        metadata: Dict = None
    ):
        """Log an interaction for analytics."""
        if user_id not in self.interaction_log:
            self.interaction_log[user_id] = []

        self.interaction_log[user_id].append({
            "timestamp": datetime.utcnow(),
            "type": interaction_type,
            "intent": intent,
            "success": success,
            "confidence": confidence,
            "metadata": metadata or {}
        })

    def log_prediction(
        self,
        user_id: str,
        prediction_type: str,
        predicted: any,
        actual: any,
        confidence: float
    ):
        """Log a prediction for calibration tracking."""
        if user_id not in self.prediction_log:
            self.prediction_log[user_id] = []

        correct = predicted == actual

        self.prediction_log[user_id].append({
            "timestamp": datetime.utcnow(),
            "type": prediction_type,
            "predicted": predicted,
            "actual": actual,
            "confidence": confidence,
            "correct": correct
        })

    def log_learning_event(
        self,
        user_id: str,
        event_type: str,
        source: str,
        impact: float
    ):
        """Log a learning event."""
        if user_id not in self.learning_events:
            self.learning_events[user_id] = []

        self.learning_events[user_id].append({
            "timestamp": datetime.utcnow(),
            "type": event_type,
            "source": source,
            "impact": impact
        })

    def get_learning_velocity(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict:
        """Calculate learning velocity over time."""
        if user_id not in self.learning_events:
            return {"velocity": 0, "trend": "stable", "events": 0}

        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [e for e in self.learning_events[user_id] if e["timestamp"] > cutoff]

        if not recent:
            return {"velocity": 0, "trend": "stable", "events": 0}

        # Events per day
        events_per_day = len(recent) / days

        # Average impact
        avg_impact = statistics.mean([e["impact"] for e in recent])

        # Velocity = rate * impact
        velocity = events_per_day * avg_impact

        # Calculate trend
        if len(recent) >= 4:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            first_impact = statistics.mean([e["impact"] for e in first_half])
            second_impact = statistics.mean([e["impact"] for e in second_half])

            if second_impact > first_impact * 1.2:
                trend = "accelerating"
            elif second_impact < first_impact * 0.8:
                trend = "decelerating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "velocity": velocity,
            "trend": trend,
            "events": len(recent),
            "avg_impact": avg_impact
        }

    def get_accuracy_metrics(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict:
        """Get accuracy metrics over time."""
        if user_id not in self.interaction_log:
            return {"accuracy": 0, "trend": "stable"}

        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [i for i in self.interaction_log[user_id] if i["timestamp"] > cutoff]

        if not recent:
            return {"accuracy": 0, "trend": "stable"}

        successes = sum(1 for i in recent if i["success"])
        accuracy = successes / len(recent)

        # Trend analysis
        if len(recent) >= 6:
            first_third = recent[:len(recent)//3]
            last_third = recent[-len(recent)//3:]

            first_acc = sum(1 for i in first_third if i["success"]) / len(first_third)
            last_acc = sum(1 for i in last_third if i["success"]) / len(last_third)

            if last_acc > first_acc + 0.1:
                trend = "improving"
            elif last_acc < first_acc - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "accuracy": accuracy,
            "trend": trend,
            "total_interactions": len(recent),
            "successes": successes
        }

    def get_confidence_calibration(self, user_id: str) -> Dict:
        """Check if confidence scores are well-calibrated."""
        if user_id not in self.prediction_log:
            return {"calibration_error": 0, "overconfident": False}

        predictions = self.prediction_log[user_id]
        if len(predictions) < 10:
            return {"calibration_error": 0, "overconfident": False, "insufficient_data": True}

        # Group by confidence buckets
        buckets = {
            "0.0-0.2": {"predictions": [], "correct": 0},
            "0.2-0.4": {"predictions": [], "correct": 0},
            "0.4-0.6": {"predictions": [], "correct": 0},
            "0.6-0.8": {"predictions": [], "correct": 0},
            "0.8-1.0": {"predictions": [], "correct": 0}
        }

        for pred in predictions:
            conf = pred["confidence"]
            if conf < 0.2:
                bucket = "0.0-0.2"
            elif conf < 0.4:
                bucket = "0.2-0.4"
            elif conf < 0.6:
                bucket = "0.4-0.6"
            elif conf < 0.8:
                bucket = "0.6-0.8"
            else:
                bucket = "0.8-1.0"

            buckets[bucket]["predictions"].append(pred)
            if pred["correct"]:
                buckets[bucket]["correct"] += 1

        # Calculate calibration error
        calibration_errors = []
        for bucket_name, bucket_data in buckets.items():
            if not bucket_data["predictions"]:
                continue

            # Expected accuracy (middle of bucket)
            expected = float(bucket_name.split("-")[0]) + 0.1
            actual = bucket_data["correct"] / len(bucket_data["predictions"])

            calibration_errors.append(abs(expected - actual))

        avg_error = statistics.mean(calibration_errors) if calibration_errors else 0

        # Check if overconfident (high confidence but low accuracy)
        high_conf = buckets["0.8-1.0"]
        overconfident = False
        if high_conf["predictions"]:
            high_conf_accuracy = high_conf["correct"] / len(high_conf["predictions"])
            overconfident = high_conf_accuracy < 0.7

        return {
            "calibration_error": avg_error,
            "overconfident": overconfident,
            "buckets": {k: len(v["predictions"]) for k, v in buckets.items()}
        }

    def get_pattern_insights(self, user_id: str) -> List[Dict]:
        """Discover patterns in user behavior."""
        insights = []

        if user_id not in self.interaction_log:
            return insights

        interactions = self.interaction_log[user_id]

        # Time patterns
        hours = [i["timestamp"].hour for i in interactions]
        if hours:
            hour_counts = Counter(hours)
            peak_hour = hour_counts.most_common(1)[0]
            insights.append({
                "type": "time_pattern",
                "insight": f"Peak activity at {peak_hour[0]}:00",
                "confidence": min(0.9, peak_hour[1] / len(hours))
            })

        # Intent patterns
        intents = [i["intent"] for i in interactions]
        if intents:
            intent_counts = Counter(intents)
            top_intents = intent_counts.most_common(3)
            insights.append({
                "type": "intent_pattern",
                "insight": f"Most common intents: {', '.join([i[0] for i in top_intents])}",
                "confidence": 0.9
            })

        # Success patterns
        by_intent_success = {}
        for i in interactions:
            intent = i["intent"]
            if intent not in by_intent_success:
                by_intent_success[intent] = {"success": 0, "total": 0}
            by_intent_success[intent]["total"] += 1
            if i["success"]:
                by_intent_success[intent]["success"] += 1

        # Find intents with low success
        for intent, data in by_intent_success.items():
            if data["total"] >= 5:
                rate = data["success"] / data["total"]
                if rate < 0.5:
                    insights.append({
                        "type": "improvement_needed",
                        "insight": f"Low success rate ({rate:.0%}) for '{intent}' - needs attention",
                        "confidence": 0.85
                    })

        return insights

    def get_dashboard(self, user_id: str) -> Dict:
        """Get complete analytics dashboard."""
        return {
            "learning": self.get_learning_velocity(user_id),
            "accuracy": self.get_accuracy_metrics(user_id),
            "calibration": self.get_confidence_calibration(user_id),
            "insights": self.get_pattern_insights(user_id),
            "totals": {
                "interactions": len(self.interaction_log.get(user_id, [])),
                "predictions": len(self.prediction_log.get(user_id, [])),
                "learning_events": len(self.learning_events.get(user_id, []))
            }
        }
