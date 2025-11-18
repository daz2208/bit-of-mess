"""Intent classification using pattern matching and ML-like scoring."""

from typing import Dict, List, Tuple
import re
from collections import defaultdict


class IntentClassifier:
    """
    Classify user intents from natural language input.

    Uses pattern matching + learned patterns for classification.
    """

    def __init__(self):
        # Base intent patterns
        self.intent_patterns = {
            "schedule": [
                r"\b(schedule|book|arrange|set up|plan)\b.*\b(meeting|call|appointment|event)\b",
                r"\b(meeting|call|appointment)\b.*\b(at|on|for)\b",
                r"\bcalendar\b",
                r"\bfree\s+(time|slot)\b"
            ],
            "reminder": [
                r"\bremind\s+me\b",
                r"\bdon'?t\s+forget\b",
                r"\bremember\s+to\b",
                r"\bset\s+(a\s+)?reminder\b",
                r"\balert\s+me\b"
            ],
            "question": [
                r"^(what|where|when|why|how|who|which|is|are|do|does|can|could|would|should)\b",
                r"\?$",
                r"\btell\s+me\b",
                r"\bexplain\b"
            ],
            "task": [
                r"\b(create|make|build|write|send|update|delete|remove|add)\b",
                r"\bto-?do\b",
                r"\btask\b",
                r"\baction\s+item\b"
            ],
            "search": [
                r"\b(find|search|look\s+for|locate|get)\b",
                r"\bwhere\s+is\b",
                r"\bshow\s+me\b"
            ],
            "preference": [
                r"\bi\s+(prefer|like|want|need|hate|dislike)\b",
                r"\balways\b.*\b(do|use|want)\b",
                r"\bnever\b.*\b(do|use|want)\b",
                r"\bmy\s+(preference|style|way)\b"
            ],
            "feedback": [
                r"\b(good|great|bad|wrong|correct|right|perfect|terrible)\b",
                r"\bthat'?s\s+(not\s+)?(what|how)\b",
                r"\bactually\b",
                r"\binstead\b"
            ],
            "greeting": [
                r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings)\b",
                r"^(how\s+are\s+you|what'?s\s+up)\b"
            ],
            "help": [
                r"\bhelp\b",
                r"\bhow\s+do\s+i\b",
                r"\bcan\s+you\b.*\b(show|teach|explain)\b",
                r"\bi'?m\s+(stuck|confused|lost)\b"
            ],
            "confirmation": [
                r"^(yes|yeah|yep|sure|ok|okay|correct|right|exactly|confirmed)\b",
                r"^(no|nope|nah|wrong|incorrect|cancel)\b"
            ],
            "status": [
                r"\b(status|progress|update)\b",
                r"\bhow\s+(is|are)\b.*\bgoing\b",
                r"\bwhat'?s\s+(the\s+)?(status|progress)\b"
            ]
        }

        # Compile patterns
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.intent_patterns.items()
        }

        # Learned intent boosts per user
        self.user_intent_history: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Intent confidence thresholds
        self.thresholds = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.3
        }

    def classify(self, text: str, user_id: str = "") -> Dict:
        """
        Classify the intent of the input text.

        Returns:
            {
                "primary_intent": str,
                "confidence": float,
                "secondary_intents": [(intent, score), ...],
                "all_scores": {intent: score, ...}
            }
        """
        text = text.strip()
        scores = {}

        # Score each intent
        for intent, patterns in self.compiled_patterns.items():
            score = self._score_intent(text, patterns)

            # Apply user history boost
            if user_id and user_id in self.user_intent_history:
                history_boost = min(0.1, self.user_intent_history[user_id][intent] * 0.01)
                score += history_boost

            scores[intent] = min(1.0, score)

        # Find primary intent
        if not scores:
            return {
                "primary_intent": "unknown",
                "confidence": 0.0,
                "secondary_intents": [],
                "all_scores": {}
            }

        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_intents[0]

        # Get secondary intents (above threshold)
        secondary = [
            (intent, score) for intent, score in sorted_intents[1:5]
            if score > self.thresholds["low"]
        ]

        return {
            "primary_intent": primary[0],
            "confidence": primary[1],
            "secondary_intents": secondary,
            "all_scores": scores
        }

    def _score_intent(self, text: str, patterns: List) -> float:
        """Score how well text matches intent patterns."""
        matches = 0
        total_patterns = len(patterns)

        for pattern in patterns:
            if pattern.search(text):
                matches += 1

        if matches == 0:
            return 0.0

        # Base score from pattern matches
        base_score = matches / total_patterns

        # Boost for multiple matches
        if matches > 1:
            base_score = min(1.0, base_score * 1.3)

        return base_score

    def learn(self, user_id: str, text: str, actual_intent: str):
        """Learn from confirmed intent classification."""
        self.user_intent_history[user_id][actual_intent] += 1

    def get_intent_keywords(self, intent: str) -> List[str]:
        """Get keywords associated with an intent."""
        keywords = {
            "schedule": ["meeting", "calendar", "schedule", "book", "appointment"],
            "reminder": ["remind", "remember", "alert", "notify"],
            "question": ["what", "how", "why", "when", "where"],
            "task": ["create", "make", "build", "send", "update"],
            "search": ["find", "search", "look", "locate"],
            "preference": ["prefer", "like", "want", "always", "never"],
            "feedback": ["good", "bad", "wrong", "correct"],
            "greeting": ["hello", "hi", "hey"],
            "help": ["help", "how do", "explain"],
            "status": ["status", "progress", "update"]
        }
        return keywords.get(intent, [])

    def add_custom_intent(self, intent_name: str, patterns: List[str]):
        """Add a custom intent with patterns."""
        self.intent_patterns[intent_name] = patterns
        self.compiled_patterns[intent_name] = [
            re.compile(p, re.IGNORECASE) for p in patterns
        ]
