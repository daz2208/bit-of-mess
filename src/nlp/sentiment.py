"""Sentiment and emotion analysis."""

from typing import Dict, List, Tuple
import re
from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    polarity: float  # -1 (negative) to 1 (positive)
    intensity: float  # 0 to 1
    emotions: Dict[str, float]  # emotion -> confidence
    indicators: List[str]  # words/phrases that influenced the result


class SentimentAnalyzer:
    """
    Analyze sentiment and emotion from text.

    Detects:
    - Overall polarity (positive/negative)
    - Intensity
    - Specific emotions (joy, anger, frustration, etc.)
    """

    def __init__(self):
        # Positive words with intensity scores
        self.positive_words = {
            # High intensity
            "amazing": 0.9, "excellent": 0.9, "fantastic": 0.9, "perfect": 0.9,
            "love": 0.85, "wonderful": 0.85, "brilliant": 0.85, "outstanding": 0.85,
            # Medium intensity
            "great": 0.7, "good": 0.6, "nice": 0.5, "helpful": 0.6, "thanks": 0.5,
            "appreciate": 0.65, "happy": 0.7, "pleased": 0.65, "glad": 0.6,
            # Low intensity
            "okay": 0.3, "fine": 0.3, "alright": 0.3, "decent": 0.4
        }

        # Negative words with intensity scores
        self.negative_words = {
            # High intensity
            "terrible": 0.9, "awful": 0.9, "horrible": 0.9, "hate": 0.85,
            "worst": 0.9, "useless": 0.8, "stupid": 0.8, "disaster": 0.85,
            # Medium intensity
            "bad": 0.6, "wrong": 0.6, "annoying": 0.65, "frustrating": 0.7,
            "disappointed": 0.65, "upset": 0.7, "angry": 0.75, "confused": 0.5,
            # Low intensity
            "not great": 0.4, "meh": 0.3, "okay": 0.2, "issue": 0.4
        }

        # Emotion patterns
        self.emotion_patterns = {
            "joy": [
                r"\b(happy|glad|pleased|delighted|excited|thrilled)\b",
                r"\b(love|enjoy|wonderful|amazing)\b",
                r"(!{2,}|\b(yay|woohoo|yes!)\b)"
            ],
            "anger": [
                r"\b(angry|furious|mad|outraged|livid)\b",
                r"\b(hate|despise|infuriating)\b",
                r"(!{3,}|[A-Z]{3,})"
            ],
            "frustration": [
                r"\b(frustrated|annoyed|irritated|stuck)\b",
                r"\b(why (won't|doesn't|can't|isn't))\b",
                r"\b(again|still|keeps?)\b.*\b(not|wrong|fail)\b"
            ],
            "confusion": [
                r"\b(confused|unclear|don't understand|lost)\b",
                r"\b(what|how|why)\b.*\?",
                r"\b(doesn't make sense|no idea)\b"
            ],
            "sadness": [
                r"\b(sad|unhappy|disappointed|upset|down)\b",
                r"\b(miss|lost|gone|unfortunately)\b"
            ],
            "anxiety": [
                r"\b(worried|anxious|nervous|concerned|stressed)\b",
                r"\b(afraid|scared|fear)\b",
                r"\b(urgent|asap|deadline|running out)\b"
            ],
            "gratitude": [
                r"\b(thank|thanks|appreciate|grateful)\b",
                r"\b(helpful|saved|lifesaver)\b"
            ],
            "surprise": [
                r"\b(surprised|shocked|amazed|unexpected)\b",
                r"\b(wow|whoa|oh my)\b",
                r"(!{2,}|\?{2,})"
            ]
        }

        # Intensifiers
        self.intensifiers = {
            "very": 1.3, "really": 1.3, "extremely": 1.5, "absolutely": 1.5,
            "incredibly": 1.4, "super": 1.3, "totally": 1.3, "completely": 1.4,
            "quite": 1.1, "pretty": 1.1, "somewhat": 0.8, "slightly": 0.7,
            "a bit": 0.7, "a little": 0.7, "kind of": 0.8, "sort of": 0.8
        }

        # Negators
        self.negators = ["not", "no", "never", "neither", "nobody", "nothing",
                        "nowhere", "hardly", "barely", "doesn't", "don't",
                        "didn't", "won't", "wouldn't", "couldn't", "shouldn't"]

    def analyze(self, text: str) -> SentimentResult:
        """Perform full sentiment analysis on text."""
        text_lower = text.lower()
        words = text_lower.split()

        # Calculate polarity
        polarity, intensity, indicators = self._calculate_polarity(text_lower, words)

        # Detect emotions
        emotions = self._detect_emotions(text)

        return SentimentResult(
            polarity=polarity,
            intensity=intensity,
            emotions=emotions,
            indicators=indicators
        )

    def _calculate_polarity(self, text: str, words: List[str]) -> Tuple[float, float, List[str]]:
        """Calculate overall sentiment polarity and intensity."""
        positive_score = 0.0
        negative_score = 0.0
        indicators = []

        i = 0
        while i < len(words):
            word = words[i]

            # Check for negation
            is_negated = False
            if i > 0 and words[i-1] in self.negators:
                is_negated = True
            if i > 1 and words[i-2] in self.negators:
                is_negated = True

            # Check for intensifier
            multiplier = 1.0
            for j in range(max(0, i-2), i):
                for intensifier, mult in self.intensifiers.items():
                    if intensifier in words[j]:
                        multiplier = mult
                        break

            # Score positive words
            if word in self.positive_words:
                score = self.positive_words[word] * multiplier
                if is_negated:
                    negative_score += score
                    indicators.append(f"not {word}")
                else:
                    positive_score += score
                    indicators.append(word)

            # Score negative words
            elif word in self.negative_words:
                score = self.negative_words[word] * multiplier
                if is_negated:
                    positive_score += score * 0.5  # Negated negative is weakly positive
                    indicators.append(f"not {word}")
                else:
                    negative_score += score
                    indicators.append(word)

            i += 1

        # Calculate final scores
        total = positive_score + negative_score
        if total == 0:
            return 0.0, 0.0, indicators

        polarity = (positive_score - negative_score) / max(1, total)
        polarity = max(-1.0, min(1.0, polarity))

        intensity = min(1.0, total / 3)  # Normalize

        return polarity, intensity, indicators

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect specific emotions in text."""
        emotions = {}

        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0
            matches = 0

            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                if found:
                    matches += len(found)

            if matches > 0:
                # Score based on number of matches (with diminishing returns)
                score = min(1.0, 0.4 + (matches * 0.2))
                emotions[emotion] = score

        return emotions

    def get_dominant_emotion(self, text: str) -> Tuple[str, float]:
        """Get the single most dominant emotion."""
        result = self.analyze(text)

        if not result.emotions:
            # Default based on polarity
            if result.polarity > 0.3:
                return ("positive", result.polarity)
            elif result.polarity < -0.3:
                return ("negative", abs(result.polarity))
            else:
                return ("neutral", 0.5)

        dominant = max(result.emotions.items(), key=lambda x: x[1])
        return dominant

    def is_frustrated(self, text: str) -> bool:
        """Quick check if user seems frustrated."""
        result = self.analyze(text)
        return (
            result.emotions.get("frustration", 0) > 0.5 or
            result.emotions.get("anger", 0) > 0.5 or
            (result.polarity < -0.3 and result.intensity > 0.5)
        )

    def needs_empathy(self, text: str) -> bool:
        """Check if response should include empathy."""
        result = self.analyze(text)
        emotional_intensity = sum(result.emotions.values())
        return (
            result.emotions.get("sadness", 0) > 0.4 or
            result.emotions.get("anxiety", 0) > 0.5 or
            result.emotions.get("frustration", 0) > 0.6 or
            emotional_intensity > 1.0
        )
