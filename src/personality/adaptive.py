"""Adaptive personality system that learns user communication style."""

from typing import Dict, List
from datetime import datetime

from ..models.action import PersonalityProfile
from ..models.feedback import InteractionEvent
from ..storage.repositories import PersonalityRepository


class AdaptivePersonality:
    """
    Adaptive personality core that adjusts communication style based on user.

    Tracks and adapts:
    - Assertiveness
    - Warmth
    - Detail orientation
    - Formality
    - Humor level
    - Pace
    - Directness
    """

    def __init__(self, personality_repo: PersonalityRepository):
        self.personality_repo = personality_repo
        self.learning_rate = 0.1
        self.interaction_history: Dict[str, List] = {}

    def get_profile(self, user_id: str) -> PersonalityProfile:
        """Get personality profile for a user."""
        return self.personality_repo.get(user_id)

    async def adapt_from_interaction(
        self,
        user_id: str,
        interaction: InteractionEvent
    ):
        """Adapt personality based on an interaction."""

        # Get current profile
        profile = self.get_profile(user_id)

        # Analyze interaction signals
        signals = self._analyze_interaction_signals(interaction)

        # Gradually adapt traits
        for trait, target_value in signals.items():
            if hasattr(profile, trait):
                current = getattr(profile, trait)
                adjustment = self.learning_rate * (target_value - current)
                new_value = max(0.0, min(1.0, current + adjustment))
                setattr(profile, trait, new_value)

        # Save updated profile
        self.personality_repo.save(profile)

        # Track history
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = []
        self.interaction_history[user_id].append({
            "timestamp": datetime.utcnow(),
            "signals": signals
        })

    def _analyze_interaction_signals(self, interaction: InteractionEvent) -> Dict[str, float]:
        """Analyze interaction to extract personality signals."""

        signals = {}
        content = interaction.content.lower()

        # Formality detection
        formal_words = ["please", "kindly", "would you", "could you", "regarding", "therefore"]
        casual_words = ["hey", "yeah", "gonna", "wanna", "lol", "btw"]

        formal_count = sum(1 for w in formal_words if w in content)
        casual_count = sum(1 for w in casual_words if w in content)

        if formal_count + casual_count > 0:
            signals["formality"] = formal_count / (formal_count + casual_count)

        # Detail orientation
        word_count = len(content.split())
        if word_count > 100:
            signals["detail_orientation"] = 0.8
        elif word_count > 50:
            signals["detail_orientation"] = 0.6
        elif word_count < 20:
            signals["detail_orientation"] = 0.3

        # Pace (based on response time if available)
        if interaction.duration_seconds > 0:
            if interaction.duration_seconds < 30:
                signals["pace"] = 0.8  # Fast
            elif interaction.duration_seconds > 120:
                signals["pace"] = 0.3  # Slow
            else:
                signals["pace"] = 0.5

        # Warmth (based on emotional language)
        warm_words = ["thanks", "appreciate", "great", "love", "wonderful", "helpful"]
        cold_words = ["just", "only", "simply", "need", "want", "must"]

        warm_count = sum(1 for w in warm_words if w in content)
        cold_count = sum(1 for w in cold_words if w in content)

        if warm_count + cold_count > 0:
            signals["warmth"] = warm_count / (warm_count + cold_count)

        # Directness (based on question vs statement)
        if "?" in content:
            signals["directness"] = 0.4  # Questions are less direct
        elif content.endswith("!"):
            signals["directness"] = 0.8  # Exclamations are direct

        # Humor (based on emoji or humor words)
        humor_indicators = ["haha", "lol", ":)", "😄", "😂", "funny"]
        if any(h in content for h in humor_indicators):
            signals["humor_level"] = 0.7

        return signals

    def generate_response_tone(
        self,
        user_id: str,
        context: Dict
    ) -> Dict:
        """Generate tone parameters for response generation."""

        profile = self.get_profile(user_id)
        urgency = context.get("urgency", "normal")

        tone = {
            "length": self._calculate_optimal_length(profile, urgency),
            "formality": self._adjust_formality(profile, context),
            "emotional_valence": self._determine_emotional_tone(profile, urgency),
            "directness": profile.directness,
            "detail_level": profile.detail_orientation,
            "warmth": profile.warmth
        }

        return tone

    def _calculate_optimal_length(
        self,
        profile: PersonalityProfile,
        urgency: str
    ) -> str:
        """Calculate optimal response length."""

        if urgency == "high":
            return "concise"

        if profile.detail_orientation > 0.7:
            return "detailed"
        elif profile.detail_orientation < 0.3:
            return "brief"
        else:
            return "moderate"

    def _adjust_formality(
        self,
        profile: PersonalityProfile,
        context: Dict
    ) -> float:
        """Adjust formality based on profile and context."""

        base_formality = profile.formality

        # Increase formality for professional contexts
        if context.get("context_type") == "professional":
            return min(1.0, base_formality + 0.2)

        # Decrease for casual contexts
        if context.get("context_type") == "casual":
            return max(0.0, base_formality - 0.2)

        return base_formality

    def _determine_emotional_tone(
        self,
        profile: PersonalityProfile,
        urgency: str
    ) -> str:
        """Determine emotional tone for response."""

        if urgency == "high":
            return "focused"

        if profile.warmth > 0.7:
            return "warm"
        elif profile.warmth < 0.3:
            return "neutral"
        else:
            return "friendly"

    def apply_style_filter(
        self,
        user_id: str,
        content: str,
        context: Dict
    ) -> str:
        """Apply personality-based style adjustments to content."""

        profile = self.get_profile(user_id)
        tone = self.generate_response_tone(user_id, context)

        # This is a simplified style application
        # In a real system, this would modify the actual content

        styled_content = content

        # Add warmth
        if tone["warmth"] > 0.7 and not content.startswith("I "):
            greetings = ["", "Sure! ", "Of course! ", "Happy to help! "]
            styled_content = greetings[int(tone["warmth"] * 3)] + styled_content

        # Adjust formality
        if tone["formality"] < 0.3:
            # Make more casual
            styled_content = styled_content.replace("I would", "I'd")
            styled_content = styled_content.replace("I will", "I'll")
            styled_content = styled_content.replace("cannot", "can't")

        return styled_content
