"""Meta-reasoning engine for decision making."""

from typing import Dict, List, Optional
from datetime import datetime

from ..models.action import ActionPlan, Decision
from ..models.base import ActionType, Context
from ..models.memory import PreferenceNode, Rule


class MetaReasoningEngine:
    """
    Meta-reasoning engine that decides when and how to act.

    Makes decisions about:
    - Whether to act autonomously
    - Whether to suggest and explain
    - Whether to ask for clarification
    - Whether to silently learn
    """

    def __init__(self):
        self.action_thresholds = {
            "immediate_act": 0.85,      # High confidence, low intrusion
            "suggest": 0.6,             # Medium confidence
            "ask_clarification": 0.3,   # Low confidence
            "silent_learn": 0.1         # Very low confidence
        }

    async def deliberate(
        self,
        context: Dict,
        preferences: List[PreferenceNode],
        rules: List[Rule],
        proposed_action: Optional[ActionPlan] = None
    ) -> Decision:
        """
        Main deliberation process.

        Analyzes context, confidence, and potential value to decide action type.
        """

        user_id = context.get("user_id", "")

        # Calculate confidence in proposed action
        confidence = await self._calculate_action_confidence(
            context, preferences, rules, proposed_action
        )

        # Calculate intrusion level
        intrusion_score = self._calculate_intrusion_level(context, proposed_action)

        # Calculate potential value
        value_score = self._estimate_value_added(context, preferences, proposed_action)

        # Select action type based on scores
        action_type = self._select_action_type(confidence, intrusion_score, value_score)

        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(
            context, confidence, intrusion_score, value_score, action_type
        )

        # Determine transparency level
        transparency = self._determine_transparency(action_type, confidence)

        return Decision(
            user_id=user_id,
            action_type=action_type,
            action_plan=proposed_action,
            confidence=confidence,
            intrusion_score=intrusion_score,
            value_score=value_score,
            reasoning_chain=reasoning_chain,
            transparency_level=transparency
        )

    async def _calculate_action_confidence(
        self,
        context: Dict,
        preferences: List[PreferenceNode],
        rules: List[Rule],
        proposed_action: Optional[ActionPlan]
    ) -> float:
        """Calculate confidence in the proposed action."""

        if not proposed_action:
            return 0.0

        confidence = 0.5  # Base confidence

        # Boost from matching rules
        action_desc = proposed_action.description.lower()
        for rule in rules:
            if any(word in action_desc for word in rule.condition.lower().split()):
                confidence += 0.2 * rule.strength
                break

        # Boost from matching preferences
        for pref in preferences:
            pref_words = set(pref.preference.lower().split())
            action_words = set(action_desc.split())
            overlap = len(pref_words & action_words)
            if overlap > 0:
                confidence += 0.1 * pref.strength * pref.confidence

        # Historical success rate could boost confidence
        if "historical_success" in context:
            confidence += 0.1 * context["historical_success"]

        return min(1.0, max(0.0, confidence))

    def _calculate_intrusion_level(
        self,
        context: Dict,
        proposed_action: Optional[ActionPlan]
    ) -> float:
        """Calculate how intrusive the action would be."""

        if not proposed_action:
            return 0.0

        intrusion = 0.3  # Base intrusion

        # Higher intrusion for certain action types
        action_type = proposed_action.action_type
        if action_type == ActionType.AUTONOMOUS_EXECUTE:
            intrusion += 0.3

        # Higher intrusion during certain times
        time_of_day = context.get("time_of_day", "")
        if time_of_day in ["night", "evening"]:
            intrusion += 0.1

        # Higher intrusion if user is busy
        if context.get("user_state", {}).get("busy", False):
            intrusion += 0.2

        # Check for sensitive parameters
        params = proposed_action.parameters
        if params.get("modifies_data", False):
            intrusion += 0.2
        if params.get("sends_communication", False):
            intrusion += 0.15

        return min(1.0, intrusion)

    def _estimate_value_added(
        self,
        context: Dict,
        preferences: List[PreferenceNode],
        proposed_action: Optional[ActionPlan]
    ) -> float:
        """Estimate the value this action would add for the user."""

        if not proposed_action:
            return 0.0

        value = 0.3  # Base value

        # Higher value if action matches strong preferences
        action_desc = proposed_action.description.lower()
        for pref in preferences:
            if pref.strength > 0.7:
                if any(word in action_desc for word in pref.preference.lower().split()):
                    value += 0.2

        # Higher value for urgent contexts
        if context.get("urgency", "low") == "high":
            value += 0.2

        # Higher value if saves user time/effort
        params = proposed_action.parameters
        if params.get("saves_time", False):
            value += 0.15
        if params.get("prevents_error", False):
            value += 0.2

        return min(1.0, value)

    def _select_action_type(
        self,
        confidence: float,
        intrusion: float,
        value: float
    ) -> ActionType:
        """Select action type based on confidence, intrusion, and value."""

        # High confidence + low intrusion = autonomous execution
        if confidence > self.action_thresholds["immediate_act"] and intrusion < 0.4:
            return ActionType.AUTONOMOUS_EXECUTE

        # High value + medium confidence = suggest with rationale
        if value > 0.7 and confidence > self.action_thresholds["suggest"]:
            return ActionType.SUGGEST_WITH_RATIONALE

        # Low confidence = ask for clarification
        if confidence < self.action_thresholds["ask_clarification"]:
            return ActionType.ASK_CLARIFICATION

        # High intrusion even with confidence = suggest instead
        if intrusion > 0.6:
            return ActionType.SUGGEST_WITH_RATIONALE

        # Default to silent learning
        if confidence < self.action_thresholds["suggest"]:
            return ActionType.SILENT_LEARN

        return ActionType.SUGGEST_WITH_RATIONALE

    def _generate_reasoning_chain(
        self,
        context: Dict,
        confidence: float,
        intrusion: float,
        value: float,
        action_type: ActionType
    ) -> List[str]:
        """Generate human-readable reasoning chain."""

        reasons = []

        # Confidence reasoning
        if confidence > 0.8:
            reasons.append(f"High confidence ({confidence:.0%}) based on matching preferences and rules")
        elif confidence > 0.5:
            reasons.append(f"Moderate confidence ({confidence:.0%}) suggests action but with explanation")
        else:
            reasons.append(f"Low confidence ({confidence:.0%}) requires clarification")

        # Intrusion reasoning
        if intrusion < 0.3:
            reasons.append("Low intrusion - safe to proceed autonomously")
        elif intrusion > 0.6:
            reasons.append("High intrusion - requires explicit user approval")

        # Value reasoning
        if value > 0.7:
            reasons.append(f"High value ({value:.0%}) - action strongly benefits user")

        # Context reasoning
        if context.get("urgency") == "high":
            reasons.append("Urgent context increases action priority")

        if context.get("time_of_day") == "morning":
            reasons.append("Morning context - user typically more receptive")

        return reasons

    def _determine_transparency(
        self,
        action_type: ActionType,
        confidence: float
    ) -> str:
        """Determine appropriate transparency level for the action."""

        if action_type == ActionType.AUTONOMOUS_EXECUTE:
            if confidence > 0.9:
                return "summary"
            return "full"

        if action_type == ActionType.SUGGEST_WITH_RATIONALE:
            return "full"

        if action_type == ActionType.ASK_CLARIFICATION:
            return "full"

        return "on_request"
