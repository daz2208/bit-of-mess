"""Transparent action execution with explanations."""

from typing import Dict, Optional
from datetime import datetime

from ..models.action import Decision, ActionResult, ActionPlan
from ..models.base import ActionType


class TransparentExecutor:
    """Execute actions with appropriate transparency and explanation."""

    def __init__(self):
        self.execution_history: Dict[str, list] = {}

    async def execute(self, decision: Decision) -> ActionResult:
        """Execute a decision with transparency."""

        if not decision.action_plan:
            return ActionResult(
                user_id=decision.user_id,
                decision_id=decision.id,
                success=False,
                explanation="No action plan provided"
            )

        # Execute the core action
        result = await self._execute_action(decision.action_plan)

        # Generate explanation based on transparency level
        explanation = await self._generate_explanation(decision, result)

        # Create result
        action_result = ActionResult(
            user_id=decision.user_id,
            decision_id=decision.id,
            success=result.get("success", True),
            result_data=result,
            explanation=explanation,
            confidence=decision.confidence
        )

        # Track execution
        self._track_execution(decision.user_id, action_result)

        return action_result

    async def _execute_action(self, action_plan: ActionPlan) -> Dict:
        """Execute the core action. Override in implementations."""

        # This is a base implementation
        # Real implementations would actually perform the action

        return {
            "success": True,
            "action": action_plan.description,
            "parameters": action_plan.parameters,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _generate_explanation(
        self,
        decision: Decision,
        result: Dict
    ) -> str:
        """Generate explanation based on transparency level."""

        if decision.transparency_level == "full":
            return await self._generate_detailed_explanation(decision, result)
        elif decision.transparency_level == "summary":
            return await self._generate_brief_explanation(decision, result)
        else:  # on_request
            return ""

    async def _generate_detailed_explanation(
        self,
        decision: Decision,
        result: Dict
    ) -> str:
        """Generate detailed explanation of the decision and action."""

        sections = []

        # Action taken
        if decision.action_plan:
            sections.append(f"**Action**: {decision.action_plan.description}")

        # Decision type
        action_type_str = decision.action_type.value.replace("_", " ").title()
        sections.append(f"**Decision Type**: {action_type_str}")

        # Confidence level
        sections.append(f"**Confidence**: {decision.confidence:.0%}")

        # Key reasoning
        if decision.reasoning_chain:
            sections.append("**Key Factors**:")
            for reason in decision.reasoning_chain[:3]:
                sections.append(f"  - {reason}")

        # Result
        if result.get("success"):
            sections.append("**Status**: Completed successfully")
        else:
            sections.append(f"**Status**: Failed - {result.get('error', 'Unknown error')}")

        return "\n".join(sections)

    async def _generate_brief_explanation(
        self,
        decision: Decision,
        result: Dict
    ) -> str:
        """Generate brief summary of the action."""

        if decision.action_plan:
            action_desc = decision.action_plan.description
        else:
            action_desc = "No action"

        status = "Done" if result.get("success") else "Failed"
        confidence = f"({decision.confidence:.0%} confident)"

        return f"{action_desc} - {status} {confidence}"

    def _track_execution(self, user_id: str, result: ActionResult):
        """Track execution history for learning."""

        if user_id not in self.execution_history:
            self.execution_history[user_id] = []

        self.execution_history[user_id].append({
            "result_id": result.id,
            "decision_id": result.decision_id,
            "success": result.success,
            "confidence": result.confidence,
            "timestamp": result.created_at
        })

        # Keep only recent history
        if len(self.execution_history[user_id]) > 100:
            self.execution_history[user_id] = self.execution_history[user_id][-100:]

    def get_success_rate(self, user_id: str) -> float:
        """Get historical success rate for a user."""

        if user_id not in self.execution_history:
            return 0.5  # Default

        history = self.execution_history[user_id]
        if not history:
            return 0.5

        successes = sum(1 for h in history if h["success"])
        return successes / len(history)
