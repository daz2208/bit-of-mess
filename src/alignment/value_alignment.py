"""Value alignment system to ensure actions align with user values."""

from typing import List, Dict, Optional
from datetime import datetime

from ..models.action import ActionPlan, Decision
from ..models.memory import Rule, PreferenceNode
from ..models.base import ActionType
from ..storage.repositories import RuleRepository, PreferenceRepository


class ValueAlignmentSystem:
    """
    System to ensure agent actions align with user values and preferences.

    Checks:
    - Explicit rules
    - Implicit preferences
    - Ethical frameworks
    """

    def __init__(
        self,
        rule_repo: RuleRepository,
        pref_repo: PreferenceRepository
    ):
        self.rule_repo = rule_repo
        self.pref_repo = pref_repo

        # Core ethical frameworks
        self.ethical_frameworks = [
            "beneficence",   # Do good
            "autonomy",      # Respect user choice
            "privacy",       # Protect user data
            "transparency",  # Be clear about actions
            "non_maleficence"  # Do no harm
        ]

    async def validate(
        self,
        user_id: str,
        decision: Decision
    ) -> Decision:
        """Validate a decision against value systems."""

        if not decision.action_plan:
            return decision

        violations = []

        # Check explicit rules
        rule_violations = await self._check_rule_violations(
            user_id, decision.action_plan
        )
        violations.extend(rule_violations)

        # Check implicit preferences
        pref_violations = await self._check_preference_violations(
            user_id, decision.action_plan
        )
        violations.extend(pref_violations)

        # Check ethical frameworks
        ethical_issues = await self._ethical_review(decision.action_plan)
        violations.extend(ethical_issues)

        # If violations found, modify or block action
        if violations:
            return await self._handle_violations(decision, violations)

        return decision

    async def _check_rule_violations(
        self,
        user_id: str,
        action_plan: ActionPlan
    ) -> List[Dict]:
        """Check if action violates any explicit rules."""

        violations = []
        rules = self.rule_repo.get_by_user(user_id)

        action_desc = action_plan.description.lower()
        action_params = action_plan.parameters

        for rule in rules:
            condition_words = rule.condition.lower().split()

            # Check if action matches condition
            matches_condition = any(
                word in action_desc for word in condition_words
            )

            if matches_condition:
                # Check if action follows the rule
                action_words = rule.action.lower().split()
                follows_rule = any(
                    word in action_desc for word in action_words
                )

                # Check for negation (rule might be "don't" do something)
                is_prohibition = any(
                    neg in rule.action.lower()
                    for neg in ["don't", "never", "avoid", "not"]
                )

                if is_prohibition and not follows_rule:
                    # Prohibition rule being violated
                    violations.append({
                        "type": "rule_violation",
                        "rule_id": rule.id,
                        "rule": rule.condition,
                        "severity": "high" if rule.strength > 0.8 else "medium",
                        "message": f"Action violates rule: {rule.condition}"
                    })
                elif not is_prohibition and not follows_rule:
                    # Positive rule not being followed
                    violations.append({
                        "type": "rule_violation",
                        "rule_id": rule.id,
                        "rule": rule.condition,
                        "severity": "medium",
                        "message": f"Action doesn't follow rule: {rule.action}"
                    })

        return violations

    async def _check_preference_violations(
        self,
        user_id: str,
        action_plan: ActionPlan
    ) -> List[Dict]:
        """Check if action might violate user preferences."""

        violations = []
        prefs = self.pref_repo.get_by_user(user_id)

        action_desc = action_plan.description.lower()

        for pref in prefs:
            if pref.strength < 0.5:  # Only check strong preferences
                continue

            pref_text = pref.preference.lower()

            # Check for negative preferences
            is_negative = any(
                neg in pref_text
                for neg in ["don't", "never", "avoid", "dislike", "hate", "not"]
            )

            if is_negative:
                # Extract what should be avoided
                avoid_terms = [
                    word for word in pref_text.split()
                    if word not in ["don't", "never", "avoid", "dislike", "hate", "not", "i"]
                ]

                if any(term in action_desc for term in avoid_terms):
                    violations.append({
                        "type": "preference_violation",
                        "preference_id": pref.id,
                        "preference": pref.preference,
                        "confidence": pref.confidence * pref.strength,
                        "severity": "high" if pref.strength > 0.8 else "medium",
                        "message": f"Action may violate preference: {pref.preference}"
                    })

        return violations

    async def _ethical_review(self, action_plan: ActionPlan) -> List[Dict]:
        """Review action against ethical frameworks."""

        issues = []
        params = action_plan.parameters

        # Autonomy check
        if params.get("overrides_user_choice", False):
            issues.append({
                "type": "ethical_issue",
                "framework": "autonomy",
                "severity": "high",
                "message": "Action overrides user choice without consent"
            })

        # Privacy check
        if params.get("shares_private_data", False):
            issues.append({
                "type": "ethical_issue",
                "framework": "privacy",
                "severity": "critical",
                "message": "Action shares private data externally"
            })

        # Transparency check
        if params.get("hidden_action", False):
            issues.append({
                "type": "ethical_issue",
                "framework": "transparency",
                "severity": "medium",
                "message": "Action is hidden from user"
            })

        # Non-maleficence check
        if params.get("potential_harm", False):
            issues.append({
                "type": "ethical_issue",
                "framework": "non_maleficence",
                "severity": "critical",
                "message": "Action has potential to cause harm"
            })

        return issues

    async def _handle_violations(
        self,
        decision: Decision,
        violations: List[Dict]
    ) -> Decision:
        """Handle violations by modifying or blocking the action."""

        # Check severity
        critical = any(v["severity"] == "critical" for v in violations)
        high = any(v["severity"] == "high" for v in violations)

        if critical:
            # Block action entirely
            decision.action_type = ActionType.ASK_CLARIFICATION
            decision.reasoning_chain.append(
                f"Action blocked due to critical violation: {violations[0]['message']}"
            )
            decision.confidence = 0.0

        elif high:
            # Change to suggestion instead of autonomous
            if decision.action_type == ActionType.AUTONOMOUS_EXECUTE:
                decision.action_type = ActionType.SUGGEST_WITH_RATIONALE
                decision.reasoning_chain.append(
                    f"Changed to suggestion due to violation: {violations[0]['message']}"
                )
                decision.confidence *= 0.5

        else:
            # Add warnings to reasoning
            for v in violations:
                decision.reasoning_chain.append(f"Warning: {v['message']}")

        # Always increase transparency for violations
        decision.transparency_level = "full"

        return decision

    async def learn_from_feedback(
        self,
        user_id: str,
        decision_id: str,
        feedback_type: str,
        feedback_data: Dict
    ):
        """Learn from user feedback on decisions."""

        if feedback_type == "explicit_correction":
            # User explicitly said this was wrong
            # This is handled by the learning system
            pass

        elif feedback_type == "implicit_rejection":
            # User rejected without explanation
            # Weaken related preferences
            pass

        elif feedback_type == "praise":
            # Reinforce the pattern
            pass
