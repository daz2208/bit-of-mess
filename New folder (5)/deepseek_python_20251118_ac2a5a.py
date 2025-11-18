class TransparentExecutor:
    async def execute_action(self, validated_plan):
        """Execute with appropriate transparency"""
        
        action_result = await self._execute_core_action(validated_plan.action_plan)
        
        if validated_plan.transparency_level == "full":
            explanation = await self._generate_detailed_explanation(validated_plan)
        elif validated_plan.transparency_level == "summary":
            explanation = await self._generate_brief_explanation(validated_plan)
        else:  # "on_request"
            explanation = None
            
        return {
            "action": action_result,
            "explanation": explanation,
            "confidence": validated_plan.confidence,
            "reasoning_available": True
        }
    
    async def _generate_detailed_explanation(self, validated_plan):
        """Generate the 'why' behind the action"""
        
        explanation_sections = [
            f"**Action Taken**: {validated_plan.action_plan.description}",
            f"**Confidence Level**: {validated_plan.confidence:.1%}",
            f"**Key Factors**:",
        ]
        
        for reason in validated_plan.reasoning_chain[:3]:  # Top 3 reasons
            explanation_sections.append(f"- {reason}")
            
        explanation_sections.append(
            f"**Decision Type**: {validated_plan.action_type.replace('_', ' ').title()}"
        )
        
        return "\n".join(explanation_sections)