class MetaReasoningEngine:
    def __init__(self):
        self.action_thresholds = {
            "immediate_act": 0.9,      # High confidence, high urgency
            "suggest": 0.7,            # Medium confidence
            "ask_clarification": 0.4,   # Low confidence
            "silent_learn": 0.1        # Very low confidence
        }
        
    async def deliberate(self, context):
        """The core decision-making process"""
        
        # Calculate confidence in proposed action
        confidence = await self._calculate_action_confidence(context)
        
        # Calculate intrusion level
        intrusion_score = self._calculate_intrusion_level(context)
        
        # Calculate potential value
        value_score = self._estimate_value_added(context)
        
        # Meta-decision
        decision_type = self._select_action_type(
            confidence, intrusion_score, value_score
        )
        
        return {
            "action_type": decision_type,
            "action_plan": context.proposed_action,
            "confidence": confidence,
            "reasoning_chain": self._explain_reasoning(context),
            "transparency_level": self._determine_transparency(decision_type)
        }
    
    def _select_action_type(self, confidence, intrusion, value):
        """Decision matrix for action selection"""
        
        if confidence > self.action_thresholds["immediate_act"] and intrusion < 0.3:
            return "autonomous_execute"
        elif value > 0.8 and confidence > self.action_thresholds["suggest"]:
            return "suggest_with_rationale"
        elif confidence < self.action_thresholds["ask_clarification"]:
            return "ask_clarification"
        else:
            return "silent_learn"