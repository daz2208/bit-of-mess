class ExplicitFeedbackProcessor:
    def __init__(self):
        self.feedback_types = {
            "direct_correction": self._handle_direct_correction,
            "praise_criticism": self._handle_praise_criticism,
            "rule_definition": self._handle_rule_definition,
            "preference_statement": self._handle_preference_statement
        }
    
    async def analyze_event(self, event):
        if event.type not in self.feedback_types:
            return None
            
        handler = self.feedback_types[event.type]
        return await handler(event)
    
    async def _handle_direct_correction(self, event):
        """User directly corrects agent behavior"""
        
        correction_data = {
            "incorrect_action": event.data["wrong_behavior"],
            "correct_action": event.data["correct_behavior"],
            "context": event.context,
            "timestamp": event.timestamp,
            "user_emotional_tone": event.emotional_tone
        }
        
        # High-confidence update
        return LearningUpdate(
            type="behavior_correction",
            confidence=0.95,
            priority="high",
            update_rules=[self._create_behavior_rule(correction_data)],
            affected_memories=["procedural", "preference"]
        )
    
    async def _handle_rule_definition(self, event):
        """User explicitly defines a new rule"""
        
        rule = {
            "condition": event.data["condition"],
            "action": event.data["preferred_action"],
            "strength": 1.0,  # Explicit rules start strong
            "exceptions": [],
            "source": "user_explicit",
            "created_at": event.timestamp
        }
        
        return LearningUpdate(
            type="explicit_rule",
            confidence=0.99,
            priority="critical",
            update_rules=[rule],
            affected_memories=["preference"]
        )