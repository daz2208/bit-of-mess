class AdaptivePersonalityCore:
    def __init__(self):
        self.base_traits = {
            "assertiveness": 0.6,      # 0=passive, 1=dominant
            "warmth": 0.7,             # 0=cold, 1=warm  
            "detail_orientation": 0.5,  # 0=big_picture, 1=detailed
            "formality": 0.4,          # 0=casual, 1=formal
            "humor_level": 0.3         # 0=serious, 1=playful
        }
        self.communication_style = None
        self.learning_rate = 0.1
        
    def adapt_style_based_on_user(self, user_style_signals):
        """Dynamically adjust personality based on user interaction patterns"""
        
        # Analyze user's communication style
        user_style = self._analyze_communication_patterns(user_style_signals)
        
        # Gradually adapt toward user's style
        for trait in self.base_traits:
            current = self.base_traits[trait]
            target = user_style.get(trait, current)
            adjustment = self.learning_rate * (target - current)
            self.base_traits[trait] = current + adjustment
            
        return self._generate_communication_guidelines()
    
    def generate_response_tone(self, message_urgency, relationship_context):
        """Apply personality filters to response generation"""
        
        tone_parameters = {
            "length": self._calculate_optimal_length(message_urgency),
            "formality": self._adjust_formality(relationship_context),
            "emotional_valence": self._determine_emotional_tone(message_urgency),
            "directness": self.base_traits["assertiveness"]
        }
        
        return tone_parameters