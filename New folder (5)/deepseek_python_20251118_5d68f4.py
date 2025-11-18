class BehavioralPatternAnalyzer:
    def __init__(self):
        self.pattern_buffers = {
            "short_term": deque(maxlen=100),  # Recent interactions
            "medium_term": deque(maxlen=1000), # Last few days
            "long_term": []  # Permanent storage for significant patterns
        }
        self.change_detector = ChangePointDetector()
    
    async def analyze_event(self, event):
        """Detect significant behavioral pattern changes"""
        
        # Add to pattern buffers
        self._add_to_buffers(event)
        
        # Check for significant pattern changes
        pattern_shifts = await self._detect_pattern_shifts()
        
        # Extract new preferences from behavioral changes
        preference_updates = await self._infer_preferences_from_behavior()
        
        return LearningUpdate(
            type="behavioral_pattern",
            pattern_shifts=pattern_shifts,
            preference_updates=preference_updates,
            confidence=self._calculate_pattern_confidence(pattern_shifts)
        )
    
    async def _detect_pattern_shifts(self):
        """Use statistical change detection to find significant shifts"""
        
        shifts = []
        
        # Analyze communication style changes
        comm_shift = self.change_detector.detect_shift(
            self.pattern_buffers["short_term"],
            self.pattern_buffers["medium_term"],
            feature="communication_style"
        )
        
        if comm_shift.significant:
            shifts.append({
                "type": "communication_style",
                "from": comm_shift.old_pattern,
                "to": comm_shift.new_pattern,
                "confidence": comm_shift.confidence
            })
        
        # Analyze work habit changes
        work_shift = self.change_detector.detect_shift(
            self.pattern_buffers["short_term"],
            self.pattern_buffers["medium_term"],
            feature="work_preferences"
        )
        
        if work_shift.significant:
            shifts.append({
                "type": "work_habits", 
                "from": work_shift.old_pattern,
                "to": work_shift.new_pattern,
                "confidence": work_shift.confidence
            })
        
        return shifts
    
    async def _infer_preferences_from_behavior(self):
        """Infer preferences from consistent behavioral patterns"""
        
        preferences = []
        
        # Analyze time-of-day preferences
        time_prefs = self._analyze_temporal_preferences()
        if time_prefs.confidence > 0.8:
            preferences.append(time_prefs)
        
        # Analyze topic engagement preferences
        topic_prefs = self._analyze_topic_engagement()
        if topic_prefs.confidence > 0.7:
            preferences.append(topic_prefs)
        
        return preferences