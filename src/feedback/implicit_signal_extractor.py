class ImplicitSignalExtractor:
    def __init__(self):
        self.signal_detectors = {
            "rejection_patterns": RejectionPatternDetector(),
            "adoption_rates": AdoptionRateAnalyzer(),
            "engagement_metrics": EngagementMetricsTracker(),
            "temporal_patterns": TemporalPatternFinder()
        }
    
    async def analyze_event(self, event):
        """Extract learning signals from implicit user behavior"""
        
        signals = []
        
        # Detect rejection patterns
        if self._is_rejection_event(event):
            rejection_signal = await self.signal_detectors["rejection_patterns"].analyze(event)
            signals.append(rejection_signal)
        
        # Track adoption rates of suggestions
        adoption_signal = await self.signal_detectors["adoption_rates"].track(event)
        if adoption_signal:
            signals.append(adoption_signal)
        
        # Analyze engagement depth
        engagement_signal = await self.signal_detectors["engagement_metrics"].measure(event)
        signals.append(engagement_signal)
        
        return self._aggregate_implicit_signals(signals)
    
    class RejectionPatternDetector:
        async def analyze(self, event):
            """Detect patterns in what the user rejects"""
            
            if event.type == "suggestion_ignored":
                pattern = {
                    "suggestion_type": event.data["suggestion_type"],
                    "content_pattern": self._extract_content_features(event.data["content"]),
                    "context": event.context,
                    "rejection_count": 1,
                    "total_occurrences": 1
                }
                
                confidence = 0.7  # Medium confidence for single rejection
                
            elif event.type == "suggestion_modified":
                # User used but modified the suggestion - partial rejection
                modifications = self._analyze_modifications(
                    event.data["original"], 
                    event.data["modified"]
                )
                
                pattern = {
                    "suggestion_type": event.data["suggestion_type"],
                    "rejected_aspects": modifications["rejected_parts"],
                    "kept_aspects": modifications["kept_parts"],
                    "modification_pattern": modifications["pattern"],
                    "context": event.context
                }
                
                confidence = 0.8  # Higher confidence for partial rejection
                
            return LearningSignal(
                type="preference_refinement",
                pattern=pattern,
                confidence=confidence,
                learning_rate=0.3
            )