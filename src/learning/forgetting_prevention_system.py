class ForgettingPreventionSystem:
    def __init__(self):
        self.importance_calculator = ImportanceCalculator()
        self.memory_rehearsal = MemoryRehearsalScheduler()
        self.knowledge_stability = KnowledgeStabilityTracker()
    
    async def prevent_catastrophic_forgetting(self, new_knowledge):
        """Ensure new learning doesn't erase important old knowledge"""
        
        # Calculate importance of existing knowledge that might be affected
        affected_knowledge = await self._find_affected_knowledge(new_knowledge)
        importance_scores = await self.importance_calculator.calculate_importance(affected_knowledge)
        
        # Protect high-importance knowledge
        protected_knowledge = [k for k, score in importance_scores.items() if score > 0.8]
        
        if protected_knowledge:
            # Schedule rehearsal of protected knowledge
            await self.memory_rehearsal.schedule_rehearsal(protected_knowledge)
            
            # Create contextual exceptions to prevent override
            await self._create_protection_exceptions(new_knowledge, protected_knowledge)
        
        # Apply new knowledge with importance-aware learning rate
        adaptive_rate = self._calculate_adaptive_learning_rate(new_knowledge, importance_scores)
        await self._apply_with_adaptive_rate(new_knowledge, adaptive_rate)
    
    async def _find_affected_knowledge(self, new_knowledge):
        """Find existing knowledge that might be contradicted or overwritten"""
        
        affected = []
        
        # Find contradictory rules
        contradictory_rules = await self._find_contradictory_rules(new_knowledge)
        affected.extend(contradictory_rules)
        
        # Find similar but different patterns
        similar_patterns = await self._find_similar_patterns(new_knowledge)
        affected.extend(similar_patterns)
        
        return affected
    
    def _calculate_adaptive_learning_rate(self, new_knowledge, importance_scores):
        """Reduce learning rate when it might affect important existing knowledge"""
        
        base_rate = 0.3
        max_importance = max(importance_scores.values()) if importance_scores else 0
        
        if max_importance > 0.9:
            # Drastically reduce rate for critical knowledge
            return base_rate * 0.1
        elif max_importance > 0.7:
            # Moderate reduction for important knowledge
            return base_rate * 0.5
        else:
            return base_rate