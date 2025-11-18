class CentralNervousSystem:
    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = HierarchicalMemory(user_id)
        self.personality = AdaptivePersonalityCore()
        self.reasoning = MetaReasoningEngine()
        self.alignment = ValueAlignmentCore()
        
    async def process_stimulus(self, stimulus, context):
        """Main entry point - how the agent perceives the world"""
        
        # 1. Contextual Understanding
        enriched_context = await self._enrich_context(stimulus, context)
        
        # 2. Personality Filtering
        personality_context = self.personality.apply_filters(enriched_context)
        
        # 3. Meta-Reasoning Decision
        action_plan = await self.reasoning.deliberate(personality_context)
        
        # 4. Value Alignment Check
        validated_plan = self.alignment.validate(action_plan)
        
        # 5. Execute with Explanation
        return await self._execute_with_transparency(validated_plan)

class HierarchicalMemory:
    def __init__(self, user_id):
        self.episodic_memory = VectorMemory("episodic")  # Specific events
        self.semantic_memory = VectorMemory("semantic")  # Concepts & knowledge  
        self.procedural_memory = VectorMemory("procedural")  # How-to knowledge
        self.preference_memory = PreferenceGraph()  # Taste & values
        
    async def retrieve_relevant_context(self, query, recency_weight=0.3):
        """Search across all memory types with temporal weighting"""
        contexts = []
        
        for memory_type in [self.episodic_memory, self.semantic_memory, 
                           self.procedural_memory, self.preference_memory]:
            results = await memory_type.similarity_search(
                query, 
                filter={"user_id": self.user_id},
                recency_weight=recency_weight
            )
            contexts.extend(results)
            
        return self._rank_contexts(contexts, query)