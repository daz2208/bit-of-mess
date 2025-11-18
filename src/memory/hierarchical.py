"""Hierarchical memory system combining multiple memory types."""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import uuid

from ..models.memory import MemoryEntry
from ..models.base import MemoryType, Context
from ..storage.repositories import MemoryRepository, PreferenceRepository
from .vector_store import VectorStore
from .preference_graph import PreferenceGraph


class HierarchicalMemory:
    """
    Hierarchical memory system with four memory types:
    - Episodic: Specific events and interactions
    - Semantic: General knowledge and concepts
    - Procedural: How-to knowledge and procedures
    - Preference: User preferences and values
    """

    def __init__(self, memory_repo: MemoryRepository, pref_repo: PreferenceRepository):
        self.memory_repo = memory_repo
        self.pref_repo = pref_repo
        self.vector_store = VectorStore(memory_repo)
        self.preference_graph = PreferenceGraph(pref_repo)

    async def store(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        context_tags: List[str] = None,
        metadata: Dict = None
    ) -> str:
        """Store a new memory."""

        # Create embedding
        embedding = self.vector_store.create_embedding(content)

        memory = MemoryEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            context_tags=context_tags or [],
            metadata=metadata or {}
        )

        return self.memory_repo.save(memory)

    async def retrieve(
        self,
        user_id: str,
        query: str,
        memory_types: List[MemoryType] = None,
        top_k: int = 10,
        recency_weight: float = 0.3
    ) -> List[Tuple[MemoryEntry, float]]:
        """Retrieve relevant memories across all or specified types."""

        all_results = []

        types_to_search = memory_types or [
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL
        ]

        for mem_type in types_to_search:
            results = await self.vector_store.similarity_search(
                user_id=user_id,
                query=query,
                memory_type=mem_type,
                top_k=top_k,
                recency_weight=recency_weight
            )
            all_results.extend(results)

        # Sort all results by score
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Update access counts for retrieved memories
        for memory, score in all_results[:top_k]:
            self.memory_repo.update_access(memory.id)

        return all_results[:top_k]

    async def retrieve_relevant_context(
        self,
        user_id: str,
        query: str,
        context: Context,
        recency_weight: float = 0.3
    ) -> Dict:
        """Retrieve comprehensive context for a query."""

        # Get relevant memories
        memories = await self.retrieve(
            user_id=user_id,
            query=query,
            recency_weight=recency_weight
        )

        # Get relevant preferences
        preferences = await self.preference_graph.query_relevant_preferences(
            user_id=user_id,
            context={
                "query": query,
                "time_of_day": context.time_of_day,
                "day_type": context.day_type,
                **context.user_state
            }
        )

        return {
            "memories": memories,
            "preferences": preferences,
            "context": context
        }

    async def update_importance(self, memory_id: str, new_importance: float):
        """Update the importance of a memory."""
        memory = self.memory_repo.get(memory_id)
        if memory:
            memory.importance = max(0.0, min(1.0, new_importance))
            self.memory_repo.save(memory)

    async def consolidate(self, user_id: str):
        """
        Consolidate memories - merge similar ones, decay old unimportant ones.
        This simulates the brain's memory consolidation during sleep.
        """
        memories = self.memory_repo.get_by_user(user_id, limit=10000)

        # Group similar memories
        groups = []
        processed = set()

        for i, mem in enumerate(memories):
            if mem.id in processed:
                continue

            group = [mem]
            processed.add(mem.id)

            for j, other in enumerate(memories[i+1:], i+1):
                if other.id in processed:
                    continue

                # Check similarity
                if mem.embedding is not None and other.embedding is not None:
                    sim = self.vector_store._cosine_similarity(mem.embedding, other.embedding)
                    if sim > 0.8:  # Very similar
                        group.append(other)
                        processed.add(other.id)

            if len(group) > 1:
                groups.append(group)

        # Merge groups
        for group in groups:
            # Keep the most important/recent one
            group.sort(key=lambda m: (m.importance, m.access_count), reverse=True)
            primary = group[0]

            # Boost importance of merged memory
            primary.importance = min(1.0, primary.importance + 0.1 * len(group))

            # Delete others
            for mem in group[1:]:
                self.memory_repo.delete(mem.id)

            self.memory_repo.save(primary)

    async def forget(self, user_id: str, decay_threshold: float = 0.1):
        """
        Forget low-importance, rarely accessed memories.
        Simulates natural forgetting.
        """
        memories = self.memory_repo.get_by_user(user_id, limit=10000)

        now = datetime.utcnow()

        for memory in memories:
            # Calculate decay based on time and access
            days_since_access = (now - memory.last_accessed).days

            # Low importance + old + rarely accessed = forget
            if (memory.importance < decay_threshold and
                days_since_access > 30 and
                memory.access_count < 3):
                self.memory_repo.delete(memory.id)
