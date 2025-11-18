"""Tests for memory components."""

import pytest
import asyncio
import os
from src.storage.database import Database
from src.storage.repositories import MemoryRepository, PreferenceRepository
from src.memory.hierarchical import HierarchicalMemory
from src.memory.preference_graph import PreferenceGraph
from src.memory.vector_store import VectorStore
from src.models.base import MemoryType


class TestDatabase:
    """Tests for Database."""

    def setup_method(self):
        self.db_path = "test_db.db"
        self.db = Database(self.db_path)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_database_creation(self):
        assert os.path.exists(self.db_path)

    def test_table_creation(self):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

        assert "memories" in tables
        assert "preferences" in tables
        assert "rules" in tables
        assert "interactions" in tables


class TestMemoryRepository:
    """Tests for MemoryRepository."""

    def setup_method(self):
        self.db_path = "test_memory_repo.db"
        self.db = Database(self.db_path)
        self.repo = MemoryRepository(self.db)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_save_and_get(self):
        from src.models.memory import MemoryEntry
        import numpy as np

        memory = MemoryEntry(
            user_id="test_user",
            memory_type=MemoryType.SEMANTIC,
            content="Test memory content",
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            importance=0.8
        )

        memory_id = self.repo.save(memory)
        retrieved = self.repo.get(memory_id)

        assert retrieved is not None
        assert retrieved.content == "Test memory content"
        assert retrieved.importance == 0.8

    def test_get_by_user(self):
        from src.models.memory import MemoryEntry

        for i in range(3):
            memory = MemoryEntry(
                user_id="test_user",
                memory_type=MemoryType.EPISODIC,
                content=f"Memory {i}"
            )
            self.repo.save(memory)

        memories = self.repo.get_by_user("test_user")
        assert len(memories) == 3

    def test_update_access(self):
        from src.models.memory import MemoryEntry

        memory = MemoryEntry(
            user_id="test_user",
            memory_type=MemoryType.SEMANTIC,
            content="Test"
        )
        memory_id = self.repo.save(memory)

        self.repo.update_access(memory_id)
        retrieved = self.repo.get(memory_id)
        assert retrieved.access_count == 1


class TestPreferenceGraph:
    """Tests for PreferenceGraph."""

    def setup_method(self):
        self.db_path = "test_pref_graph.db"
        self.db = Database(self.db_path)
        self.pref_repo = PreferenceRepository(self.db)
        self.graph = PreferenceGraph(self.pref_repo)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    @pytest.mark.asyncio
    async def test_add_preference(self):
        pref_id = await self.graph.add_preference(
            user_id="test_user",
            category="communication",
            preference="brief responses",
            strength=0.8
        )
        assert pref_id is not None

    @pytest.mark.asyncio
    async def test_get_preferences(self):
        await self.graph.add_preference(
            user_id="test_user",
            category="scheduling",
            preference="morning meetings",
            strength=0.9
        )

        prefs = await self.graph.get_preferences("test_user", category="scheduling")
        assert len(prefs) > 0
        assert prefs[0].preference == "morning meetings"

    @pytest.mark.asyncio
    async def test_predict_discomfort(self):
        await self.graph.add_preference(
            user_id="test_user",
            category="time",
            preference="never schedule before 9am",
            strength=0.95
        )

        violations = await self.graph.predict_discomfort(
            "test_user",
            {"action": "schedule meeting at 8am"}
        )
        assert len(violations) > 0


class TestHierarchicalMemory:
    """Tests for HierarchicalMemory."""

    def setup_method(self):
        self.db_path = "test_hier_memory.db"
        self.db = Database(self.db_path)
        self.memory_repo = MemoryRepository(self.db)
        self.pref_repo = PreferenceRepository(self.db)
        self.memory = HierarchicalMemory(self.memory_repo, self.pref_repo)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    @pytest.mark.asyncio
    async def test_store_memory(self):
        memory_id = await self.memory.store(
            user_id="test_user",
            content="Important meeting notes",
            memory_type=MemoryType.EPISODIC,
            importance=0.9
        )
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_retrieve_memory(self):
        await self.memory.store(
            user_id="test_user",
            content="Meeting with client about project",
            memory_type=MemoryType.EPISODIC,
            importance=0.8
        )

        results = await self.memory.retrieve(
            user_id="test_user",
            query="client project meeting"
        )
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
