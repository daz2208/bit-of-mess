"""Tests for core system components."""

import pytest
import asyncio
import os
from src.core.central_system import CentralNervousSystem
from src.models.base import Context, Stimulus
from src.models.feedback import FeedbackType, EmotionalTone


class TestCentralNervousSystem:
    """Tests for CentralNervousSystem."""

    def setup_method(self):
        self.db_path = "test_cns.db"
        self.agent = CentralNervousSystem("test_user", self.db_path)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    @pytest.mark.asyncio
    async def test_process_stimulus(self):
        context = Context(user_id="test_user")
        stimulus = Stimulus(
            type="question",
            data={"message": "What time is the meeting?"},
            context=context
        )

        result = await self.agent.process_stimulus(stimulus)
        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "confidence")

    @pytest.mark.asyncio
    async def test_add_preference(self):
        await self.agent.add_preference(
            category="communication",
            preference="brief responses",
            strength=0.9
        )

        profile = await self.agent.get_profile_summary()
        assert profile["preferences_count"] >= 1

    @pytest.mark.asyncio
    async def test_add_rule(self):
        await self.agent.add_rule(
            condition="morning hours",
            action="protect focus time"
        )

        profile = await self.agent.get_profile_summary()
        assert profile["rules_count"] >= 1

    @pytest.mark.asyncio
    async def test_correct_behavior(self):
        await self.agent.correct_behavior(
            wrong_behavior="scheduled meeting at 8am",
            correct_behavior="never schedule before 9am",
            emotional_tone=EmotionalTone.FRUSTRATED
        )

        # Should create a learning update
        profile = await self.agent.get_profile_summary()
        assert profile is not None

    @pytest.mark.asyncio
    async def test_store_knowledge(self):
        await self.agent.store_knowledge(
            content="Important project deadline is next Friday",
            memory_type="semantic",
            importance=0.9,
            tags=["project", "deadline"]
        )

        # Should be retrievable
        results = await self.agent.memory.retrieve(
            user_id="test_user",
            query="project deadline"
        )
        assert len(results) > 0

    def test_classify_intent(self):
        result = self.agent.classify_intent("Schedule a meeting tomorrow")
        assert result["primary_intent"] == "schedule"

    def test_extract_entities(self):
        result = self.agent.extract_entities("Meeting at 3pm tomorrow")
        assert "times" in result
        assert "dates" in result

    def test_sentiment_analysis(self):
        result = self.agent.get_sentiment_analysis("This is great!")
        assert result["polarity"] > 0

    def test_create_goal(self):
        goal = self.agent.create_goal(
            title="Learn Python",
            description="Master Python programming",
            milestones=["Basics", "OOP", "Async"]
        )
        assert goal.title == "Learn Python"
        assert len(goal.milestones) == 3

    def test_get_active_goals(self):
        self.agent.create_goal(title="Goal 1")
        self.agent.create_goal(title="Goal 2")

        goals = self.agent.get_active_goals()
        assert len(goals) == 2

    def test_analytics_dashboard(self):
        dashboard = self.agent.get_analytics_dashboard()
        assert "learning" in dashboard
        assert "accuracy" in dashboard
        assert "totals" in dashboard

    def test_conversation_summary(self):
        summary = self.agent.get_conversation_summary()
        assert isinstance(summary, str)

    def test_list_plugins(self):
        plugins = self.agent.list_plugins()
        assert isinstance(plugins, list)
        # Should have built-in plugins
        assert len(plugins) > 0

    @pytest.mark.asyncio
    async def test_multiple_interactions(self):
        """Test multiple sequential interactions."""
        messages = [
            "Hello there",
            "Schedule a meeting tomorrow",
            "What time works best?",
            "Make it at 2pm"
        ]

        for msg in messages:
            context = Context(user_id="test_user")
            stimulus = Stimulus(
                type="message",
                data={"message": msg},
                context=context
            )
            result = await self.agent.process_stimulus(stimulus)
            assert result is not None

        # Conversation should have tracked all turns
        conv_context = self.agent.conversation.get_current_context("test_user")
        assert conv_context["turn_count"] >= len(messages)


class TestGoalTracking:
    """Tests for goal tracking functionality."""

    def setup_method(self):
        self.db_path = "test_goals.db"
        self.agent = CentralNervousSystem("test_user", self.db_path)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_goal_progress(self):
        goal = self.agent.create_goal(
            title="Test Goal",
            milestones=["Step 1", "Step 2"]
        )

        self.agent.update_goal_progress(goal.id, 0.5)
        updated = self.agent.goals.get_goal("test_user", goal.id)
        assert updated.progress == 0.5

    def test_goal_suggestions(self):
        self.agent.create_goal(
            title="Important Goal",
            milestones=["First milestone"],
            priority=9
        )

        suggestions = self.agent.get_goal_suggestions()
        assert len(suggestions) > 0
        assert "Important Goal" in suggestions[0]["goal"]


class TestProactiveSuggestions:
    """Tests for proactive suggestion system."""

    def setup_method(self):
        self.db_path = "test_proactive.db"
        self.agent = CentralNervousSystem("test_user", self.db_path)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_get_suggestions(self):
        suggestions = self.agent.get_proactive_suggestions()
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_suggestions_with_preferences(self):
        await self.agent.add_preference(
            category="focus",
            preference="morning focus time",
            strength=0.9
        )

        suggestions = self.agent.get_proactive_suggestions()
        # Should consider the preference
        assert isinstance(suggestions, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
