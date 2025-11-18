#!/usr/bin/env python3
"""
Adaptive AI Agent - Command Line Interface

A fully functional personalized AI agent that learns and adapts to users.
"""

import asyncio
import argparse
import json
from datetime import datetime

from .core.central_system import CentralNervousSystem
from .models.base import Context, Stimulus, ActionType
from .models.action import ActionPlan
from .models.feedback import FeedbackType, EmotionalTone


class AgentCLI:
    """Command-line interface for the adaptive agent."""

    def __init__(self, user_id: str, db_path: str = "agent_memory.db"):
        self.agent = CentralNervousSystem(user_id, db_path)
        self.user_id = user_id

    async def run_interactive(self):
        """Run interactive CLI session."""
        print("\n" + "="*60)
        print("  Adaptive AI Agent - Interactive Mode")
        print("="*60)
        print(f"\nUser: {self.user_id}")
        print("\nCommands:")
        print("  /help          - Show all commands")
        print("  /profile       - Show learned profile")
        print("  /preferences   - List preferences")
        print("  /rules         - List rules")
        print("  /add-rule      - Add a new rule")
        print("  /add-pref      - Add a new preference")
        print("  /correct       - Correct a behavior")
        print("  /store         - Store knowledge")
        print("  /recall        - Recall memories")
        print("  /consolidate   - Run memory consolidation")
        print("  /quit          - Exit")
        print("\nOr type any message to interact with the agent.\n")

        while True:
            try:
                user_input = input(f"[{self.user_id}] > ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                else:
                    await self._handle_message(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                break

    async def _handle_command(self, command: str):
        """Handle CLI commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._show_help()

        elif cmd == "/profile":
            await self._show_profile()

        elif cmd == "/preferences":
            await self._show_preferences()

        elif cmd == "/rules":
            await self._show_rules()

        elif cmd == "/add-rule":
            await self._add_rule_interactive()

        elif cmd == "/add-pref":
            await self._add_preference_interactive()

        elif cmd == "/correct":
            await self._correct_behavior_interactive()

        elif cmd == "/store":
            await self._store_knowledge_interactive()

        elif cmd == "/recall":
            await self._recall_memories(args)

        elif cmd == "/consolidate":
            await self._consolidate()

        elif cmd in ["/quit", "/exit", "/q"]:
            print("Goodbye!")
            raise KeyboardInterrupt

        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands.")

    async def _handle_message(self, message: str):
        """Handle a regular message as a stimulus."""

        # Detect message type
        if "?" in message:
            stimulus_type = "question"
        elif any(word in message.lower() for word in ["schedule", "meeting", "calendar"]):
            stimulus_type = "scheduling"
        elif any(word in message.lower() for word in ["remind", "remember", "note"]):
            stimulus_type = "reminder"
        else:
            stimulus_type = "general"

        # Create context
        context = Context(user_id=self.user_id)

        # Create stimulus
        stimulus = Stimulus(
            type=stimulus_type,
            data={"message": message, "raw_input": message},
            context=context
        )

        # Process through agent
        print("\nProcessing...")
        result = await self.agent.process_stimulus(stimulus)

        # Display result
        print("\n" + "-"*40)
        if result.explanation:
            print(result.explanation)
        else:
            print(f"Action completed: {result.success}")

        if result.result_data:
            print(f"\nDetails: {json.dumps(result.result_data, indent=2, default=str)}")
        print("-"*40 + "\n")

    def _show_help(self):
        """Show detailed help."""
        print("\n" + "="*60)
        print("  Available Commands")
        print("="*60)
        print("""
  /profile
      Show the learned personality profile for the current user.
      Displays traits like warmth, formality, detail orientation.

  /preferences
      List all learned preferences with their strength scores.

  /rules
      List all explicit rules defined for this user.

  /add-rule
      Add a new explicit rule. You'll be prompted for:
      - Condition (when should this apply?)
      - Action (what should be done?)

  /add-pref
      Add a new explicit preference. You'll be prompted for:
      - Category (e.g., communication, scheduling)
      - Preference description
      - Strength (0.0 to 1.0)

  /correct
      Correct a past behavior. You'll be prompted for:
      - What was wrong
      - What should have happened

  /store
      Store new knowledge in memory. You'll be prompted for:
      - Content to store
      - Memory type (episodic, semantic, procedural)
      - Importance (0.0 to 1.0)

  /recall <query>
      Search memories for relevant content.
      Example: /recall morning meetings

  /consolidate
      Run memory consolidation to merge similar memories
      and strengthen important ones.

  /quit or /exit
      Exit the interactive session.
""")

    async def _show_profile(self):
        """Show user profile summary."""
        summary = await self.agent.get_profile_summary()

        print("\n" + "="*60)
        print("  User Profile Summary")
        print("="*60)

        print("\nPersonality Traits:")
        for trait, value in summary["personality"].items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {trait:20s} [{bar}] {value:.2f}")

        print(f"\nPreferences: {summary['preferences_count']} learned")
        if summary['top_preferences']:
            print("  Top preferences:")
            for pref in summary['top_preferences']:
                print(f"    - {pref['preference']} (strength: {pref['strength']:.2f})")

        print(f"\nRules: {summary['rules_count']} defined")
        print(f"Success Rate: {summary['success_rate']:.0%}")

        protection = summary['knowledge_protection']
        print(f"\nKnowledge Protection:")
        print(f"  Protected items: {protection['protected_count']}")
        print(f"  Critical items: {protection['critical_count']}")
        print(f"  Scheduled rehearsals: {protection['scheduled_rehearsals']}")
        print()

    async def _show_preferences(self):
        """Show all preferences."""
        prefs = self.agent.pref_repo.get_by_user(self.user_id)

        print("\n" + "="*60)
        print("  Learned Preferences")
        print("="*60 + "\n")

        if not prefs:
            print("  No preferences learned yet.\n")
            return

        for pref in prefs:
            strength_bar = "█" * int(pref.strength * 10) + "░" * (10 - int(pref.strength * 10))
            print(f"  [{pref.category}] {pref.preference}")
            print(f"    Strength: [{strength_bar}] {pref.strength:.2f}")
            print(f"    Source: {pref.source}")
            print()

    async def _show_rules(self):
        """Show all rules."""
        rules = self.agent.rule_repo.get_by_user(self.user_id)

        print("\n" + "="*60)
        print("  Defined Rules")
        print("="*60 + "\n")

        if not rules:
            print("  No rules defined yet.\n")
            return

        for i, rule in enumerate(rules, 1):
            print(f"  Rule {i}:")
            print(f"    IF: {rule.condition}")
            print(f"    THEN: {rule.action}")
            print(f"    Strength: {rule.strength:.2f}")
            print()

    async def _add_rule_interactive(self):
        """Add a new rule interactively."""
        print("\n--- Add New Rule ---")
        condition = input("  Condition (when should this apply?): ").strip()
        action = input("  Action (what should be done?): ").strip()

        if condition and action:
            await self.agent.add_rule(condition, action)
            print(f"\n  ✓ Rule added: IF '{condition}' THEN '{action}'\n")
        else:
            print("\n  ✗ Cancelled - both condition and action required.\n")

    async def _add_preference_interactive(self):
        """Add a new preference interactively."""
        print("\n--- Add New Preference ---")
        category = input("  Category (e.g., communication, scheduling): ").strip()
        preference = input("  Preference description: ").strip()
        strength_str = input("  Strength (0.0-1.0, default 0.8): ").strip()

        try:
            strength = float(strength_str) if strength_str else 0.8
        except ValueError:
            strength = 0.8

        if category and preference:
            await self.agent.add_preference(category, preference, strength)
            print(f"\n  ✓ Preference added: [{category}] {preference}\n")
        else:
            print("\n  ✗ Cancelled - category and preference required.\n")

    async def _correct_behavior_interactive(self):
        """Correct a behavior interactively."""
        print("\n--- Correct Behavior ---")
        wrong = input("  What was wrong: ").strip()
        correct = input("  What should have happened: ").strip()
        tone_str = input("  Your tone (neutral/frustrated/pleased): ").strip().lower()

        tone_map = {
            "frustrated": EmotionalTone.FRUSTRATED,
            "pleased": EmotionalTone.PLEASED,
            "neutral": EmotionalTone.NEUTRAL
        }
        tone = tone_map.get(tone_str, EmotionalTone.NEUTRAL)

        if wrong and correct:
            await self.agent.correct_behavior(wrong, correct, tone)
            print(f"\n  ✓ Correction recorded and learned.\n")
        else:
            print("\n  ✗ Cancelled - both fields required.\n")

    async def _store_knowledge_interactive(self):
        """Store knowledge interactively."""
        print("\n--- Store Knowledge ---")
        content = input("  Content to store: ").strip()
        mem_type = input("  Type (episodic/semantic/procedural): ").strip().lower()
        importance_str = input("  Importance (0.0-1.0, default 0.5): ").strip()

        if mem_type not in ["episodic", "semantic", "procedural"]:
            mem_type = "semantic"

        try:
            importance = float(importance_str) if importance_str else 0.5
        except ValueError:
            importance = 0.5

        tags_str = input("  Tags (comma-separated, optional): ").strip()
        tags = [t.strip() for t in tags_str.split(",")] if tags_str else []

        if content:
            await self.agent.store_knowledge(content, mem_type, importance, tags)
            print(f"\n  ✓ Knowledge stored as {mem_type} memory.\n")
        else:
            print("\n  ✗ Cancelled - content required.\n")

    async def _recall_memories(self, query: str):
        """Recall memories matching a query."""
        if not query:
            query = input("  Search query: ").strip()

        if not query:
            print("  No query provided.\n")
            return

        print(f"\n  Searching for: '{query}'...")

        results = await self.agent.memory.retrieve(
            user_id=self.user_id,
            query=query,
            top_k=5
        )

        print("\n" + "-"*40)
        if not results:
            print("  No memories found.")
        else:
            for i, (memory, score) in enumerate(results, 1):
                print(f"\n  {i}. [{memory.memory_type.value}] (score: {score:.2f})")
                print(f"     {memory.content[:100]}...")
                print(f"     Importance: {memory.importance:.2f}, Accessed: {memory.access_count}x")
        print("-"*40 + "\n")

    async def _consolidate(self):
        """Run memory consolidation."""
        print("\n  Running memory consolidation...")
        await self.agent.consolidate_memory()
        print("  ✓ Consolidation complete.\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive AI Agent - A personalized learning agent"
    )
    parser.add_argument(
        "--user", "-u",
        default="default_user",
        help="User ID for the agent"
    )
    parser.add_argument(
        "--db", "-d",
        default="agent_memory.db",
        help="Database file path"
    )

    args = parser.parse_args()

    cli = AgentCLI(args.user, args.db)
    await cli.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
