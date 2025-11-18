# Adaptive AI Agent

A fully functional personalized AI agent system that learns and adapts to individual users over time.

## Features

### Core Capabilities

- **Hierarchical Memory System**: Four types of memory (episodic, semantic, procedural, preference) with TF-IDF based vector similarity search
- **Multi-Modal Learning**: Learns from explicit corrections, implicit signals, and behavioral patterns
- **Adaptive Personality**: Adjusts communication style based on user interactions
- **Meta-Reasoning Engine**: Decides when to act autonomously, suggest, ask for clarification, or silently learn
- **Value Alignment**: Validates actions against explicit rules, preferences, and ethical frameworks
- **Catastrophic Forgetting Prevention**: Protects important knowledge with rehearsal scheduling

### What Makes It "10x"

1. **Real Persistence**: SQLite backend stores all memories, preferences, and rules
2. **Working Vector Search**: TF-IDF based similarity search for relevant memory retrieval
3. **Conflict Resolution**: Handles contradictory learning signals intelligently
4. **Spaced Repetition**: Memory rehearsal system based on importance
5. **Full CLI Interface**: Interactive session for testing all features

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd adaptive-agent

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Usage

### Interactive Mode

```bash
python -m src.main --user myuser

# Or with custom database
python -m src.main --user myuser --db my_agent.db
```

### CLI Commands

```
/profile       - Show learned personality profile
/preferences   - List all learned preferences
/rules         - List all defined rules
/add-rule      - Add a new explicit rule
/add-pref      - Add a new preference
/correct       - Correct a past behavior
/store         - Store new knowledge
/recall        - Search memories
/consolidate   - Run memory consolidation
/quit          - Exit
```

### Programmatic Usage

```python
import asyncio
from src.core.central_system import CentralNervousSystem
from src.models.base import Context, Stimulus
from src.models.feedback import FeedbackType, EmotionalTone

async def main():
    # Initialize agent
    agent = CentralNervousSystem(user_id="alice")

    # Add a rule
    await agent.add_rule(
        condition="morning focus time",
        action="don't schedule meetings before 11 AM"
    )

    # Add a preference
    await agent.add_preference(
        category="communication",
        preference="concise responses without jargon",
        strength=0.9
    )

    # Correct a behavior
    await agent.correct_behavior(
        wrong_behavior="scheduled meeting at 9 AM",
        correct_behavior="protect morning focus time",
        emotional_tone=EmotionalTone.FRUSTRATED
    )

    # Process a stimulus
    context = Context(user_id="alice")
    stimulus = Stimulus(
        type="scheduling",
        data={
            "request": "schedule meeting",
            "time": "10 AM",
            "importance": "medium"
        },
        context=context
    )

    result = await agent.process_stimulus(stimulus)
    print(result.explanation)

    # Get profile summary
    profile = await agent.get_profile_summary()
    print(profile)

asyncio.run(main())
```

## Architecture

```
src/
├── core/
│   └── central_system.py      # Main orchestrator
├── models/
│   ├── base.py                # Base models (Context, Stimulus)
│   ├── memory.py              # Memory entries, preferences, rules
│   ├── feedback.py            # Feedback events and signals
│   ├── learning.py            # Learning updates and patterns
│   └── action.py              # Decisions and action results
├── storage/
│   ├── database.py            # SQLite connection
│   └── repositories.py        # Data access layer
├── memory/
│   ├── hierarchical.py        # Multi-type memory system
│   ├── vector_store.py        # TF-IDF similarity search
│   └── preference_graph.py    # Preference relationships
├── feedback/
│   ├── explicit.py            # Direct corrections, rules
│   ├── implicit.py            # Ignored/modified suggestions
│   └── behavioral.py          # Pattern detection
├── learning/
│   ├── multi_modal.py         # Combines all feedback types
│   ├── integrator.py          # Conflict resolution
│   ├── knowledge_updater.py   # Apply updates to storage
│   └── forgetting_prevention.py # Memory protection
├── reasoning/
│   └── meta_reasoning.py      # Decision engine
├── personality/
│   └── adaptive.py            # Style adaptation
├── alignment/
│   └── value_alignment.py     # Rule/preference validation
├── execution/
│   └── transparent.py         # Action execution
└── main.py                    # CLI entry point
```

## How Learning Works

### Explicit Feedback (High Confidence)
- Direct corrections: "That's wrong, do this instead"
- Rule definitions: "Always/Never do X in situation Y"
- Preference statements: "I prefer X over Y"

### Implicit Signals (Medium Confidence)
- Suggestions ignored: Tracks what user doesn't want
- Suggestions modified: Learns what to keep/remove
- Engagement patterns: Time spent, response depth

### Behavioral Patterns (Lower Confidence)
- Time-of-day preferences
- Communication style shifts
- Interaction duration patterns

### Conflict Resolution
1. Explicit > Implicit
2. Specific > General
3. Recent patterns weighted against historical consistency

## Example Scenario

```python
# User explicitly sets focus time protection
await agent.add_rule(
    "9-11 AM",
    "protect as focus time, no meetings"
)

# Agent receives meeting request at 10 AM
stimulus = Stimulus(
    type="schedule_conflict",
    data={
        "new_meeting": "high_importance",
        "requested_time": "10 AM"
    },
    context=context
)

result = await agent.process_stimulus(stimulus)

# Result:
# **Action**: Declined the meeting
# **Decision Type**: Autonomous Execute
# **Confidence**: 85%
# **Key Factors**:
#   - Matches rule: protect 9-11 AM as focus time
#   - Historical preference for morning focus
#   - Low intrusion - acting on explicit rule
```

## Testing

```bash
# Run the interactive CLI
python -m src.main --user testuser

# In the CLI:
/add-rule
> Condition: morning hours
> Action: keep responses brief

/add-pref
> Category: communication
> Preference: no emoji
> Strength: 0.9

/correct
> What was wrong: sent long detailed response
> What should have happened: send brief bullet points

/profile
# See updated personality traits

/recall morning meetings
# Search for relevant memories
```

## License

MIT
