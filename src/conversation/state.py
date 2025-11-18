"""Conversation state management for multi-turn interactions."""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import uuid


@dataclass
class Turn:
    """A single turn in the conversation."""
    id: str
    role: str  # "user" or "agent"
    content: str
    intent: str
    entities: Dict
    sentiment: Dict
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConversationState:
    """Current state of the conversation."""
    id: str
    user_id: str
    turns: List[Turn]
    current_topic: str
    topics_discussed: List[str]
    pending_questions: List[str]
    context_stack: List[Dict]
    last_activity: datetime
    sentiment_trend: List[float]
    unresolved_intents: List[str]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "turn_count": len(self.turns),
            "current_topic": self.current_topic,
            "topics_discussed": self.topics_discussed,
            "pending_questions": self.pending_questions,
            "sentiment_trend": self.sentiment_trend[-5:],
            "unresolved_intents": self.unresolved_intents
        }


class ConversationManager:
    """
    Manage conversation state across multiple turns.

    Features:
    - Topic tracking
    - Context stack for nested conversations
    - Sentiment trend monitoring
    - Pending question tracking
    - Conversation summarization
    """

    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
        self.max_turns = 100
        self.context_timeout = timedelta(minutes=30)

    def get_or_create(self, user_id: str) -> ConversationState:
        """Get existing conversation or create new one."""
        if user_id in self.conversations:
            conv = self.conversations[user_id]
            # Check if conversation timed out
            if datetime.utcnow() - conv.last_activity > self.context_timeout:
                # Archive old conversation and start new
                self._archive_conversation(conv)
                return self._create_new(user_id)
            return conv

        return self._create_new(user_id)

    def _create_new(self, user_id: str) -> ConversationState:
        """Create new conversation state."""
        conv = ConversationState(
            id=str(uuid.uuid4()),
            user_id=user_id,
            turns=[],
            current_topic="",
            topics_discussed=[],
            pending_questions=[],
            context_stack=[],
            last_activity=datetime.utcnow(),
            sentiment_trend=[],
            unresolved_intents=[]
        )
        self.conversations[user_id] = conv
        return conv

    def add_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        intent: str,
        entities: Dict,
        sentiment: Dict,
        metadata: Dict = None
    ) -> Turn:
        """Add a turn to the conversation."""
        conv = self.get_or_create(user_id)

        turn = Turn(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

        conv.turns.append(turn)
        conv.last_activity = turn.timestamp

        # Trim if too long
        if len(conv.turns) > self.max_turns:
            conv.turns = conv.turns[-self.max_turns:]

        # Update topic
        if role == "user":
            self._update_topic(conv, intent, entities)
            self._track_sentiment(conv, sentiment)
            self._track_pending_questions(conv, content, intent)

        return turn

    def _update_topic(self, conv: ConversationState, intent: str, entities: Dict):
        """Update current topic based on turn."""
        # Extract topic from intent and entities
        topic = intent

        # Enhance with key entities
        if entities.get("people"):
            topic += f"_with_{entities['people'][0]['value']}"
        if entities.get("times"):
            topic += "_scheduled"

        if topic != conv.current_topic:
            if conv.current_topic and conv.current_topic not in conv.topics_discussed:
                conv.topics_discussed.append(conv.current_topic)
            conv.current_topic = topic

    def _track_sentiment(self, conv: ConversationState, sentiment: Dict):
        """Track sentiment over time."""
        polarity = sentiment.get("polarity", 0.0)
        conv.sentiment_trend.append(polarity)

        # Keep last 20
        if len(conv.sentiment_trend) > 20:
            conv.sentiment_trend = conv.sentiment_trend[-20:]

    def _track_pending_questions(self, conv: ConversationState, content: str, intent: str):
        """Track questions that need answers."""
        if intent == "question" or "?" in content:
            conv.pending_questions.append(content)
            conv.unresolved_intents.append(intent)

    def resolve_question(self, user_id: str, question_index: int = 0):
        """Mark a question as resolved."""
        conv = self.get_or_create(user_id)
        if conv.pending_questions and question_index < len(conv.pending_questions):
            conv.pending_questions.pop(question_index)
        if conv.unresolved_intents and question_index < len(conv.unresolved_intents):
            conv.unresolved_intents.pop(question_index)

    def push_context(self, user_id: str, context: Dict):
        """Push a context onto the stack (for nested conversations)."""
        conv = self.get_or_create(user_id)
        conv.context_stack.append(context)

    def pop_context(self, user_id: str) -> Optional[Dict]:
        """Pop context from stack."""
        conv = self.get_or_create(user_id)
        if conv.context_stack:
            return conv.context_stack.pop()
        return None

    def get_current_context(self, user_id: str) -> Dict:
        """Get current conversation context."""
        conv = self.get_or_create(user_id)

        # Get recent turns
        recent_turns = conv.turns[-5:] if conv.turns else []

        # Calculate average sentiment
        avg_sentiment = 0.0
        if conv.sentiment_trend:
            avg_sentiment = sum(conv.sentiment_trend) / len(conv.sentiment_trend)

        # Get sentiment direction
        sentiment_direction = "stable"
        if len(conv.sentiment_trend) >= 3:
            recent = conv.sentiment_trend[-3:]
            if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                sentiment_direction = "improving"
            elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                sentiment_direction = "declining"

        return {
            "conversation_id": conv.id,
            "turn_count": len(conv.turns),
            "current_topic": conv.current_topic,
            "topics_discussed": conv.topics_discussed[-5:],
            "pending_questions": conv.pending_questions,
            "unresolved_intents": conv.unresolved_intents,
            "recent_turns": [
                {"role": t.role, "intent": t.intent, "summary": t.content[:50]}
                for t in recent_turns
            ],
            "avg_sentiment": avg_sentiment,
            "sentiment_direction": sentiment_direction,
            "context_depth": len(conv.context_stack),
            "stacked_context": conv.context_stack[-1] if conv.context_stack else None
        }

    def get_summary(self, user_id: str) -> str:
        """Generate conversation summary."""
        conv = self.get_or_create(user_id)

        if not conv.turns:
            return "No conversation history."

        # Count intents
        intent_counts = {}
        for turn in conv.turns:
            if turn.role == "user":
                intent_counts[turn.intent] = intent_counts.get(turn.intent, 0) + 1

        # Find dominant intent
        dominant_intent = max(intent_counts, key=intent_counts.get) if intent_counts else "unknown"

        # Build summary
        summary_parts = [
            f"Conversation with {len(conv.turns)} turns.",
            f"Main topic: {conv.current_topic or 'general'}.",
            f"Dominant intent: {dominant_intent}."
        ]

        if conv.topics_discussed:
            summary_parts.append(f"Also discussed: {', '.join(conv.topics_discussed[-3:])}.")

        if conv.pending_questions:
            summary_parts.append(f"{len(conv.pending_questions)} pending question(s).")

        avg_sentiment = sum(conv.sentiment_trend) / len(conv.sentiment_trend) if conv.sentiment_trend else 0
        if avg_sentiment > 0.3:
            summary_parts.append("Overall positive sentiment.")
        elif avg_sentiment < -0.3:
            summary_parts.append("Overall negative sentiment.")

        return " ".join(summary_parts)

    def should_clarify(self, user_id: str) -> Tuple[bool, str]:
        """Check if clarification is needed."""
        conv = self.get_or_create(user_id)

        # Multiple pending questions
        if len(conv.pending_questions) > 2:
            return True, f"You have {len(conv.pending_questions)} pending questions. Which would you like me to address first?"

        # Declining sentiment
        if len(conv.sentiment_trend) >= 3:
            recent = conv.sentiment_trend[-3:]
            if all(s < -0.3 for s in recent):
                return True, "I sense some frustration. Would you like me to approach this differently?"

        # Repeated same intent
        if len(conv.turns) >= 3:
            recent_intents = [t.intent for t in conv.turns[-3:] if t.role == "user"]
            if len(set(recent_intents)) == 1 and recent_intents[0] not in ["greeting", "confirmation"]:
                return True, f"We've been discussing {recent_intents[0]} for a while. Is there something specific I'm missing?"

        return False, ""

    def _archive_conversation(self, conv: ConversationState):
        """Archive old conversation (stub for future persistence)."""
        # In a full implementation, this would save to database
        pass

    def clear(self, user_id: str):
        """Clear conversation state for user."""
        if user_id in self.conversations:
            del self.conversations[user_id]


# Import for type hints
from typing import Tuple
