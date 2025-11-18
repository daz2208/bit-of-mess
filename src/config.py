"""Configuration management for the adaptive agent system."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import json


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "agent_memory.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_connections: int = 5


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    max_memories_per_user: int = 10000
    embedding_dimension: int = 100
    similarity_threshold: float = 0.3
    consolidation_interval_hours: int = 8
    forgetting_protection_threshold: float = 0.8
    rehearsal_batch_size: int = 10


@dataclass
class LearningConfig:
    """Learning system configuration."""
    default_learning_rate: float = 0.3
    explicit_confidence_boost: float = 0.95
    implicit_confidence_base: float = 0.6
    min_confidence_for_update: float = 0.5
    conflict_resolution_decay: float = 0.9


@dataclass
class ConversationConfig:
    """Conversation management configuration."""
    max_turns: int = 100
    context_timeout_minutes: int = 30
    sentiment_window_size: int = 20
    clarification_threshold: int = 2


@dataclass
class NLPConfig:
    """NLP processing configuration."""
    intent_confidence_threshold: float = 0.3
    entity_confidence_threshold: float = 0.5
    sentiment_intensity_threshold: float = 0.3
    empathy_trigger_threshold: float = -0.5
    ngram_range: tuple[int, int] = (1, 2)


@dataclass
class AnalyticsConfig:
    """Analytics configuration."""
    interaction_retention_days: int = 90
    accuracy_window_size: int = 100
    trend_detection_threshold: float = 0.1


@dataclass
class ProactiveConfig:
    """Proactive suggestion configuration."""
    max_suggestions: int = 5
    suggestion_confidence_threshold: float = 0.5
    time_pattern_min_occurrences: int = 3
    stalled_goal_days: int = 7


@dataclass
class PluginConfig:
    """Plugin system configuration."""
    enabled: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str | None = None
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class AgentConfig:
    """Main configuration for the adaptive agent."""
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global settings
    debug_mode: bool = False
    version: str = "2.0.0"

    @classmethod
    def from_file(cls, path: str) -> 'AgentConfig':
        """Load configuration from a JSON file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'AgentConfig':
        """Create configuration from dictionary."""
        config = cls()

        # Update database config
        if 'database' in data:
            for key, value in data['database'].items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)

        # Update memory config
        if 'memory' in data:
            for key, value in data['memory'].items():
                if hasattr(config.memory, key):
                    setattr(config.memory, key, value)

        # Update learning config
        if 'learning' in data:
            for key, value in data['learning'].items():
                if hasattr(config.learning, key):
                    setattr(config.learning, key, value)

        # Update conversation config
        if 'conversation' in data:
            for key, value in data['conversation'].items():
                if hasattr(config.conversation, key):
                    setattr(config.conversation, key, value)

        # Update NLP config
        if 'nlp' in data:
            for key, value in data['nlp'].items():
                if hasattr(config.nlp, key):
                    setattr(config.nlp, key, value)

        # Update analytics config
        if 'analytics' in data:
            for key, value in data['analytics'].items():
                if hasattr(config.analytics, key):
                    setattr(config.analytics, key, value)

        # Update proactive config
        if 'proactive' in data:
            for key, value in data['proactive'].items():
                if hasattr(config.proactive, key):
                    setattr(config.proactive, key, value)

        # Update plugins config
        if 'plugins' in data:
            for key, value in data['plugins'].items():
                if hasattr(config.plugins, key):
                    setattr(config.plugins, key, value)

        # Update logging config
        if 'logging' in data:
            for key, value in data['logging'].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)

        # Update global settings
        if 'debug_mode' in data:
            config.debug_mode = data['debug_mode']
        if 'version' in data:
            config.version = data['version']

        return config

    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Load configuration from environment variables."""
        config = cls()

        # Database
        if 'AGENT_DB_PATH' in os.environ:
            config.database.path = os.environ['AGENT_DB_PATH']

        # Memory
        if 'AGENT_MAX_MEMORIES' in os.environ:
            config.memory.max_memories_per_user = int(os.environ['AGENT_MAX_MEMORIES'])
        if 'AGENT_SIMILARITY_THRESHOLD' in os.environ:
            config.memory.similarity_threshold = float(os.environ['AGENT_SIMILARITY_THRESHOLD'])

        # Learning
        if 'AGENT_LEARNING_RATE' in os.environ:
            config.learning.default_learning_rate = float(os.environ['AGENT_LEARNING_RATE'])

        # Conversation
        if 'AGENT_MAX_TURNS' in os.environ:
            config.conversation.max_turns = int(os.environ['AGENT_MAX_TURNS'])
        if 'AGENT_CONTEXT_TIMEOUT' in os.environ:
            config.conversation.context_timeout_minutes = int(os.environ['AGENT_CONTEXT_TIMEOUT'])

        # NLP
        if 'AGENT_INTENT_THRESHOLD' in os.environ:
            config.nlp.intent_confidence_threshold = float(os.environ['AGENT_INTENT_THRESHOLD'])

        # Logging
        if 'AGENT_LOG_LEVEL' in os.environ:
            config.logging.level = os.environ['AGENT_LOG_LEVEL']
        if 'AGENT_LOG_FILE' in os.environ:
            config.logging.file_path = os.environ['AGENT_LOG_FILE']

        # Global
        if 'AGENT_DEBUG' in os.environ:
            config.debug_mode = os.environ['AGENT_DEBUG'].lower() in ('true', '1', 'yes')

        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'database': {
                'path': self.database.path,
                'backup_enabled': self.database.backup_enabled,
                'backup_interval_hours': self.database.backup_interval_hours,
                'max_connections': self.database.max_connections
            },
            'memory': {
                'max_memories_per_user': self.memory.max_memories_per_user,
                'embedding_dimension': self.memory.embedding_dimension,
                'similarity_threshold': self.memory.similarity_threshold,
                'consolidation_interval_hours': self.memory.consolidation_interval_hours,
                'forgetting_protection_threshold': self.memory.forgetting_protection_threshold,
                'rehearsal_batch_size': self.memory.rehearsal_batch_size
            },
            'learning': {
                'default_learning_rate': self.learning.default_learning_rate,
                'explicit_confidence_boost': self.learning.explicit_confidence_boost,
                'implicit_confidence_base': self.learning.implicit_confidence_base,
                'min_confidence_for_update': self.learning.min_confidence_for_update,
                'conflict_resolution_decay': self.learning.conflict_resolution_decay
            },
            'conversation': {
                'max_turns': self.conversation.max_turns,
                'context_timeout_minutes': self.conversation.context_timeout_minutes,
                'sentiment_window_size': self.conversation.sentiment_window_size,
                'clarification_threshold': self.conversation.clarification_threshold
            },
            'nlp': {
                'intent_confidence_threshold': self.nlp.intent_confidence_threshold,
                'entity_confidence_threshold': self.nlp.entity_confidence_threshold,
                'sentiment_intensity_threshold': self.nlp.sentiment_intensity_threshold,
                'empathy_trigger_threshold': self.nlp.empathy_trigger_threshold,
                'ngram_range': self.nlp.ngram_range
            },
            'analytics': {
                'interaction_retention_days': self.analytics.interaction_retention_days,
                'accuracy_window_size': self.analytics.accuracy_window_size,
                'trend_detection_threshold': self.analytics.trend_detection_threshold
            },
            'proactive': {
                'max_suggestions': self.proactive.max_suggestions,
                'suggestion_confidence_threshold': self.proactive.suggestion_confidence_threshold,
                'time_pattern_min_occurrences': self.proactive.time_pattern_min_occurrences,
                'stalled_goal_days': self.proactive.stalled_goal_days
            },
            'plugins': {
                'enabled': self.plugins.enabled,
                'timeout_seconds': self.plugins.timeout_seconds,
                'max_retries': self.plugins.max_retries
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_path': self.logging.file_path,
                'max_file_size_mb': self.logging.max_file_size_mb,
                'backup_count': self.logging.backup_count
            },
            'debug_mode': self.debug_mode,
            'version': self.version
        }

    def save(self, path: str):
        """Save configuration to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> list[str]:
        """Validate configuration and return any errors."""
        errors = []

        # Database validation
        if not self.database.path:
            errors.append("Database path cannot be empty")
        if self.database.max_connections < 1:
            errors.append("Database max_connections must be at least 1")

        # Memory validation
        if self.memory.similarity_threshold < 0 or self.memory.similarity_threshold > 1:
            errors.append("Memory similarity_threshold must be between 0 and 1")
        if self.memory.embedding_dimension < 1:
            errors.append("Memory embedding_dimension must be at least 1")

        # Learning validation
        if self.learning.default_learning_rate <= 0 or self.learning.default_learning_rate > 1:
            errors.append("Learning default_learning_rate must be between 0 and 1")

        # Conversation validation
        if self.conversation.max_turns < 1:
            errors.append("Conversation max_turns must be at least 1")
        if self.conversation.context_timeout_minutes < 1:
            errors.append("Conversation context_timeout_minutes must be at least 1")

        # NLP validation
        if self.nlp.intent_confidence_threshold < 0 or self.nlp.intent_confidence_threshold > 1:
            errors.append("NLP intent_confidence_threshold must be between 0 and 1")

        return errors


# Global configuration instance
_config: AgentConfig | None = None


def get_config() -> AgentConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Try to load from environment first, then fall back to defaults
        _config = AgentConfig.from_env()
    return _config


def set_config(config: AgentConfig):
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(path: str) -> AgentConfig:
    """Load configuration from file and set as global."""
    global _config
    _config = AgentConfig.from_file(path)
    return _config
