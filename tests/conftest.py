"""Pytest configuration and fixtures."""

import pytest
import asyncio
import os


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    db_path = f"test_temp_{os.getpid()}.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def sample_context():
    """Create a sample context for testing."""
    from src.models.base import Context
    return Context(
        user_id="test_user",
        user_state={"busy": False},
        environment={"location": "office"}
    )


@pytest.fixture
def sample_stimulus(sample_context):
    """Create a sample stimulus for testing."""
    from src.models.base import Stimulus
    return Stimulus(
        type="question",
        data={"message": "What time is the meeting?"},
        context=sample_context
    )
