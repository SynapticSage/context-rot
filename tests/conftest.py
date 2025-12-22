# Shared pytest fixtures for Context Rot tests
# Created: 2025-12-22

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add experiments to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_input_df():
    """Create sample input DataFrame for provider tests."""
    return pd.DataFrame({
        'prompt': ['Test prompt 1', 'Test prompt 2', 'Test prompt 3'],
        'token_count': [100, 200, 150],
        'max_output_tokens': [50, 50, 50],
    })


@pytest.fixture
def sample_output_df():
    """Create sample output DataFrame with some completed rows."""
    return pd.DataFrame({
        'token_count': [100, 200, 150],
        'output': ['Result 1', None, 'ERROR: test error'],
        '_error_count': [0, 0, 1],
    })


@pytest.fixture
def sample_niah_df():
    """Create sample NIAH experiment DataFrame."""
    return pd.DataFrame({
        'prompt': ['prompt 1', 'prompt 2'],
        'token_count': [1000, 2000],
        'needle_depth': [50, 75],
        'trial': [0, 0],
        'question': ['What is X?', 'What is Y?'],
        'answer': ['Answer X', 'Answer Y'],
    })


@pytest.fixture
def mock_litellm_response():
    """Create mock LiteLLM response object."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Test response content"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response._hidden_params = {'response_cost': 0.001}
    return response


@pytest.fixture
def mock_litellm_empty_response():
    """Create mock LiteLLM response with empty content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = ""
    response.choices[0].finish_reason = "stop"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 0
    return response


@pytest.fixture
def sample_haystack_texts(temp_dir):
    """Create sample text files for haystack creation."""
    texts = [
        "This is the first essay. It has multiple sentences. The content is about testing.",
        "This is the second essay. It discusses different topics. Testing is important.",
        "Third essay content here. More sentences follow. Testing continues.",
    ]
    for i, text in enumerate(texts):
        with open(os.path.join(temp_dir, f"essay_{i}.txt"), 'w') as f:
            f.write(text)
    return temp_dir


@pytest.fixture
def sample_distractors_json(temp_dir):
    """Create sample distractors JSON file."""
    import json
    distractors = {
        "1": {"rewrite_for_analysis": "Distractor one content"},
        "2": {"rewrite_for_analysis": "Distractor two content"},
    }
    path = os.path.join(temp_dir, "distractors.json")
    with open(path, 'w') as f:
        json.dump(distractors, f)
    return path


@pytest.fixture
def clean_registry():
    """Reset the provider registry before each test."""
    from experiments.models.registry import _REGISTRY, _ALIASES
    original_registry = _REGISTRY.copy()
    original_aliases = _ALIASES.copy()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(original_registry)
    _ALIASES.clear()
    _ALIASES.update(original_aliases)
