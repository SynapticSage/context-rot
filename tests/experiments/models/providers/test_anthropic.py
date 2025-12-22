# Tests for Anthropic Provider
# Created: 2025-12-22

import pytest
from unittest.mock import patch, MagicMock
from experiments.models.providers.anthropic import AnthropicProvider


class TestAnthropicProviderInit:
    """Test AnthropicProvider initialization."""

    def test_get_client_returns_none(self):
        """Test that get_client returns None (LiteLLM doesn't need client)."""
        provider = AnthropicProvider()
        assert provider.get_client() is None

    def test_inherits_from_base_provider(self):
        """Test that AnthropicProvider inherits from BaseProvider."""
        from experiments.models.base_provider import BaseProvider
        provider = AnthropicProvider()
        assert isinstance(provider, BaseProvider)


class TestProcessSinglePrompt:
    """Test the process_single_prompt method."""

    @patch('experiments.models.providers.anthropic.litellm')
    def test_successful_completion(self, mock_litellm, mock_litellm_response):
        """Test successful API completion."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = AnthropicProvider()
        index, response = provider.process_single_prompt(
            prompt="Test prompt",
            model_name="claude-3-opus-20240229",
            max_output_tokens=100,
            index=42,
        )

        assert index == 42
        assert response == "Test response content"

    @patch('experiments.models.providers.anthropic.litellm')
    def test_passes_correct_parameters(self, mock_litellm, mock_litellm_response):
        """Test that correct parameters are passed to litellm."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = AnthropicProvider()
        provider.process_single_prompt(
            prompt="My prompt",
            model_name="claude-3-sonnet",
            max_output_tokens=500,
            index=0,
        )

        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs['model'] == "claude-3-sonnet"
        assert call_kwargs['max_tokens'] == 500
        assert call_kwargs['temperature'] == 0
        assert call_kwargs['messages'][0]['content'] == "My prompt"

    @patch('experiments.models.providers.anthropic.litellm')
    def test_handles_no_choices(self, mock_litellm):
        """Test handling of response with no choices."""
        response = MagicMock()
        response.choices = []
        mock_litellm.completion.return_value = response

        provider = AnthropicProvider()
        index, response = provider.process_single_prompt(
            prompt="Test",
            model_name="claude-3-opus",
            max_output_tokens=100,
            index=0,
        )

        assert index == 0
        assert response == "ERROR_NO_CONTENT"

    @patch('experiments.models.providers.anthropic.litellm')
    def test_handles_exception(self, mock_litellm):
        """Test handling of API exception."""
        mock_litellm.completion.side_effect = Exception("API Error")

        provider = AnthropicProvider()
        index, response = provider.process_single_prompt(
            prompt="Test",
            model_name="claude-3-opus",
            max_output_tokens=100,
            index=5,
        )

        assert index == 5
        assert response.startswith("ERROR:")
        assert "API Error" in response

    @patch('experiments.models.providers.anthropic.litellm')
    def test_tracks_tokens_when_tracker_available(self, mock_litellm, mock_litellm_response):
        """Test that tokens are tracked when tracker is set."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = AnthropicProvider()
        provider.token_tracker = MagicMock()

        provider.process_single_prompt(
            prompt="Test",
            model_name="claude-3-opus",
            max_output_tokens=100,
            index=0,
        )

        provider.token_tracker.track_call.assert_called_once()


class TestRegistration:
    """Test that Anthropic provider is properly registered."""

    def test_registered_with_anthropic_name(self):
        """Test that provider is registered as 'anthropic'."""
        from experiments.models.registry import get_provider
        provider = get_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_registered_with_claude_alias(self):
        """Test that provider is registered with 'claude' alias."""
        from experiments.models.registry import get_provider
        provider = get_provider("claude")
        assert isinstance(provider, AnthropicProvider)
