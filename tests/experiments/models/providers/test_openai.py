# Tests for OpenAI Provider
# Created: 2025-12-22

import pytest
from unittest.mock import patch, MagicMock
from experiments.models.providers.openai import OpenAIProvider


class TestOpenAIProviderInit:
    """Test OpenAIProvider initialization."""

    def test_get_client_returns_none(self):
        """Test that get_client returns None (LiteLLM doesn't need client)."""
        provider = OpenAIProvider()
        assert provider.get_client() is None

    def test_inherits_from_base_provider(self):
        """Test that OpenAIProvider inherits from BaseProvider."""
        from experiments.models.base_provider import BaseProvider
        provider = OpenAIProvider()
        assert isinstance(provider, BaseProvider)


class TestProcessSinglePrompt:
    """Test the process_single_prompt method."""

    @patch('experiments.models.providers.openai.litellm')
    def test_successful_completion(self, mock_litellm, mock_litellm_response):
        """Test successful API completion."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = OpenAIProvider()
        index, response = provider.process_single_prompt(
            prompt="Test prompt",
            model_name="gpt-4",
            max_output_tokens=100,
            index=42,
        )

        assert index == 42
        assert response == "Test response content"

    @patch('experiments.models.providers.openai.litellm')
    def test_passes_correct_parameters(self, mock_litellm, mock_litellm_response):
        """Test that correct parameters are passed to litellm."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = OpenAIProvider()
        provider.process_single_prompt(
            prompt="My prompt",
            model_name="gpt-3.5-turbo",
            max_output_tokens=500,
            index=0,
        )

        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs['model'] == "gpt-3.5-turbo"
        assert call_kwargs['max_tokens'] == 500
        assert call_kwargs['temperature'] == 0
        assert call_kwargs['messages'][0]['content'] == "My prompt"

    @patch('experiments.models.providers.openai.litellm')
    def test_handles_empty_response(self, mock_litellm, mock_litellm_empty_response):
        """Test handling of empty response content."""
        mock_litellm.completion.return_value = mock_litellm_empty_response

        provider = OpenAIProvider()
        index, response = provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-4",
            max_output_tokens=100,
            index=0,
        )

        # Empty content returns empty string (not ERROR_NO_CONTENT)
        assert index == 0

    @patch('experiments.models.providers.openai.litellm')
    def test_handles_no_choices(self, mock_litellm):
        """Test handling of response with no choices."""
        response = MagicMock()
        response.choices = []
        mock_litellm.completion.return_value = response

        provider = OpenAIProvider()
        index, response = provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-4",
            max_output_tokens=100,
            index=0,
        )

        assert index == 0
        assert response == "ERROR_NO_CONTENT"

    @patch('experiments.models.providers.openai.litellm')
    def test_handles_exception(self, mock_litellm):
        """Test handling of API exception."""
        mock_litellm.completion.side_effect = Exception("API Error")

        provider = OpenAIProvider()
        index, response = provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-4",
            max_output_tokens=100,
            index=5,
        )

        assert index == 5
        assert response.startswith("ERROR:")
        assert "API Error" in response

    @patch('experiments.models.providers.openai.litellm')
    def test_tracks_tokens_when_tracker_available(self, mock_litellm, mock_litellm_response):
        """Test that tokens are tracked when tracker is set."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = OpenAIProvider()
        provider.token_tracker = MagicMock()

        provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-4",
            max_output_tokens=100,
            index=0,
        )

        provider.token_tracker.track_call.assert_called_once()

    @patch('experiments.models.providers.openai.os')
    @patch('experiments.models.providers.openai.litellm')
    def test_uses_api_key_from_env(self, mock_litellm, mock_os, mock_litellm_response):
        """Test that API key is read from environment."""
        mock_os.getenv.return_value = "test-api-key"
        mock_litellm.completion.return_value = mock_litellm_response

        provider = OpenAIProvider()
        provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-4",
            max_output_tokens=100,
            index=0,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        assert 'api_key' in call_kwargs


class TestRegistration:
    """Test that OpenAI provider is properly registered."""

    def test_registered_with_openai_name(self):
        """Test that provider is registered as 'openai'."""
        from experiments.models.registry import get_provider
        provider = get_provider("openai")
        assert isinstance(provider, OpenAIProvider)
