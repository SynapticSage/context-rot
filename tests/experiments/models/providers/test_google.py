# Tests for Google Vertex AI Provider
# Created: 2025-12-22

import pytest
from unittest.mock import patch, MagicMock
from experiments.models.providers.google import GoogleProvider


class TestGoogleProviderInit:
    """Test GoogleProvider initialization."""

    def test_get_client_returns_none(self):
        """Test that get_client returns None (LiteLLM doesn't need client)."""
        provider = GoogleProvider()
        assert provider.get_client() is None

    def test_inherits_from_base_provider(self):
        """Test that GoogleProvider inherits from BaseProvider."""
        from experiments.models.base_provider import BaseProvider
        provider = GoogleProvider()
        assert isinstance(provider, BaseProvider)


class TestProcessSinglePrompt:
    """Test the process_single_prompt method."""

    @patch('experiments.models.providers.google.litellm')
    def test_successful_completion(self, mock_litellm, mock_litellm_response):
        """Test successful API completion."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = GoogleProvider()
        index, response = provider.process_single_prompt(
            prompt="Test prompt",
            model_name="gemini-pro",
            max_output_tokens=100,
            index=42,
        )

        assert index == 42
        assert response == "Test response content"

    @patch('experiments.models.providers.google.litellm')
    def test_adds_vertex_prefix(self, mock_litellm, mock_litellm_response):
        """Test that vertex_ai/ prefix is added to model name."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = GoogleProvider()
        provider.process_single_prompt(
            prompt="My prompt",
            model_name="gemini-pro",
            max_output_tokens=500,
            index=0,
        )

        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs['model'] == "vertex_ai/gemini-pro"

    @patch('experiments.models.providers.google.litellm')
    def test_passes_vertex_config(self, mock_litellm, mock_litellm_response):
        """Test that Vertex project/location are passed."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = GoogleProvider()
        provider.process_single_prompt(
            prompt="Test",
            model_name="gemini-pro",
            max_output_tokens=100,
            index=0,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        assert 'vertex_project' in call_kwargs
        assert 'vertex_location' in call_kwargs

    @patch('experiments.models.providers.google.litellm')
    def test_handles_no_choices(self, mock_litellm):
        """Test handling of response with no choices."""
        response = MagicMock()
        response.choices = []
        mock_litellm.completion.return_value = response

        provider = GoogleProvider()
        index, response = provider.process_single_prompt(
            prompt="Test",
            model_name="gemini-pro",
            max_output_tokens=100,
            index=0,
        )

        assert index == 0
        assert response == "ERROR_NO_CONTENT"

    @patch('experiments.models.providers.google.litellm')
    def test_handles_exception(self, mock_litellm):
        """Test handling of API exception."""
        mock_litellm.completion.side_effect = Exception("Vertex API Error")

        provider = GoogleProvider()
        index, response = provider.process_single_prompt(
            prompt="Test",
            model_name="gemini-pro",
            max_output_tokens=100,
            index=5,
        )

        assert index == 5
        assert response.startswith("ERROR:")
        assert "Vertex API Error" in response

    @patch('experiments.models.providers.google.litellm')
    def test_tracks_tokens_when_tracker_available(self, mock_litellm, mock_litellm_response):
        """Test that tokens are tracked when tracker is set."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = GoogleProvider()
        provider.token_tracker = MagicMock()

        provider.process_single_prompt(
            prompt="Test",
            model_name="gemini-pro",
            max_output_tokens=100,
            index=0,
        )

        provider.token_tracker.track_call.assert_called_once()


class TestRegistration:
    """Test that Google provider is properly registered."""

    def test_registered_with_google_name(self):
        """Test that provider is registered as 'google'."""
        from experiments.models.registry import get_provider
        provider = get_provider("google")
        assert isinstance(provider, GoogleProvider)

    def test_registered_with_vertex_alias(self):
        """Test that provider is registered with 'vertex' alias."""
        from experiments.models.registry import get_provider
        provider = get_provider("vertex")
        assert isinstance(provider, GoogleProvider)

    def test_registered_with_gemini_alias(self):
        """Test that provider is registered with 'gemini' alias."""
        from experiments.models.registry import get_provider
        provider = get_provider("gemini")
        assert isinstance(provider, GoogleProvider)
