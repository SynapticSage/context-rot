# Tests for GPT-OSS Provider
# Created: 2025-12-22

import pytest
import os
from unittest.mock import patch, MagicMock
from experiments.models.providers.gptoss import GptOssProvider


class TestGptOssProviderInit:
    """Test GptOssProvider initialization."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_detects_openrouter_mode(self):
        """Test that OpenRouter mode is detected when API key is set."""
        provider = GptOssProvider()
        assert provider.deployment_mode == "openrouter"

    @patch.dict(os.environ, {"GPT_OSS_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_detects_local_mode(self):
        """Test that local mode is detected when base URL is set."""
        provider = GptOssProvider()
        assert provider.deployment_mode == "local"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_detects_openai_mode(self):
        """Test that OpenAI mode is detected as fallback."""
        provider = GptOssProvider()
        assert provider.deployment_mode == "openai"

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_without_configuration(self):
        """Test that ValueError is raised without any configuration."""
        with pytest.raises(ValueError, match="No GPT-OSS configuration found"):
            GptOssProvider()

    @patch.dict(os.environ, {"GPT_OSS_PREFER_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "key"}, clear=True)
    def test_respects_prefer_provider(self):
        """Test that GPT_OSS_PREFER_PROVIDER is respected."""
        provider = GptOssProvider()
        assert provider.deployment_mode == "openrouter"

    @patch.dict(os.environ, {"GPT_OSS_PREFER_PROVIDER": "invalid"}, clear=True)
    def test_rejects_invalid_prefer_provider(self):
        """Test that invalid GPT_OSS_PREFER_PROVIDER raises error."""
        with pytest.raises(ValueError, match="Invalid GPT_OSS_PREFER_PROVIDER"):
            GptOssProvider()

    @patch.dict(os.environ, {"GPT_OSS_PREFER_PROVIDER": "openrouter"}, clear=True)
    def test_validates_credentials_for_preferred(self):
        """Test that credentials are validated for preferred provider."""
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
            GptOssProvider()


class TestContextSafetyMargin:
    """Test the context safety margin."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_safety_margin_is_085(self):
        """Test that safety margin is 0.85."""
        provider = GptOssProvider()
        assert provider.get_context_safety_margin() == 0.85


class TestFormatModelName:
    """Test the _format_model_name method."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_openai_mode_formatting(self):
        """Test model name formatting in OpenAI mode."""
        provider = GptOssProvider()
        assert provider.deployment_mode == "openai"

        # Should strip provider prefix and add openai/
        assert provider._format_model_name("gpt-4") == "openai/gpt-4"
        assert provider._format_model_name("openai/gpt-4") == "openai/gpt-4"
        assert provider._format_model_name("nvidia/model") == "openai/model"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_openrouter_mode_formatting(self):
        """Test model name formatting in OpenRouter mode."""
        provider = GptOssProvider()
        assert provider.deployment_mode == "openrouter"

        # With provider prefix
        assert provider._format_model_name("openai/gpt-oss-20b") == "openrouter/openai/gpt-oss-20b"
        assert provider._format_model_name("nvidia/nemotron") == "openrouter/nvidia/nemotron"

        # Without provider prefix - assumes openai
        assert provider._format_model_name("gpt-oss-20b") == "openrouter/openai/gpt-oss-20b"

    @patch.dict(os.environ, {"GPT_OSS_BASE_URL": "http://localhost:11434/v1"}, clear=True)
    def test_ollama_mode_formatting(self):
        """Test model name formatting for Ollama (local with 11434 port)."""
        provider = GptOssProvider()
        assert provider.deployment_mode == "local"

        # Should use ollama/ prefix
        assert provider._format_model_name("llama3") == "ollama/llama3"
        assert provider._format_model_name("openai/gpt-4") == "ollama/gpt-4"

    @patch.dict(os.environ, {"GPT_OSS_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_vllm_mode_formatting(self):
        """Test model name formatting for vLLM (non-ollama local)."""
        provider = GptOssProvider()
        assert provider.deployment_mode == "local"

        # Should use openai/ prefix for vLLM/TGI
        assert provider._format_model_name("model") == "openai/model"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_strips_openrouter_prefix(self):
        """Test that openrouter/ prefix is stripped before reformatting."""
        provider = GptOssProvider()

        # Input already has openrouter prefix
        result = provider._format_model_name("openrouter/nvidia/nemotron")
        assert result == "openrouter/nvidia/nemotron"


class TestGetApiBaseForLitellm:
    """Test the _get_api_base_for_litellm method."""

    @patch.dict(os.environ, {"GPT_OSS_BASE_URL": "http://localhost:11434/v1"}, clear=True)
    def test_strips_v1_for_ollama(self):
        """Test that /v1 is stripped for Ollama."""
        provider = GptOssProvider()
        api_base = provider._get_api_base_for_litellm()
        assert api_base == "http://localhost:11434"

    @patch.dict(os.environ, {"GPT_OSS_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_keeps_v1_for_vllm(self):
        """Test that /v1 is kept for non-Ollama."""
        provider = GptOssProvider()
        api_base = provider._get_api_base_for_litellm()
        assert api_base == "http://localhost:8000/v1"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_returns_none_for_openai(self):
        """Test that None is returned for OpenAI mode."""
        provider = GptOssProvider()
        api_base = provider._get_api_base_for_litellm()
        assert api_base is None


class TestIsOllama:
    """Test the _is_ollama method."""

    @patch.dict(os.environ, {"GPT_OSS_BASE_URL": "http://localhost:11434/v1"}, clear=True)
    def test_detects_ollama_port(self):
        """Test that Ollama is detected by port 11434."""
        provider = GptOssProvider()
        assert provider._is_ollama() is True

    @patch.dict(os.environ, {"GPT_OSS_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_non_ollama_port(self):
        """Test that other ports are not Ollama."""
        provider = GptOssProvider()
        assert provider._is_ollama() is False


class TestProcessSinglePrompt:
    """Test the process_single_prompt method."""

    @patch('experiments.models.providers.gptoss.litellm')
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_successful_completion(self, mock_litellm, mock_litellm_response):
        """Test successful API completion."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = GptOssProvider()
        index, response = provider.process_single_prompt(
            prompt="Test prompt",
            model_name="openai/gpt-oss-20b",
            max_output_tokens=100,
            index=42,
        )

        assert index == 42
        assert response == "Test response content"

    @patch('experiments.models.providers.gptoss.litellm')
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_handles_empty_content(self, mock_litellm):
        """Test handling of empty response content."""
        # Create a message mock that doesn't have reasoning_content attr
        message = MagicMock(spec=['content', 'role'])
        message.content = ""

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"

        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 0
        mock_litellm.completion.return_value = response

        provider = GptOssProvider()
        index, result = provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-oss-20b",
            max_output_tokens=100,
            index=0,
        )

        assert index == 0
        assert result == "ERROR_EMPTY_CONTENT"

    @patch('experiments.models.providers.gptoss.litellm')
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_handles_exception(self, mock_litellm):
        """Test handling of API exception."""
        mock_litellm.completion.side_effect = Exception("API Error")

        provider = GptOssProvider()
        index, response = provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-oss-20b",
            max_output_tokens=100,
            index=5,
        )

        assert index == 5
        assert "ERROR_EXCEPTION" in response
        assert "API Error" in response

    @patch('experiments.models.providers.gptoss.litellm')
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_handles_no_choices(self, mock_litellm):
        """Test handling of response with no choices."""
        response = MagicMock()
        response.choices = []
        mock_litellm.completion.return_value = response

        provider = GptOssProvider()
        index, response = provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-oss-20b",
            max_output_tokens=100,
            index=0,
        )

        assert index == 0
        assert response == "ERROR_NO_CONTENT"

    @patch('experiments.models.providers.gptoss.litellm')
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_handles_reasoning_content(self, mock_litellm):
        """Test handling of reasoning_content field (o1-style models)."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = ""
        response.choices[0].message.reasoning_content = "Reasoning output"
        mock_litellm.completion.return_value = response

        provider = GptOssProvider()
        index, result = provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-oss-20b",
            max_output_tokens=100,
            index=0,
        )

        assert result == "Reasoning output"

    @patch('experiments.models.providers.gptoss.litellm')
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_tracks_tokens(self, mock_litellm, mock_litellm_response):
        """Test that tokens are tracked when tracker is available."""
        mock_litellm.completion.return_value = mock_litellm_response

        provider = GptOssProvider()
        provider.token_tracker = MagicMock()

        provider.process_single_prompt(
            prompt="Test",
            model_name="gpt-oss-20b",
            max_output_tokens=100,
            index=0,
        )

        provider.token_tracker.track_call.assert_called_once()


class TestRegistration:
    """Test that GPT-OSS provider is properly registered."""

    def test_registered_with_gptoss_name(self):
        """Test that provider is registered as 'gptoss'."""
        from experiments.models.registry import get_provider, _REGISTRY
        assert "gptoss" in _REGISTRY

    def test_aliases_registered(self):
        """Test that aliases are registered."""
        from experiments.models.registry import _ALIASES
        assert "ollama" in _ALIASES
        assert "local" in _ALIASES
        assert "openrouter" in _ALIASES


class TestGetApiKey:
    """Test the _get_api_key method."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True)
    def test_openai_mode_uses_openai_key(self):
        """Test that OpenAI mode uses OPENAI_API_KEY."""
        provider = GptOssProvider()
        assert provider.api_key == "openai-key"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}, clear=True)
    def test_openrouter_mode_uses_openrouter_key(self):
        """Test that OpenRouter mode uses OPENROUTER_API_KEY."""
        provider = GptOssProvider()
        assert provider.api_key == "router-key"

    @patch.dict(os.environ, {"GPT_OSS_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_local_mode_uses_dummy_key(self):
        """Test that local mode uses dummy key by default."""
        provider = GptOssProvider()
        assert provider.api_key == "dummy"

    @patch.dict(os.environ, {"GPT_OSS_BASE_URL": "http://localhost:8000/v1", "GPT_OSS_API_KEY": "custom-key"}, clear=True)
    def test_local_mode_uses_custom_key(self):
        """Test that local mode uses GPT_OSS_API_KEY if set."""
        provider = GptOssProvider()
        assert provider.api_key == "custom-key"
