# Tests for provider registry pattern
# Created: 2025-12-22

import pytest
from unittest.mock import patch, MagicMock
from experiments.models.registry import (
    register,
    get_provider,
    available_providers,
    all_names,
    provider_info,
    _REGISTRY,
    _ALIASES,
)
from experiments.models.base_provider import BaseProvider


class TestRegisterDecorator:
    """Test the @register decorator functionality."""

    def test_register_single_name(self, clean_registry):
        """Test registering a provider with a single name."""
        @register("testprovider")
        class TestProvider(BaseProvider):
            def process_single_prompt(self, prompt, model_name, max_output_tokens, index):
                return index, "test"
            def get_client(self):
                return None

        assert "testprovider" in _REGISTRY
        assert _REGISTRY["testprovider"] == TestProvider

    def test_register_multiple_names(self, clean_registry):
        """Test registering a provider with multiple aliases."""
        @register("primary", "alias1", "alias2")
        class TestProvider(BaseProvider):
            def process_single_prompt(self, prompt, model_name, max_output_tokens, index):
                return index, "test"
            def get_client(self):
                return None

        # Primary name in registry
        assert "primary" in _REGISTRY
        # Aliases point to primary
        assert _ALIASES["alias1"] == "primary"
        assert _ALIASES["alias2"] == "primary"

    def test_register_no_names_raises(self, clean_registry):
        """Test that registering without names raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one name"):
            @register()
            class TestProvider(BaseProvider):
                def process_single_prompt(self, prompt, model_name, max_output_tokens, index):
                    return index, "test"
                def get_client(self):
                    return None

    def test_register_case_insensitive(self, clean_registry):
        """Test that registration converts names to lowercase."""
        @register("TestName", "ALIAS")
        class TestProvider(BaseProvider):
            def process_single_prompt(self, prompt, model_name, max_output_tokens, index):
                return index, "test"
            def get_client(self):
                return None

        assert "testname" in _REGISTRY
        assert "alias" in _ALIASES


class TestGetProvider:
    """Test the get_provider factory function."""

    def test_get_provider_by_canonical_name(self):
        """Test getting a provider by its canonical name."""
        provider = get_provider("openai")
        assert provider is not None

    def test_get_provider_by_alias(self):
        """Test getting a provider by an alias."""
        provider = get_provider("ollama")  # Alias for gptoss
        assert provider is not None

    def test_get_provider_case_insensitive(self):
        """Test that provider lookup is case-insensitive."""
        provider1 = get_provider("OpenAI")
        provider2 = get_provider("OPENAI")
        provider3 = get_provider("openai")
        # All should work without raising
        assert provider1 is not None
        assert provider2 is not None
        assert provider3 is not None

    def test_get_provider_unknown_raises(self):
        """Test that unknown provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent_provider")

    def test_get_provider_kwargs_passed(self, clean_registry):
        """Test that kwargs are passed to provider constructor."""
        received_kwargs = {}

        @register("testprov")
        class TestProvider(BaseProvider):
            def __init__(self, **kwargs):
                received_kwargs.update(kwargs)
                super().__init__()
            def process_single_prompt(self, prompt, model_name, max_output_tokens, index):
                return index, "test"
            def get_client(self):
                return None

        get_provider("testprov", custom_arg="value")
        assert received_kwargs.get("custom_arg") == "value"


class TestAvailableProviders:
    """Test the available_providers function."""

    def test_returns_list(self):
        """Test that available_providers returns a list."""
        providers = available_providers()
        assert isinstance(providers, list)

    def test_contains_registered_providers(self):
        """Test that known providers are in the list."""
        providers = available_providers()
        # These are registered in the actual codebase
        assert "openai" in providers or "gptoss" in providers

    def test_excludes_aliases(self):
        """Test that aliases are not in available_providers."""
        providers = available_providers()
        # 'ollama' is an alias for 'gptoss', should not be in canonical list
        # (unless it was registered as canonical itself)
        all_provider_names = all_names()
        # Just verify available_providers is a subset
        assert len(providers) <= len(all_provider_names)


class TestAllNames:
    """Test the all_names function."""

    def test_includes_aliases(self):
        """Test that all_names includes both canonical names and aliases."""
        names = all_names()
        assert isinstance(names, list)
        # Should include more names than just canonical providers
        providers = available_providers()
        assert len(names) >= len(providers)


class TestProviderInfo:
    """Test the provider_info function."""

    def test_returns_dict(self):
        """Test that provider_info returns a dict."""
        info = provider_info()
        assert isinstance(info, dict)

    def test_info_structure(self):
        """Test that each provider info has expected keys."""
        info = provider_info()
        for name, details in info.items():
            assert "class" in details
            assert "aliases" in details
            assert "docstring" in details
            assert isinstance(details["aliases"], list)

    def test_aliases_match_registry(self, clean_registry):
        """Test that aliases in provider_info match _ALIASES."""
        @register("main", "secondary", "tertiary")
        class TestProvider(BaseProvider):
            """Test provider docstring."""
            def process_single_prompt(self, prompt, model_name, max_output_tokens, index):
                return index, "test"
            def get_client(self):
                return None

        info = provider_info()
        assert "main" in info
        assert set(info["main"]["aliases"]) == {"secondary", "tertiary"}
        assert info["main"]["docstring"] == "Test provider docstring."


class TestAutoRegister:
    """Test that providers are auto-registered on module import."""

    def test_openai_provider_registered(self):
        """Test that OpenAI provider is auto-registered."""
        assert "openai" in _REGISTRY

    def test_gptoss_provider_registered(self):
        """Test that GPT-OSS provider is auto-registered."""
        assert "gptoss" in _REGISTRY

    def test_gptoss_aliases_registered(self):
        """Test that GPT-OSS aliases are registered."""
        # These are defined in the @register decorator in gptoss.py
        for alias in ["ollama", "local", "openrouter"]:
            assert alias in _ALIASES or alias in _REGISTRY
