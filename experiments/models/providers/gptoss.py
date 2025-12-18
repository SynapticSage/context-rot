# GPT-OSS Provider for Context Rot Research
# Created: 2025-11-19
# Modified: 2025-11-30 (configurable provider priority, 85% safety margin)
# Supports: OpenAI GPT-OSS models (gpt-oss-20b, gpt-oss-120b)
# Provider priority: Configurable via GPT_OSS_PREFER_PROVIDER env var
# Safety margin: 85% due to tokenizer mismatch (~15% variance vs o200k_base)

import os
import litellm
from typing import Any
from ..base_provider import BaseProvider

class GptOssProvider(BaseProvider):
    """
    Provider for OpenAI's GPT-OSS models.

    Target models:
    - openai/gpt-oss-20b (21B MoE, 3.6B active, 128k context)
    - openai/gpt-oss-120b (117B MoE, 5.1B active, 128k context)

    Provider selection (set GPT_OSS_PREFER_PROVIDER env var):
    - "openrouter": Use OpenRouter (requires OPENROUTER_API_KEY) - RECOMMENDED
    - "local": Use local deployment (requires GPT_OSS_BASE_URL)
    - "openai": Use OpenAI API directly - NOTE: GPT-OSS models NOT available here

    If GPT_OSS_PREFER_PROVIDER is not set, auto-detects based on available keys
    with priority: OpenRouter > local > OpenAI

    Note: GPT-OSS tokenizer differs from o200k_base (~15% more tokens).
    Rows exceeding context will error and be skipped (retry on resume).
    """

    # Safety margin to account for tokenizer mismatch (~15% variance)
    # GPT-OSS tokenizer counts ~15% more tokens than tiktoken/o200k_base
    # Pre-filter rows that would likely exceed API limit after tokenizer conversion
    CONTEXT_SAFETY_MARGIN = 0.85

    def __init__(self):
        """Initialize client based on environment configuration."""
        self.deployment_mode = self._detect_deployment_mode()
        self.base_url = self._get_base_url()
        self.api_key = self._get_api_key()
        print(f"[GptOssProvider] Deployment mode: {self.deployment_mode}")
        if self.base_url:
            print(f"[GptOssProvider] Base URL: {self.base_url}")
        super().__init__()

    def get_context_safety_margin(self) -> float:
        """Return safety margin (1.0 = no margin, let errors happen)."""
        return self.CONTEXT_SAFETY_MARGIN

    def _detect_deployment_mode(self) -> str:
        """Detect which deployment mode to use based on env vars.

        Set GPT_OSS_PREFER_PROVIDER to explicitly choose: openai, openrouter, local
        If not set, auto-detects with priority: OpenAI > OpenRouter > local
        """
        # Check for explicit preference
        preferred = os.getenv("GPT_OSS_PREFER_PROVIDER", "").lower()
        if preferred:
            valid_providers = {"openai", "openrouter", "local"}
            if preferred not in valid_providers:
                raise ValueError(
                    f"Invalid GPT_OSS_PREFER_PROVIDER='{preferred}'. "
                    f"Must be one of: {', '.join(valid_providers)}"
                )
            # Validate required credentials exist
            if preferred == "openai" and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("GPT_OSS_PREFER_PROVIDER=openai but OPENAI_API_KEY not set")
            if preferred == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
                raise ValueError("GPT_OSS_PREFER_PROVIDER=openrouter but OPENROUTER_API_KEY not set")
            if preferred == "local" and not os.getenv("GPT_OSS_BASE_URL"):
                raise ValueError("GPT_OSS_PREFER_PROVIDER=local but GPT_OSS_BASE_URL not set")
            return preferred

        # Auto-detect: OpenRouter > local > OpenAI
        # Note: GPT-OSS models are ONLY available on OpenRouter, not OpenAI direct
        if os.getenv("OPENROUTER_API_KEY"):
            return "openrouter"
        elif os.getenv("GPT_OSS_BASE_URL"):
            return "local"
        elif os.getenv("OPENAI_API_KEY"):
            # Warn: OpenAI direct doesn't have GPT-OSS models
            print("[GptOssProvider] WARNING: OpenAI API selected but GPT-OSS models may not be available.")
            print("[GptOssProvider] Consider setting OPENROUTER_API_KEY for GPT-OSS access.")
            return "openai"
        else:
            raise ValueError(
                "No GPT-OSS configuration found. Please set one of:\n"
                "  - OPENROUTER_API_KEY (required for GPT-OSS models)\n"
                "  - GPT_OSS_BASE_URL (for local deployment)\n"
                "Or set GPT_OSS_PREFER_PROVIDER to choose explicitly."
            )

    def _get_base_url(self) -> str:
        """Get base URL for the deployment mode."""
        if self.deployment_mode == "local":
            return os.getenv("GPT_OSS_BASE_URL")
        elif self.deployment_mode == "openrouter":
            return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return None  # OpenAI uses default

    def _get_api_key(self) -> str:
        """Get API key for the deployment mode."""
        if self.deployment_mode == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.deployment_mode == "openrouter":
            return os.getenv("OPENROUTER_API_KEY")
        return os.getenv("GPT_OSS_API_KEY", "dummy")

    def get_client(self) -> Any:
        """LiteLLM doesn't need a client object - returns None."""
        return None

    def _is_ollama(self) -> bool:
        """Check if using ollama backend."""
        return self.base_url and "11434" in self.base_url

    def _format_model_name(self, model_name: str) -> str:
        """Format model name with appropriate provider prefix for LiteLLM.

        Model name in config: openai/gpt-oss-20b
        LiteLLM requires provider prefix to route correctly:
        - OpenAI direct: openai/gpt-oss-20b
        - OpenRouter: openrouter/openai/gpt-oss-20b
        - Local ollama: ollama/gpt-oss-20b
        """
        # Strip any existing prefix to get base name
        base_name = model_name
        for prefix in ["openai/", "openrouter/", "ollama/"]:
            if model_name.startswith(prefix):
                base_name = model_name[len(prefix):]
                break

        if self.deployment_mode == "openai":
            # OpenAI direct: LiteLLM needs openai/ prefix to route
            return f"openai/{base_name}"
        elif self.deployment_mode == "openrouter":
            # OpenRouter: needs openrouter/openai/ for OpenAI models on OpenRouter
            return f"openrouter/openai/{base_name}"
        elif self.deployment_mode == "local":
            if self._is_ollama():
                return f"ollama/{base_name}"
            # For vLLM/TGI with custom base_url
            return f"openai/{base_name}"
        return f"openai/{base_name}"

    def _get_api_base_for_litellm(self) -> str:
        """Get the correct api_base for LiteLLM.

        Ollama requires base URL without /v1 suffix.
        """
        if self._is_ollama() and self.base_url:
            # Strip /v1 suffix if present for ollama
            return self.base_url.rstrip('/').replace('/v1', '')
        return self.base_url

    def process_single_prompt(
        self,
        prompt: str,
        model_name: str,
        max_output_tokens: int,
        index: int
    ) -> tuple[int, str]:
        """
        Process a single prompt using the GPT-OSS model.

        Args:
            prompt: Input text to process
            model_name: Model identifier (e.g., "gpt-oss:20b", "gpt-oss:120b")
            max_output_tokens: Maximum tokens to generate
            index: Row index for tracking

        Returns:
            Tuple of (index, response_text)
            Returns (index, "ERROR_...") on failure
        """
        try:
            # Format model name with appropriate provider prefix
            formatted_model = self._format_model_name(model_name)

            # Build litellm.completion kwargs
            kwargs = {
                "model": formatted_model,
                "temperature": 0,
                "max_tokens": max_output_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            # Add API key and base URL if configured
            if self.api_key:
                kwargs["api_key"] = self.api_key
            api_base = self._get_api_base_for_litellm()
            if api_base:
                kwargs["api_base"] = api_base

            response = litellm.completion(**kwargs)

            # Track tokens if tracker is available
            if self.token_tracker:
                self.token_tracker.track_call(response, model_name)

            # Validate response structure
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                content = choice.message.content

                # Some models put output in reasoning/thinking field instead of content
                if (content == "" or content is None) and hasattr(choice.message, 'reasoning_content'):
                    content = choice.message.reasoning_content

                # Also check for tool_calls or function_call content
                if (content == "" or content is None) and hasattr(choice.message, 'tool_calls'):
                    if choice.message.tool_calls:
                        content = str(choice.message.tool_calls)

                # Normalize content (strip whitespace, handle None)
                if content:
                    content = content.strip()

                if not content:
                    print(f"[GptOssProvider] Warning: Empty response for index {index}")
                    print(f"[GptOssProvider] Finish reason: {getattr(choice, 'finish_reason', 'unknown')}")
                    if hasattr(response, 'usage') and response.usage:
                        print(f"[GptOssProvider] Usage: {response.usage}")
                    # Debug: show all message attributes
                    msg = choice.message
                    print(f"[GptOssProvider] Message attrs: {[a for a in dir(msg) if not a.startswith('_')]}")
                    return index, "ERROR_EMPTY_CONTENT"

                return index, content
            else:
                print(f"[GptOssProvider] Error: No choices in response for index {index}")
                return index, "ERROR_NO_CONTENT"

        except Exception as e:
            error_msg = f"ERROR_EXCEPTION: {str(e)}"
            print(f"[GptOssProvider] Exception for index {index}: {e}")
            return index, error_msg
