# Provider Registry for Context Rot Research
# Created: 2025-12-18
#
# Implements a decorator-based registry pattern for provider classes.
# This eliminates the need to edit multiple files when adding new providers.
#
# Usage:
#   @register("openai")
#   class OpenAIProvider(BaseProvider): ...
#
#   # Multiple aliases supported:
#   @register("gptoss", "ollama", "local")
#   class GptOssProvider(BaseProvider): ...

from typing import Dict, Type, List, Optional, Any
from .base_provider import BaseProvider

_REGISTRY: Dict[str, Type[BaseProvider]] = {}
_ALIASES: Dict[str, str] = {}  # Maps alias -> canonical name


def register(*names: str):
    """
    Decorator to register a provider class under one or more names.

    The first name is the canonical name; subsequent names are aliases.

    Example:
        @register("gptoss", "ollama", "local")
        class GptOssProvider(BaseProvider):
            pass

        # All these work:
        get_provider("gptoss")
        get_provider("ollama")
        get_provider("local")
    """
    def decorator(cls: Type[BaseProvider]) -> Type[BaseProvider]:
        if not names:
            raise ValueError(f"@register requires at least one name for {cls.__name__}")

        canonical = names[0].lower()
        _REGISTRY[canonical] = cls

        # Register aliases pointing to canonical name
        for alias in names[1:]:
            _ALIASES[alias.lower()] = canonical

        return cls
    return decorator


def get_provider(name: str, **kwargs) -> BaseProvider:
    """
    Factory function to instantiate a provider by name.

    Args:
        name: Provider name or alias (case-insensitive)
        **kwargs: Arguments passed to provider constructor

    Returns:
        Instantiated provider

    Raises:
        ValueError: If provider name is not registered
    """
    key = name.lower()

    # Resolve alias to canonical name
    if key in _ALIASES:
        key = _ALIASES[key]

    if key not in _REGISTRY:
        available = ", ".join(sorted(available_providers()))
        raise ValueError(
            f"Unknown provider: '{name}'. "
            f"Available providers: {available}"
        )

    return _REGISTRY[key](**kwargs)


def available_providers() -> List[str]:
    """Return list of all registered provider names (excluding aliases)."""
    return list(_REGISTRY.keys())


def all_names() -> List[str]:
    """Return all registered names including aliases."""
    return list(_REGISTRY.keys()) + list(_ALIASES.keys())


def provider_info() -> Dict[str, Dict[str, Any]]:
    """
    Return detailed info about all registered providers.

    Returns dict mapping canonical name to:
        - class: The provider class
        - aliases: List of alias names
        - docstring: First line of class docstring
    """
    info = {}
    for name, cls in _REGISTRY.items():
        aliases = [a for a, canonical in _ALIASES.items() if canonical == name]
        doc = (cls.__doc__ or "").strip().split('\n')[0]
        info[name] = {
            "class": cls,
            "aliases": aliases,
            "docstring": doc
        }
    return info


# Auto-import providers to trigger registration
# This must be at the bottom after all definitions
def _auto_register():
    """Import all provider modules to trigger @register decorators."""
    import importlib
    import pkgutil
    from pathlib import Path

    providers_dir = Path(__file__).parent / "providers"
    if not providers_dir.exists():
        return

    for module_info in pkgutil.iter_modules([str(providers_dir)]):
        if not module_info.name.startswith('_'):
            try:
                importlib.import_module(f".providers.{module_info.name}", package=__package__)
            except ImportError as e:
                print(f"Warning: Could not import provider {module_info.name}: {e}")


# Run auto-registration on module load
_auto_register()
