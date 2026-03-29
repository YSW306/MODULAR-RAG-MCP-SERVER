"""
LLM Factory Module - Factory Pattern for LLM Provider Creation

Implements the Factory Pattern to create and manage LLM instances based on
configuration settings. Supports multiple providers (OpenAI, Azure, Ollama, etc.)
"""

from typing import Dict, Type, Optional
from src.core.settings import LLMSettings
from .base_llm import BaseLLM, FakeLLM


class LLMFactory:
    """
    Factory for creating LLM provider instances.

    This factory implements the Factory Pattern to:
    1. Register different LLM provider implementations
    2. Create appropriate provider instances based on settings
    3. Support provider switching via configuration changes
    4. Provide fallback to FakeLLM for testing
    """

    # Registry of available LLM providers
    _providers: Dict[str, Type[BaseLLM]] = {}

    @classmethod
    def register(cls, provider_name: str, provider_class: Type[BaseLLM]) -> None:
        """
        Register a new LLM provider.

        Args:
            provider_name: Name of the provider (e.g., "openai", "azure")
            provider_class: The BaseLLM subclass implementing this provider
        """
        if not issubclass(provider_class, BaseLLM):
            raise ValueError(
                f"Provider class {provider_class} must inherit from BaseLLM"
            )
        cls._providers[provider_name.lower()] = provider_class

    @classmethod
    def create(cls, settings: LLMSettings) -> BaseLLM:
        """
        Create an LLM instance based on settings.

        Args:
            settings: LLMSettings object containing provider, model, and credentials

        Returns:
            An instance of the requested LLM provider

        Raises:
            ValueError: If provider is not registered
            RuntimeError: If provider initialization fails
        """
        provider_name = settings.provider.lower()

        # Check if provider is registered
        if provider_name not in cls._providers:
            registered = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. "
                f"Registered providers: {registered}"
            )

        provider_class = cls._providers[provider_name]

        try:
            # Create provider instance with settings
            return provider_class(
                provider=provider_name,
                model=settings.model,
                api_key=settings.api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                azure_endpoint=settings.azure_endpoint,
                api_version=settings.api_version,
                deployment_name=settings.deployment_name,
                base_url=settings.base_url,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create LLM provider {provider_name}: {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list:
        """
        Get list of all registered providers.

        Returns:
            List of registered provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def get_provider_class(cls, provider_name: str) -> Optional[Type[BaseLLM]]:
        """
        Get the class for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            The provider class, or None if not registered
        """
        return cls._providers.get(provider_name.lower())


# Register FakeLLM for testing (always available)
LLMFactory.register("fake", FakeLLM)
