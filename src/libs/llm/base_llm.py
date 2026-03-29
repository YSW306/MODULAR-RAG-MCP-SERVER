"""
Base LLM Module - Abstract Interface for Language Models

Defines the BaseLLM abstract base class that all LLM providers must implement.
This enables pluggable LLM backends (OpenAI, Azure, Ollama, DeepSeek, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Represents a single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ChatResponse:
    """Represents an LLM response."""
    content: str
    model: str
    usage: Dict[str, int]  # {"prompt_tokens": ..., "completion_tokens": ...}
    stop_reason: Optional[str] = None


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.

    All LLM implementations (OpenAI, Azure, Ollama, etc.) must inherit from this
    and implement the required methods.
    """

    def __init__(self, provider: str, model: str, **kwargs):
        """
        Initialize the LLM provider.

        Args:
            provider: Name of the provider (e.g., "openai", "azure", "ollama")
            model: Model name/identifier
            **kwargs: Additional provider-specific configuration
        """
        self.provider = provider
        self.model = model
        self.config = kwargs

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a chat completion request to the LLM.

        Args:
            messages: List of chat messages in conversation history
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens in the response
            **kwargs: Additional parameters for the specific provider

        Returns:
            ChatResponse object containing the model's response

        Raises:
            ValueError: If input validation fails
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the LLM instance."""
        pass


class FakeLLM(BaseLLM):
    """
    Fake LLM implementation for testing purposes.

    Returns deterministic responses without making actual API calls.
    """

    def __init__(self, provider: str = "fake", model: str = "fake-model", **kwargs):
        """Initialize FakeLLM."""
        super().__init__(provider, model, **kwargs)

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Return a deterministic fake response.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature (ignored for fake)
            max_tokens: Maximum tokens (ignored for fake)
            **kwargs: Additional parameters (ignored for fake)

        Returns:
            A fixed ChatResponse for testing
        """
        # Build a deterministic response based on input
        last_message = messages[-1].content if messages else "no message"
        response_content = f"Fake response to: {last_message[:50]}"

        return ChatResponse(
            content=response_content,
            model=self.model,
            usage={
                "prompt_tokens": len(last_message.split()),
                "completion_tokens": len(response_content.split()),
            },
            stop_reason="max_tokens" if max_tokens else "stop",
        )

    def __repr__(self) -> str:
        return f"FakeLLM(provider={self.provider}, model={self.model})"
