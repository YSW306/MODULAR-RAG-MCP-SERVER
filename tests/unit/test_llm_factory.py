"""
Test LLM Factory - Unit tests for LLM factory and provider routing

Tests verify:
1. Factory registration and provider lookup
2. Provider creation with valid settings
3. Error handling for unknown providers
4. FakeLLM for testing purposes
"""

import pytest
from src.libs.llm.base_llm import BaseLLM, FakeLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import LLMFactory
from src.core.settings import LLMSettings


class TestFakeLLM:
    """Test FakeLLM implementation."""

    def test_fake_llm_creation(self):
        """Test creating a FakeLLM instance."""
        llm = FakeLLM()
        assert llm.provider == "fake"
        assert llm.model == "fake-model"

    def test_fake_llm_chat(self):
        """Test FakeLLM chat method returns valid response."""
        llm = FakeLLM()
        messages = [ChatMessage(role="user", content="Hello")]

        response = llm.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.model == llm.model
        assert "Hello" in response.content
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage

    def test_fake_llm_deterministic(self):
        """Test that FakeLLM returns deterministic responses."""
        llm = FakeLLM()
        messages = [ChatMessage(role="user", content="Test message")]

        response1 = llm.chat(messages)
        response2 = llm.chat(messages)

        assert response1.content == response2.content

    def test_fake_llm_repr(self):
        """Test FakeLLM string representation."""
        llm = FakeLLM(provider="fake", model="test-model")
        repr_str = repr(llm)

        assert "FakeLLM" in repr_str
        assert "fake" in repr_str
        assert "test-model" in repr_str


class TestLLMFactory:
    """Test LLMFactory for provider registration and creation."""

    def test_fake_provider_registered(self):
        """Test that FakeLLM provider is registered by default."""
        assert "fake" in LLMFactory.list_providers()

    def test_factory_create_fake_provider(self):
        """Test creating FakeLLM through factory."""
        settings = LLMSettings(provider="fake", model="fake-model")

        llm = LLMFactory.create(settings)

        assert isinstance(llm, FakeLLM)
        assert llm.provider == "fake"
        assert llm.model == "fake-model"

    def test_factory_unknown_provider_error(self):
        """Test that factory raises error for unknown provider."""
        settings = LLMSettings(provider="unknown_provider", model="some-model")

        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create(settings)

        assert "unknown_provider" in str(exc_info.value)
        assert "not registered" in str(exc_info.value)

    def test_factory_provider_case_insensitive(self):
        """Test that provider name matching is case-insensitive."""
        settings = LLMSettings(provider="FAKE", model="test-model")

        llm = LLMFactory.create(settings)

        assert isinstance(llm, FakeLLM)

    def test_factory_register_custom_provider(self):
        """Test registering a custom provider."""

        class CustomLLM(BaseLLM):
            """Custom LLM for testing."""

            def chat(self, messages, temperature=0.7, max_tokens=None, **kwargs):
                return ChatResponse(
                    content="Custom response",
                    model=self.model,
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                )

            def __repr__(self):
                return f"CustomLLM({self.model})"

        # Register custom provider
        LLMFactory.register("custom", CustomLLM)

        # Create instance through factory
        settings = LLMSettings(provider="custom", model="custom-model")
        llm = LLMFactory.create(settings)

        assert isinstance(llm, CustomLLM)
        assert llm.provider == "custom"

    def test_factory_list_providers(self):
        """Test listing all registered providers."""
        providers = LLMFactory.list_providers()

        assert isinstance(providers, list)
        assert "fake" in providers

    def test_factory_get_provider_class(self):
        """Test retrieving provider class by name."""
        provider_class = LLMFactory.get_provider_class("fake")

        assert provider_class == FakeLLM

    def test_factory_get_nonexistent_provider_class(self):
        """Test that getting nonexistent provider returns None."""
        provider_class = LLMFactory.get_provider_class("nonexistent")

        assert provider_class is None

    def test_factory_invalid_provider_class_registration(self):
        """Test that registering non-BaseLLM class raises error."""

        class NotAnLLM:
            pass

        with pytest.raises(ValueError) as exc_info:
            LLMFactory.register("invalid", NotAnLLM)

        assert "must inherit from BaseLLM" in str(exc_info.value)


class TestChatMessage:
    """Test ChatMessage dataclass."""

    def test_chat_message_creation(self):
        """Test creating a ChatMessage."""
        msg = ChatMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"


class TestChatResponse:
    """Test ChatResponse dataclass."""

    def test_chat_response_creation(self):
        """Test creating a ChatResponse."""
        response = ChatResponse(
            content="Response text",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            stop_reason="stop",
        )

        assert response.content == "Response text"
        assert response.model == "test-model"
        assert response.usage["prompt_tokens"] == 10
        assert response.stop_reason == "stop"


class TestLLMFactoryIntegration:
    """Integration tests for LLM factory."""

    def test_factory_create_and_chat(self):
        """Test creating an LLM and using it to chat."""
        settings = LLMSettings(provider="fake", model="test-model")

        llm = LLMFactory.create(settings)
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="Hello, how are you?"),
        ]

        response = llm.chat(messages, temperature=0.5, max_tokens=100)

        assert isinstance(response, ChatResponse)
        assert len(response.content) > 0
        assert response.usage["prompt_tokens"] > 0
