"""Tests for llm_ocr.core.base module."""

from typing import List

import pytest

from llm_ocr.core.base import LLMProvider


class ConcreteProvider(LLMProvider):
    """Concrete implementation of LLMProvider for testing."""

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.closed = False

    async def generate_content(self, prompt: str, image_bytes_list: List[bytes]) -> str:
        """Mock implementation."""
        return "Generated content"

    async def aclose(self):
        """Mock cleanup."""
        self.closed = True

    @property
    def name(self) -> str:
        """Return provider name."""
        return "concrete"


class TestLLMProvider:
    """Tests for LLMProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider("api-key", "model")

    def test_concrete_implementation_init(self):
        """Test initialization of concrete implementation."""
        provider = ConcreteProvider("test-key", "test-model", extra_param="value")
        assert provider.api_key == "test-key"
        assert provider.model == "test-model"
        assert provider.kwargs == {"extra_param": "value"}

    def test_concrete_implementation_name(self):
        """Test name property of concrete implementation."""
        provider = ConcreteProvider("test-key", "test-model")
        assert provider.name == "concrete"

    @pytest.mark.asyncio
    async def test_concrete_implementation_generate_content(self):
        """Test generate_content method of concrete implementation."""
        provider = ConcreteProvider("test-key", "test-model")
        result = await provider.generate_content("test prompt", [b"image data"])
        assert result == "Generated content"

    @pytest.mark.asyncio
    async def test_concrete_implementation_aclose(self):
        """Test aclose method of concrete implementation."""
        provider = ConcreteProvider("test-key", "test-model")
        assert provider.closed is False
        await provider.aclose()
        assert provider.closed is True

    def test_kwargs_empty_by_default(self):
        """Test that kwargs is empty dict when no extra params provided."""
        provider = ConcreteProvider("test-key", "test-model")
        assert provider.kwargs == {}

    def test_kwargs_with_multiple_params(self):
        """Test that multiple kwargs are stored correctly."""
        provider = ConcreteProvider(
            "test-key", "test-model", param1="value1", param2=42, param3=True
        )
        assert provider.kwargs == {"param1": "value1", "param2": 42, "param3": True}


class IncompleteProvider(LLMProvider):
    """Incomplete implementation missing required methods."""

    @property
    def name(self) -> str:
        return "incomplete"


class TestAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_missing_generate_content_raises_error(self):
        """Test that missing generate_content raises TypeError."""
        with pytest.raises(TypeError, match="generate_content"):
            # This should fail because generate_content and aclose are not implemented
            IncompleteProvider("key", "model")

    def test_all_abstract_methods_required(self):
        """Test that all abstract methods must be implemented."""
        # Try to instantiate incomplete provider
        with pytest.raises(TypeError) as exc_info:
            IncompleteProvider("key", "model")

        # Check that error mentions abstract methods
        error_msg = str(exc_info.value)
        assert "abstract" in error_msg.lower()
