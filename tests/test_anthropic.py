"""Tests for llm_ocr.providers.anthropic module."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_ocr.providers.anthropic import Anthropic


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with explicit API key."""
        with patch("llm_ocr.providers.anthropic.AsyncAnthropic") as mock_client:
            provider = Anthropic(api_key=mock_api_key)
            assert provider.api_key == mock_api_key
            assert provider.model == Anthropic.CLAUDE_SONNET_4
            assert provider.max_tokens == 4096
            mock_client.assert_called_once_with(api_key=mock_api_key)

    def test_init_with_env_api_key(self, mock_api_key, set_anthropic_env_key):
        """Test initialization with API key from environment."""
        with patch("llm_ocr.providers.anthropic.AsyncAnthropic") as mock_client:
            provider = Anthropic()
            assert provider.api_key == mock_api_key
            mock_client.assert_called_once_with(api_key=mock_api_key)

    def test_init_without_api_key_raises_error(self, unset_env_keys):
        """Test initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            Anthropic()

    def test_init_with_custom_model_and_max_tokens(self, mock_api_key):
        """Test initialization with custom model and max_tokens."""
        with patch("llm_ocr.providers.anthropic.AsyncAnthropic"):
            provider = Anthropic(
                api_key=mock_api_key,
                model=Anthropic.CLAUDE_3_7_SONNET,
                max_tokens=8192,
                temperature=0.0,
            )
            assert provider.model == Anthropic.CLAUDE_3_7_SONNET
            assert provider.max_tokens == 8192
            assert provider.kwargs == {"temperature": 0.0}

    def test_name_property(self, mock_api_key):
        """Test name property returns 'anthropic'."""
        with patch("llm_ocr.providers.anthropic.AsyncAnthropic"):
            provider = Anthropic(api_key=mock_api_key)
            assert provider.name == "anthropic"

    def test_model_constants(self):
        """Test that model constants are defined correctly."""
        assert Anthropic.CLAUDE_SONNET_4 == "claude-sonnet-4-20250514"
        assert Anthropic.CLAUDE_OPUS_4 == "claude-opus-4-20250514"
        assert Anthropic.CLAUDE_3_7_SONNET == "claude-3-7-sonnet-latest"
        assert Anthropic.CLAUDE_3_5_SONNET == "claude-3-5-sonnet-latest"
        assert Anthropic.CLAUDE_3_5_HAIKU == "claude-3-5-haiku-latest"

    @pytest.mark.asyncio
    async def test_generate_content_single_image(
        self, mock_api_key, sample_image_bytes, mock_anthropic_client
    ):
        """Test generate_content with a single image."""
        with patch(
            "llm_ocr.providers.anthropic.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            provider = Anthropic(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )

            assert result == "# Test Markdown\nSample content"
            mock_anthropic_client.messages.create.assert_called_once()

            call_args = mock_anthropic_client.messages.create.call_args
            assert call_args.kwargs["model"] == Anthropic.CLAUDE_SONNET_4
            assert call_args.kwargs["max_tokens"] == 4096

            messages = call_args.kwargs["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

            content = messages[0]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "image"
            assert content[0]["source"]["type"] == "base64"
            assert content[0]["source"]["media_type"] == "image/png"
            decoded = base64.b64decode(content[0]["source"]["data"])
            assert decoded == sample_image_bytes
            assert content[1] == {"type": "text", "text": "Test prompt"}

    @pytest.mark.asyncio
    async def test_generate_content_multiple_images(
        self, mock_api_key, sample_image_bytes_list, mock_anthropic_client
    ):
        """Test generate_content with multiple images."""
        with patch(
            "llm_ocr.providers.anthropic.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            provider = Anthropic(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", sample_image_bytes_list
            )
            assert result == "# Test Markdown\nSample content"

            call_args = mock_anthropic_client.messages.create.call_args
            content = call_args.kwargs["messages"][0]["content"]
            assert len(content) == 3  # 2 images + 1 text

    @pytest.mark.asyncio
    async def test_generate_content_empty_response(
        self, mock_api_key, sample_image_bytes
    ):
        """Test generate_content with empty response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch(
            "llm_ocr.providers.anthropic.AsyncAnthropic", return_value=mock_client
        ):
            provider = Anthropic(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_content_api_error(self, mock_api_key, sample_image_bytes):
        """Test generate_content handles API errors."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))

        with patch(
            "llm_ocr.providers.anthropic.AsyncAnthropic", return_value=mock_client
        ):
            provider = Anthropic(api_key=mock_api_key)
            with pytest.raises(Exception, match="API Error"):
                await provider.generate_content("Test prompt", [sample_image_bytes])

    @pytest.mark.asyncio
    async def test_aclose(self, mock_api_key, mock_anthropic_client):
        """Test aclose method."""
        with patch(
            "llm_ocr.providers.anthropic.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            provider = Anthropic(api_key=mock_api_key)
            await provider.aclose()
            mock_anthropic_client.close.assert_called_once()
