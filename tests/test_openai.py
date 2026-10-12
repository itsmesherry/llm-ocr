"""Tests for llm_ocr.providers.openai module."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_ocr.providers.openai import OpenAI


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with explicit API key."""
        with patch("llm_ocr.providers.openai.AsyncOpenAI") as mock_client:
            provider = OpenAI(api_key=mock_api_key)
            assert provider.api_key == mock_api_key
            assert provider.model == OpenAI.GPT_4O_MINI
            mock_client.assert_called_once_with(api_key=mock_api_key)

    def test_init_with_env_api_key(self, mock_api_key, set_openai_env_key):
        """Test initialization with API key from environment."""
        with patch("llm_ocr.providers.openai.AsyncOpenAI") as mock_client:
            provider = OpenAI()
            assert provider.api_key == mock_api_key
            mock_client.assert_called_once_with(api_key=mock_api_key)

    def test_init_without_api_key_raises_error(self, unset_env_keys):
        """Test initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAI()

    def test_init_with_base_url_and_no_api_key(self, unset_env_keys):
        """Test initialization with base_url defaults api_key to 'not-needed'."""
        with patch("llm_ocr.providers.openai.AsyncOpenAI") as mock_client:
            provider = OpenAI(base_url="http://localhost:11434/v1", model="llava")
            assert provider.api_key == "not-needed"
            assert provider.base_url == "http://localhost:11434/v1"
            assert provider.model == "llava"
            mock_client.assert_called_once_with(
                api_key="not-needed", base_url="http://localhost:11434/v1"
            )

    def test_init_with_env_base_url(self, unset_env_keys, monkeypatch):
        """Test initialization with OPENAI_BASE_URL environment variable."""
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        with patch("llm_ocr.providers.openai.AsyncOpenAI") as mock_client:
            provider = OpenAI(api_key="custom-key")
            assert provider.base_url == "http://localhost:8000/v1"
            mock_client.assert_called_once_with(
                api_key="custom-key", base_url="http://localhost:8000/v1"
            )

    def test_init_with_custom_model(self, mock_api_key):
        """Test initialization with custom model."""
        with patch("llm_ocr.providers.openai.AsyncOpenAI"):
            provider = OpenAI(api_key=mock_api_key, model=OpenAI.GPT_4O)
            assert provider.model == OpenAI.GPT_4O

    def test_init_with_kwargs(self, mock_api_key):
        """Test initialization with additional kwargs."""
        with patch("llm_ocr.providers.openai.AsyncOpenAI"):
            provider = OpenAI(
                api_key=mock_api_key,
                model=OpenAI.GPT_4O,
                max_tokens=500,
                temperature=0.7,
            )
            assert provider.kwargs == {"max_tokens": 500, "temperature": 0.7}

    def test_name_property(self, mock_api_key):
        """Test name property returns 'openai'."""
        with patch("llm_ocr.providers.openai.AsyncOpenAI"):
            provider = OpenAI(api_key=mock_api_key)
            assert provider.name == "openai"

    def test_model_constants(self):
        """Test that model constants are defined correctly."""
        assert OpenAI.GPT_5 == "gpt-5"
        assert OpenAI.GPT_5_MINI == "gpt-5-mini"
        assert OpenAI.GPT_5_NANO == "gpt-5-nano"
        assert OpenAI.GPT_5_CHAT_LATEST == "gpt-5-chat-latest"
        assert OpenAI.GPT_4_1 == "gpt-4.1"
        assert OpenAI.GPT_4_1_MINI == "gpt-4.1-mini"
        assert OpenAI.GPT_4_1_NANO == "gpt-4.1-nano"
        assert OpenAI.O4_MINI == "o4-mini"
        assert OpenAI.O1 == "o1"
        assert OpenAI.O3 == "o3"
        assert OpenAI.O3_MINI == "o3-mini"
        assert OpenAI.GPT_4O == "gpt-4o"
        assert OpenAI.GPT_4O_MINI == "gpt-4o-mini"
        assert OpenAI.CHATGPT_4O_LATEST == "chatgpt-4o-latest"

    @pytest.mark.asyncio
    async def test_generate_content_single_image(
        self, mock_api_key, sample_image_bytes, mock_openai_client
    ):
        """Test generate_content with a single image."""
        with patch(
            "llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider = OpenAI(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )

            assert result == "# Test Markdown\nSample content"
            mock_openai_client.chat.completions.create.assert_called_once()

            # Verify the call structure
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == OpenAI.GPT_4O_MINI

            messages = call_args.kwargs["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

            content = messages[0]["content"]
            assert len(content) == 2  # text + 1 image
            assert content[0]["type"] == "text"
            assert content[0]["text"] == "Test prompt"
            assert content[1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_generate_content_multiple_images(
        self, mock_api_key, sample_image_bytes_list, mock_openai_client
    ):
        """Test generate_content with multiple images."""
        with patch(
            "llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider = OpenAI(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", sample_image_bytes_list
            )

            assert result == "# Test Markdown\nSample content"

            # Verify multiple images were included
            call_args = mock_openai_client.chat.completions.create.call_args
            content = call_args.kwargs["messages"][0]["content"]
            assert len(content) == 3  # text + 2 images

    @pytest.mark.asyncio
    async def test_generate_content_with_kwargs(
        self, mock_api_key, sample_image_bytes, mock_openai_client
    ):
        """Test generate_content passes kwargs to API."""
        with patch(
            "llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider = OpenAI(api_key=mock_api_key, max_tokens=100, temperature=0.5)
            await provider.generate_content("Test prompt", [sample_image_bytes])

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args.kwargs["max_tokens"] == 100
            assert call_args.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_content_empty_response(
        self, mock_api_key, sample_image_bytes
    ):
        """Test generate_content with empty response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAI(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_content_none_content(
        self, mock_api_key, sample_image_bytes
    ):
        """Test generate_content when message content is None."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAI(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_content_api_error(self, mock_api_key, sample_image_bytes):
        """Test generate_content handles API errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        with patch("llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAI(api_key=mock_api_key)
            with pytest.raises(Exception, match="API Error"):
                await provider.generate_content("Test prompt", [sample_image_bytes])

    @pytest.mark.asyncio
    async def test_image_encoding(self, mock_api_key, sample_image_bytes):
        """Test that images are correctly base64 encoded."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "test"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAI(api_key=mock_api_key)
            await provider.generate_content("Test prompt", [sample_image_bytes])

            call_args = mock_client.chat.completions.create.call_args
            content = call_args.kwargs["messages"][0]["content"]
            image_url = content[1]["image_url"]["url"]

            # Verify it's a data URL with base64 encoding
            assert image_url.startswith("data:image/png;base64,")

            # Verify the base64 part can be decoded
            b64_data = image_url.split(",")[1]
            decoded = base64.b64decode(b64_data)
            assert decoded == sample_image_bytes

    @pytest.mark.asyncio
    async def test_aclose(self, mock_api_key, mock_openai_client):
        """Test aclose method."""
        with patch(
            "llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider = OpenAI(api_key=mock_api_key)
            await provider.aclose()
            mock_openai_client.close.assert_called_once()
