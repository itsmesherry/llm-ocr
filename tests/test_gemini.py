"""Tests for llm_ocr.providers.gemini module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_ocr.providers.gemini import Gemini


class TestGeminiProvider:
    """Tests for Gemini provider."""

    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with explicit API key."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key)
            assert provider.api_key == mock_api_key
            assert provider.model == Gemini.FLASH_2_5
            mock_client_class.assert_called_once_with(api_key=mock_api_key)

    def test_init_with_env_api_key(self, mock_api_key, set_gemini_env_key):
        """Test initialization with API key from environment."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider = Gemini()
            assert provider.api_key == mock_api_key
            mock_client_class.assert_called_once_with(api_key=mock_api_key)

    def test_init_without_api_key_raises_error(self, unset_env_keys):
        """Test initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="Gemini API key is required"):
            Gemini()

    def test_init_with_custom_model(self, mock_api_key):
        """Test initialization with custom model."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key, model=Gemini.PRO_2_5)
            assert provider.model == Gemini.PRO_2_5

    def test_init_with_kwargs(self, mock_api_key):
        """Test initialization with additional kwargs."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider = Gemini(
                api_key=mock_api_key, model=Gemini.PRO_2_5, temperature=0.7, top_p=0.9
            )
            assert provider.kwargs == {"temperature": 0.7, "top_p": 0.9}

    def test_name_property(self, mock_api_key):
        """Test name property returns 'gemini'."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key)
            assert provider.name == "gemini"

    def test_model_constants(self):
        """Test that model constants are defined correctly."""
        assert Gemini.FLASH_2_5 == "gemini-2.5-flash"
        assert Gemini.PRO_2_5 == "gemini-2.5-pro"
        assert Gemini.FLASH_LITE_2_5 == "gemini-2.5-flash-lite"
        assert Gemini.FLASH_2_0 == "gemini-2.0-flash"
        assert Gemini.FLASH_LITE_2_0 == "gemini-2.0-flash-lite"
        assert Gemini.PRO_2_0 == "gemini-2.0-pro"
        assert Gemini.FLASH_1_5 == "gemini-1.5-flash"
        assert Gemini.PRO_1_5 == "gemini-1.5-pro"

    @pytest.mark.asyncio
    async def test_generate_content_single_image(
        self, mock_api_key, sample_image_bytes, mock_gemini_client
    ):
        """Test generate_content with a single image."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = mock_gemini_client
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )

            assert result == "# Test Markdown\nSample content"
            mock_gemini_client.models.generate_content.assert_called_once()

            # Verify the call structure
            call_args = mock_gemini_client.models.generate_content.call_args
            assert call_args.kwargs["model"] == Gemini.FLASH_2_5

            contents = call_args.kwargs["contents"]
            assert len(contents) == 2  # text + 1 image
            assert contents[0] == "Test prompt"

    @pytest.mark.asyncio
    async def test_generate_content_multiple_images(
        self, mock_api_key, sample_image_bytes_list, mock_gemini_client
    ):
        """Test generate_content with multiple images."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = mock_gemini_client
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", sample_image_bytes_list
            )

            assert result == "# Test Markdown\nSample content"

            # Verify multiple images were included
            call_args = mock_gemini_client.models.generate_content.call_args
            contents = call_args.kwargs["contents"]
            assert len(contents) == 3  # text + 2 images

    @pytest.mark.asyncio
    async def test_generate_content_with_kwargs(
        self, mock_api_key, sample_image_bytes, mock_gemini_client
    ):
        """Test generate_content passes kwargs to API."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = mock_gemini_client
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key, temperature=0.5, top_p=0.9)
            await provider.generate_content("Test prompt", [sample_image_bytes])

            call_args = mock_gemini_client.models.generate_content.call_args
            assert call_args.kwargs["temperature"] == 0.5
            assert call_args.kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_generate_content_empty_response(
        self, mock_api_key, sample_image_bytes
    ):
        """Test generate_content with empty response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_async_client = MagicMock()
        mock_async_client.models.generate_content = AsyncMock(
            return_value=mock_response
        )
        mock_client.aio = mock_async_client

        with patch("llm_ocr.providers.gemini.Client", return_value=mock_client):
            provider = Gemini(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_content_none_text(self, mock_api_key, sample_image_bytes):
        """Test generate_content when response text is None."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_async_client = MagicMock()
        mock_async_client.models.generate_content = AsyncMock(
            return_value=mock_response
        )
        mock_client.aio = mock_async_client

        with patch("llm_ocr.providers.gemini.Client", return_value=mock_client):
            provider = Gemini(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_content_value_error(self, mock_api_key, sample_image_bytes):
        """Test generate_content handles ValueError when accessing text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Simulate ValueError when accessing .text
        type(mock_response).text = property(
            lambda self: (_ for _ in ()).throw(ValueError("No text"))
        )
        mock_async_client = MagicMock()
        mock_async_client.models.generate_content = AsyncMock(
            return_value=mock_response
        )
        mock_client.aio = mock_async_client

        with patch("llm_ocr.providers.gemini.Client", return_value=mock_client):
            provider = Gemini(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_content_attribute_error(
        self, mock_api_key, sample_image_bytes
    ):
        """Test generate_content handles AttributeError when accessing text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Delete text attribute to simulate AttributeError
        del mock_response.text
        mock_async_client = MagicMock()
        mock_async_client.models.generate_content = AsyncMock(
            return_value=mock_response
        )
        mock_client.aio = mock_async_client

        with patch("llm_ocr.providers.gemini.Client", return_value=mock_client):
            provider = Gemini(api_key=mock_api_key)
            result = await provider.generate_content(
                "Test prompt", [sample_image_bytes]
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_content_api_error(self, mock_api_key, sample_image_bytes):
        """Test generate_content handles API errors."""
        mock_client = MagicMock()
        mock_async_client = MagicMock()
        mock_async_client.models.generate_content = AsyncMock(
            side_effect=Exception("API Error")
        )
        mock_client.aio = mock_async_client

        with patch("llm_ocr.providers.gemini.Client", return_value=mock_client):
            provider = Gemini(api_key=mock_api_key)
            with pytest.raises(Exception, match="API Error"):
                await provider.generate_content("Test prompt", [sample_image_bytes])

    @pytest.mark.asyncio
    async def test_aclose(self, mock_api_key):
        """Test aclose method (no-op for Gemini)."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key)
            await provider.aclose()  # Should not raise any errors
