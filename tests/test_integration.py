"""Integration tests for llm-ocr package."""

import os
from unittest.mock import MagicMock, patch

import pytest

from llm_ocr import LLMOCR, Gemini, OCRConfig, OpenAI


class TestPackageImports:
    """Test that all public APIs can be imported."""

    def test_import_main_classes(self):
        """Test importing main classes from package root."""
        from llm_ocr import LLMOCR, Gemini, LLMProvider, OCRConfig, OpenAI

        assert LLMOCR is not None
        assert OpenAI is not None
        assert Gemini is not None
        assert OCRConfig is not None
        assert LLMProvider is not None

    def test_package_version(self):
        """Test that package version is defined."""
        from llm_ocr import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_package_all(self):
        """Test that __all__ is properly defined."""
        from llm_ocr import __all__

        expected_exports = [
            "LLMOCR",
            "OCRConfig",
            "LLMProvider",
            "Gemini",
            "OpenAI",
        ]
        for export in expected_exports:
            assert export in __all__


class TestOpenAIIntegration:
    """Integration tests for OpenAI provider with LLMOCR."""

    def test_openai_initialization_with_llmocr(self, mock_api_key):
        """Test OpenAI provider can be initialized with LLMOCR."""
        with patch("llm_ocr.providers.openai.AsyncOpenAI"):
            provider = OpenAI(api_key=mock_api_key)
            ocr = LLMOCR(provider)

            assert ocr.provider == provider
            assert isinstance(ocr.config, OCRConfig)

    def test_openai_with_custom_config(self, mock_api_key):
        """Test OpenAI provider with custom OCRConfig."""
        with patch("llm_ocr.providers.openai.AsyncOpenAI"):
            provider = OpenAI(api_key=mock_api_key, model=OpenAI.GPT_4O)
            config = OCRConfig(dpi=300, max_pages=10)
            ocr = LLMOCR(provider, config)

            assert ocr.config.dpi == 300
            assert ocr.config.max_pages == 10
            assert provider.model == OpenAI.GPT_4O

    @pytest.mark.asyncio
    async def test_openai_async_context_manager(self, mock_api_key, mock_openai_client):
        """Test OpenAI provider with LLMOCR as async context manager."""
        with patch(
            "llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider = OpenAI(api_key=mock_api_key)

            async with LLMOCR(provider) as ocr:
                assert isinstance(ocr, LLMOCR)

            mock_openai_client.close.assert_called_once()


class TestGeminiIntegration:
    """Integration tests for Gemini provider with LLMOCR."""

    def test_gemini_initialization_with_llmocr(self, mock_api_key):
        """Test Gemini provider can be initialized with LLMOCR."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key)
            ocr = LLMOCR(provider)

            assert ocr.provider == provider
            assert isinstance(ocr.config, OCRConfig)

    def test_gemini_with_custom_config(self, mock_api_key):
        """Test Gemini provider with custom OCRConfig."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key, model=Gemini.PRO_2_5)
            config = OCRConfig(dpi=150, llm_batch_size=3)
            ocr = LLMOCR(provider, config)

            assert ocr.config.dpi == 150
            assert ocr.config.llm_batch_size == 3
            assert provider.model == Gemini.PRO_2_5

    @pytest.mark.asyncio
    async def test_gemini_async_context_manager(self, mock_api_key):
        """Test Gemini provider with LLMOCR as async context manager."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider = Gemini(api_key=mock_api_key)

            async with LLMOCR(provider) as ocr:
                assert isinstance(ocr, LLMOCR)

            # Gemini's aclose is a no-op, so just verify it doesn't error


class TestEndToEndWorkflow:
    """End-to-end workflow tests (mocked)."""

    @pytest.mark.asyncio
    async def test_complete_pdf_processing_workflow_openai(
        self, tmp_path, mock_api_key, mock_openai_client
    ):
        """Test complete PDF to markdown workflow with OpenAI."""
        with patch(
            "llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_openai_client
        ):
            # Setup
            provider = OpenAI(api_key=mock_api_key)
            config = OCRConfig(dpi=150, max_pages=2)

            pdf_path = tmp_path / "test.pdf"
            output_path = tmp_path / "output.md"

            # Mock PDF processing
            mock_info = {"Pages": 2}

            async def mock_iter(self, pdf_path):
                yield 1, b"page1_bytes"
                yield 2, b"page2_bytes"

            with patch(
                "llm_ocr.core.processor.pdfinfo_from_path", return_value=mock_info
            ):
                with patch(
                    "llm_ocr.core.processor.LLMOCR._iter_pdf_pages_as_png_bytes",
                    mock_iter,
                ):
                    pdf_path.write_text("dummy")

                    async with LLMOCR(provider, config) as ocr:
                        result = await ocr.convert(pdf_path, output_path)

                    # Verify
                    assert output_path.exists()
                    content = output_path.read_text()
                    assert len(content) > 0
                    assert content == result

    @pytest.mark.asyncio
    async def test_complete_pdf_processing_workflow_gemini(
        self, tmp_path, mock_api_key, mock_gemini_client
    ):
        """Test complete PDF to markdown workflow with Gemini."""
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = mock_gemini_client
            mock_client_class.return_value = mock_client

            # Setup
            provider = Gemini(api_key=mock_api_key)
            config = OCRConfig(dpi=200, max_pages=1)

            pdf_path = tmp_path / "test.pdf"
            output_path = tmp_path / "output.md"

            # Mock PDF processing
            mock_info = {"Pages": 1}

            async def mock_iter(self, pdf_path):
                yield 1, b"page1_bytes"

            with patch(
                "llm_ocr.core.processor.pdfinfo_from_path", return_value=mock_info
            ):
                with patch(
                    "llm_ocr.core.processor.LLMOCR._iter_pdf_pages_as_png_bytes",
                    mock_iter,
                ):
                    pdf_path.write_text("dummy")

                    async with LLMOCR(provider, config) as ocr:
                        await ocr.convert(pdf_path, output_path)

                    # Verify
                    assert output_path.exists()
                    content = output_path.read_text()
                    assert len(content) > 0


class TestProviderSwitching:
    """Test switching between providers."""

    @pytest.mark.asyncio
    async def test_switch_providers(self, mock_api_key, mock_openai_client):
        """Test that the same LLMOCR config works with different providers."""
        config = OCRConfig(dpi=200, max_pages=5)

        # Test with OpenAI
        with patch(
            "llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider1 = OpenAI(api_key=mock_api_key)
            ocr1 = LLMOCR(provider1, config)
            assert ocr1.provider.name == "openai"

        # Test with Gemini
        with patch("llm_ocr.providers.gemini.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client_class.return_value = mock_client

            provider2 = Gemini(api_key=mock_api_key)
            ocr2 = LLMOCR(provider2, config)
            assert ocr2.provider.name == "gemini"


@pytest.mark.requires_api
class TestRealAPIIntegration:
    """Tests that require real API keys (skipped by default)."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    @pytest.mark.asyncio
    async def test_real_openai_api(self, sample_image_bytes):
        """Test with real OpenAI API (requires valid API key)."""
        provider = OpenAI()  # Uses env var

        result = await provider.generate_content(
            "Describe this image briefly", [sample_image_bytes]
        )

        assert isinstance(result, str)
        assert len(result) > 0

        await provider.aclose()

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    )
    @pytest.mark.asyncio
    async def test_real_gemini_api(self, sample_image_bytes):
        """Test with real Gemini API (requires valid API key)."""
        provider = Gemini()  # Uses env var

        result = await provider.generate_content(
            "Describe this image briefly", [sample_image_bytes]
        )

        assert isinstance(result, str)
        assert len(result) > 0

        await provider.aclose()
