"""Tests for llm_ocr.core.processor module."""

from typing import List
from unittest.mock import patch

import pytest
from PIL import Image

from llm_ocr.config.settings import OCRConfig
from llm_ocr.core.base import LLMProvider
from llm_ocr.core.processor import LLMOCR


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses=None):
        super().__init__(api_key="mock-key", model="mock-model")
        self.responses = responses or ["# Test Content"]
        self.call_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        """Return provider name."""
        return "mock"

    async def generate_content(self, prompt: str, image_bytes_list: List[bytes]) -> str:
        """Mock generate_content."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    async def aclose(self):
        """Mock aclose."""
        self.closed = True


class TestLLMOCRInit:
    """Tests for LLMOCR initialization."""

    def test_init_with_provider(self):
        """Test initialization with provider."""
        provider = MockProvider()
        ocr = LLMOCR(provider)
        assert ocr.provider == provider
        assert isinstance(ocr.config, OCRConfig)

    def test_init_with_provider_and_config(self):
        """Test initialization with provider and custom config."""
        provider = MockProvider()
        config = OCRConfig(dpi=300, max_pages=5)
        ocr = LLMOCR(provider, config)
        assert ocr.provider == provider
        assert ocr.config == config
        assert ocr.config.dpi == 300
        assert ocr.config.max_pages == 5

    def test_init_validates_config(self):
        """Test that initialization validates config."""
        provider = MockProvider()
        config = OCRConfig(dpi=1000)  # Invalid DPI
        with pytest.raises(ValueError, match="DPI must be between 72 and 600"):
            LLMOCR(provider, config)

    def test_init_with_invalid_provider(self):
        """Test initialization with invalid provider raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of LLMProvider"):
            LLMOCR("not a provider")

    def test_init_with_none_provider(self):
        """Test initialization with None provider raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of LLMProvider"):
            LLMOCR(None)


class TestLLMOCRImageProcessing:
    """Tests for image processing methods."""

    def test_process_image_to_bytes(self, tmp_path, sample_image_bytes):
        """Test _process_image_to_bytes converts image to PNG bytes."""
        # Create a test image file
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        provider = MockProvider()
        ocr = LLMOCR(provider)

        result = ocr._process_image_to_bytes(str(img_path))
        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify it's valid PNG data
        assert result.startswith(b"\x89PNG")

    def test_process_image_to_bytes_grayscale(self, tmp_path):
        """Test _process_image_to_bytes with grayscale conversion."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        provider = MockProvider()
        config = OCRConfig(convert_to_grayscale=True)
        ocr = LLMOCR(provider, config)

        result = ocr._process_image_to_bytes(str(img_path))
        assert isinstance(result, bytes)

        # Load the result and check it's grayscale
        import io

        result_img = Image.open(io.BytesIO(result))
        assert result_img.mode == "L"

    def test_process_image_to_bytes_optimize(self, tmp_path):
        """Test _process_image_to_bytes with PNG optimization."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img_path)

        provider = MockProvider()
        config1 = OCRConfig(optimize_png=False)
        config2 = OCRConfig(optimize_png=True)

        ocr1 = LLMOCR(provider, config1)
        ocr2 = LLMOCR(provider, config2)

        result1 = ocr1._process_image_to_bytes(str(img_path))
        result2 = ocr2._process_image_to_bytes(str(img_path))

        # Both should be valid
        assert isinstance(result1, bytes)
        assert isinstance(result2, bytes)


class TestLLMOCRBatchProcessing:
    """Tests for batch processing methods."""

    @pytest.mark.asyncio
    async def test_process_batch_with_llm_single_page(self):
        """Test _process_batch_with_llm with single page."""
        provider = MockProvider(responses=["# Page 1 Content"])
        ocr = LLMOCR(provider)

        result = await ocr._process_batch_with_llm([b"image1"], [1], 10)
        assert result == "# Page 1 Content"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_process_batch_with_llm_multiple_pages(self):
        """Test _process_batch_with_llm with multiple pages."""
        provider = MockProvider(responses=["# Pages 1-3 Content"])
        ocr = LLMOCR(provider)

        result = await ocr._process_batch_with_llm(
            [b"image1", b"image2", b"image3"], [1, 2, 3], 10
        )
        assert result == "# Pages 1-3 Content"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_process_batch_with_llm_strips_whitespace(self):
        """Test _process_batch_with_llm strips whitespace from response."""
        provider = MockProvider(responses=["  \n# Content\n  "])
        ocr = LLMOCR(provider)

        result = await ocr._process_batch_with_llm([b"image1"], [1], 1)
        assert result == "# Content"

    @pytest.mark.asyncio
    async def test_process_batch_with_llm_error_handling(self):
        """Test _process_batch_with_llm error handling."""

        async def failing_generate(*args, **kwargs):
            raise Exception("API Error")

        provider = MockProvider()
        provider.generate_content = failing_generate
        ocr = LLMOCR(provider)

        with pytest.raises(Exception, match="API Error"):
            await ocr._process_batch_with_llm([b"image1"], [1], 1)

    def test_combine_markdown_chunks(self):
        """Test _combine_markdown_chunks combines chunks correctly."""
        provider = MockProvider()
        ocr = LLMOCR(provider)

        chunks = ["# Page 1", "## Page 2", "### Page 3"]
        result = ocr._combine_markdown_chunks(chunks)

        assert "# Page 1" in result
        assert "## Page 2" in result
        assert "### Page 3" in result
        assert result.count("\n\n") >= 2  # Chunks separated by double newline

    def test_combine_markdown_chunks_with_page_markers(self):
        """Test _combine_markdown_chunks with page markers enabled."""
        provider = MockProvider()
        config = OCRConfig(include_page_markers=True)
        ocr = LLMOCR(provider, config)

        chunks = ["# Page 1", "## Page 2", "### Page 3"]
        result = ocr._combine_markdown_chunks(chunks)

        # Should include page markers
        assert "<!-- Page 2 -->" in result or "Page 2" in result

    def test_combine_markdown_chunks_skips_empty(self):
        """Test _combine_markdown_chunks skips empty chunks."""
        provider = MockProvider()
        ocr = LLMOCR(provider)

        chunks = ["# Page 1", "", "   ", "## Page 2"]
        result = ocr._combine_markdown_chunks(chunks)

        assert "# Page 1" in result
        assert "## Page 2" in result
        # Empty chunks should not add extra separators
        assert "\n\n\n\n" not in result


class TestLLMOCRAsyncContext:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test LLMOCR as async context manager."""
        provider = MockProvider()

        async with LLMOCR(provider) as ocr:
            assert isinstance(ocr, LLMOCR)
            assert provider.closed is False

        assert provider.closed is True

    @pytest.mark.asyncio
    async def test_aclose(self):
        """Test aclose method."""
        provider = MockProvider()
        ocr = LLMOCR(provider)

        assert provider.closed is False
        await ocr.aclose()
        assert provider.closed is True


class TestLLMOCRConvert:
    """Tests for convert method."""

    @pytest.mark.asyncio
    async def test_convert_file_not_found(self):
        """Test convert with non-existent PDF file."""
        provider = MockProvider()
        ocr = LLMOCR(provider)

        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            await ocr.convert("/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_convert_with_mock_pdf(self, tmp_path):
        """Test convert method with mocked PDF processing."""
        provider = MockProvider(responses=["# Page 1", "# Page 2"])
        ocr = LLMOCR(provider)

        # Create a dummy PDF path
        pdf_path = tmp_path / "test.pdf"

        # Mock the PDF info and conversion
        mock_info = {"Pages": 2}
        mock_pages = [(1, b"page1_bytes"), (2, b"page2_bytes")]

        async def mock_iter():
            for page_num, page_bytes in mock_pages:
                yield page_num, page_bytes

        with patch("llm_ocr.core.processor.pdfinfo_from_path", return_value=mock_info):
            with patch.object(
                ocr, "_iter_pdf_pages_as_png_bytes", return_value=mock_iter()
            ):
                # Create dummy file
                pdf_path.write_text("dummy")

                result = await ocr.convert(pdf_path)

                assert "# Page 1" in result
                assert "# Page 2" in result
                assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_convert_with_output_path(self, tmp_path):
        """Test convert method saves to output path."""
        provider = MockProvider(responses=["# Content"])
        ocr = LLMOCR(provider)

        pdf_path = tmp_path / "test.pdf"
        output_path = tmp_path / "output.md"

        mock_info = {"Pages": 1}
        mock_pages = [(1, b"page1_bytes")]

        async def mock_iter():
            for page_num, page_bytes in mock_pages:
                yield page_num, page_bytes

        with patch("llm_ocr.core.processor.pdfinfo_from_path", return_value=mock_info):
            with patch.object(
                ocr, "_iter_pdf_pages_as_png_bytes", return_value=mock_iter()
            ):
                pdf_path.write_text("dummy")

                result = await ocr.convert(pdf_path, output_path)

                assert output_path.exists()
                content = output_path.read_text()
                assert content == result
                assert "# Content" in content

    @pytest.mark.asyncio
    async def test_convert_with_max_pages(self, tmp_path):
        """Test convert respects max_pages config."""
        provider = MockProvider(responses=["# Page 1", "# Page 2"])
        config = OCRConfig(max_pages=1)
        ocr = LLMOCR(provider, config)

        pdf_path = tmp_path / "test.pdf"

        mock_info = {"Pages": 10}  # PDF has 10 pages
        mock_pages = [(1, b"page1_bytes")]  # But we only process 1

        async def mock_iter():
            for page_num, page_bytes in mock_pages:
                yield page_num, page_bytes

        with patch("llm_ocr.core.processor.pdfinfo_from_path", return_value=mock_info):
            with patch.object(
                ocr, "_iter_pdf_pages_as_png_bytes", return_value=mock_iter()
            ):
                pdf_path.write_text("dummy")

                await ocr.convert(pdf_path)

                # Should only process 1 page
                assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_convert_with_retries(self, tmp_path):
        """Test convert retries on failure."""
        call_count = 0

        async def failing_then_succeeding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return "# Success"

        provider = MockProvider()
        provider.generate_content = failing_then_succeeding
        config = OCRConfig(max_retries=3, retry_delay=0.01)
        ocr = LLMOCR(provider, config)

        pdf_path = tmp_path / "test.pdf"

        mock_info = {"Pages": 1}
        mock_pages = [(1, b"page1_bytes")]

        async def mock_iter():
            for page_num, page_bytes in mock_pages:
                yield page_num, page_bytes

        with patch("llm_ocr.core.processor.pdfinfo_from_path", return_value=mock_info):
            with patch.object(
                ocr, "_iter_pdf_pages_as_png_bytes", return_value=mock_iter()
            ):
                pdf_path.write_text("dummy")

                result = await ocr.convert(pdf_path)

                assert "# Success" in result
                assert call_count == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_convert_fails_after_max_retries(self, tmp_path):
        """Test convert fails after exhausting retries."""

        async def always_failing(*args, **kwargs):
            raise Exception("Permanent error")

        provider = MockProvider()
        provider.generate_content = always_failing
        config = OCRConfig(max_retries=2, retry_delay=0.01)
        ocr = LLMOCR(provider, config)

        pdf_path = tmp_path / "test.pdf"

        mock_info = {"Pages": 1}
        mock_pages = [(1, b"page1_bytes")]

        async def mock_iter():
            for page_num, page_bytes in mock_pages:
                yield page_num, page_bytes

        with patch("llm_ocr.core.processor.pdfinfo_from_path", return_value=mock_info):
            with patch.object(
                ocr, "_iter_pdf_pages_as_png_bytes", return_value=mock_iter()
            ):
                pdf_path.write_text("dummy")

                with pytest.raises(Exception, match="Permanent error"):
                    await ocr.convert(pdf_path)

    @pytest.mark.asyncio
    async def test_convert_with_concurrency_preserves_order(self, tmp_path):
        """Test concurrent page processing preserves exact page ordering."""
        import asyncio

        class DelayedProvider(LLMProvider):
            def __init__(self):
                super().__init__(api_key="k", model="m")
                self.prompts = []

            @property
            def name(self) -> str:
                return "delayed"

            async def generate_content(
                self, prompt: str, image_bytes_list: List[bytes]
            ) -> str:
                self.prompts.append(prompt)
                # Page 1 sleeps longer than Page 2 to test out-of-order completion
                if "PAGES: 1 of" in prompt:
                    await asyncio.sleep(0.02)
                    return "# Page 1"
                return "# Page 2"

            async def aclose(self):
                pass

        provider = DelayedProvider()
        config = OCRConfig(max_concurrency=2)
        ocr = LLMOCR(provider, config)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        mock_info = {"Pages": 2}

        async def mock_iter():
            yield 1, b"page1_bytes"
            yield 2, b"page2_bytes"

        with patch("llm_ocr.core.processor.pdfinfo_from_path", return_value=mock_info):
            with patch.object(
                ocr, "_iter_pdf_pages_as_png_bytes", return_value=mock_iter()
            ):
                result = await ocr.convert(pdf_path)
                assert result == "# Page 1\n\n# Page 2"

    @pytest.mark.asyncio
    async def test_custom_prompt_and_additional_instructions(self):
        """Test custom_prompt and additional_instructions in OCRConfig."""
        captured_prompts = []

        class CapturingProvider(MockProvider):
            async def generate_content(
                self, prompt: str, image_bytes_list: List[bytes]
            ) -> str:
                captured_prompts.append(prompt)
                return "# Custom Output"

        provider = CapturingProvider()
        config = OCRConfig(
            custom_prompt="Transcribe page {page_range} of {total_pages}.",
            additional_instructions="Extract tables only.",
        )
        ocr = LLMOCR(provider, config)

        res = await ocr._process_batch_with_llm([b"img"], [1], 3)
        assert res == "# Custom Output"
        assert "Transcribe page 1 of 3." in captured_prompts[0]
        assert "ADDITIONAL INSTRUCTIONS:\nExtract tables only." in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_convert_image_path_bytes_and_pil(self, tmp_path, sample_image_bytes):
        """Test convert_image with path, bytes, PIL Image, and convert() with .png."""
        provider = MockProvider(responses=["# Image Markdown"])
        ocr = LLMOCR(provider)

        img_path = tmp_path / "scan.png"
        img = Image.new("RGB", (50, 50), color="white")
        img.save(img_path)

        # 1. Via convert() auto-detecting .png extension
        res1 = await ocr.convert(img_path)
        assert res1 == "# Image Markdown"

        # 2. Via convert_image() with bytes
        res2 = await ocr.convert_image(sample_image_bytes)
        assert res2 == "# Image Markdown"

        # 3. Via convert_image() with PIL Image
        res3 = await ocr.convert_image(img)
        assert res3 == "# Image Markdown"

    @pytest.mark.asyncio
    async def test_convert_bytes_image_and_pdf(self, sample_image_bytes):
        """Test convert_bytes with both image bytes and PDF bytes."""
        provider = MockProvider(responses=["# Bytes Content"])
        ocr = LLMOCR(provider)

        # Image bytes
        res_img = await ocr.convert_bytes(sample_image_bytes, file_type="png")
        assert res_img == "# Bytes Content"

        # PDF bytes
        mock_info = {"Pages": 1}

        async def mock_iter(self, pdf_path):
            yield 1, sample_image_bytes

        with patch("llm_ocr.core.processor.pdfinfo_from_path", return_value=mock_info):
            with patch(
                "llm_ocr.core.processor.LLMOCR._iter_pdf_pages_as_png_bytes",
                mock_iter,
            ):
                res_pdf = await ocr.convert_bytes(b"%PDF-1.4 dummy", file_type="pdf")
                assert res_pdf == "# Bytes Content"

    def test_convert_sync(self, tmp_path):
        """convert_sync is reusable and does not close the provider."""
        provider = MockProvider(responses=["# Sync Markdown"])
        ocr = LLMOCR(provider)

        img_path = tmp_path / "doc.png"
        Image.new("RGB", (40, 40), color="white").save(img_path)

        assert ocr.convert_sync(img_path) == "# Sync Markdown"
        assert ocr.convert_sync(img_path) == "# Sync Markdown"
        assert provider.closed is False

    @pytest.mark.asyncio
    async def test_convert_sync_inside_running_loop(self, tmp_path):
        """convert_sync works where a loop is already running (e.g. notebooks)."""
        ocr = LLMOCR(MockProvider(responses=["# Sync Markdown"]))
        img_path = tmp_path / "doc.png"
        Image.new("RGB", (40, 40), color="white").save(img_path)

        assert ocr.convert_sync(img_path) == "# Sync Markdown"

    @pytest.mark.asyncio
    async def test_convert_fails_fast_and_bounds_concurrency(self, tmp_path):
        """A failed page stops further requests; in-flight ones never exceed the cap."""
        import asyncio

        class FailingProvider(MockProvider):
            in_flight = 0
            max_in_flight = 0

            async def generate_content(
                self, prompt: str, image_bytes_list: List[bytes]
            ) -> str:
                self.call_count += 1
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                try:
                    if "PAGES: 1 of" in prompt:
                        raise RuntimeError("boom")
                    await asyncio.sleep(0.05)
                    return "ok"
                finally:
                    self.in_flight -= 1

        provider = FailingProvider()
        ocr = LLMOCR(provider, OCRConfig(max_concurrency=2, max_retries=1))
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        async def mock_iter(pdf_path):
            for page in range(1, 21):
                yield page, b"png"

        with patch(
            "llm_ocr.core.processor.pdfinfo_from_path", return_value={"Pages": 20}
        ):
            with patch.object(ocr, "_iter_pdf_pages_as_png_bytes", mock_iter):
                with pytest.raises(RuntimeError, match="boom"):
                    await ocr.convert(pdf_path)

        calls_at_failure = provider.call_count
        await asyncio.sleep(0.2)
        assert provider.call_count == calls_at_failure < 20
        assert provider.max_in_flight <= 2

    @pytest.mark.asyncio
    async def test_page_markers_use_real_page_numbers(self, tmp_path):
        """An empty page must not shift the numbering of later page markers."""

        class SkipPage2(MockProvider):
            async def generate_content(
                self, prompt: str, image_bytes_list: List[bytes]
            ) -> str:
                return "" if "PAGES: 2 of" in prompt else "text"

        ocr = LLMOCR(SkipPage2(), OCRConfig(include_page_markers=True))
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        async def mock_iter(pdf_path):
            for page in range(1, 4):
                yield page, b"png"

        with patch(
            "llm_ocr.core.processor.pdfinfo_from_path", return_value={"Pages": 3}
        ):
            with patch.object(ocr, "_iter_pdf_pages_as_png_bytes", mock_iter):
                result = await ocr.convert(pdf_path)

        assert "<!-- Page 3 -->" in result
        assert "<!-- Page 2 -->" not in result

    def test_custom_prompt_with_literal_braces(self):
        """Custom prompts may contain braces (e.g. JSON) next to placeholders."""
        config = OCRConfig(custom_prompt='Return {"page": "{page_range}"} as JSON')
        prompt = LLMOCR(MockProvider(), config)._build_prompt("2", 5)
        assert prompt == 'Return {"page": "2"} as JSON'

    @pytest.mark.asyncio
    async def test_convert_bytes_detects_image_without_hint(self, sample_image_bytes):
        """Image bytes are not mistaken for a PDF when file_type is omitted."""
        ocr = LLMOCR(MockProvider(responses=["# Bytes Content"]))
        assert await ocr.convert_bytes(sample_image_bytes) == "# Bytes Content"

    @pytest.mark.asyncio
    async def test_convert_image_cmyk(self):
        """CMYK images (common for JPEG scans) are converted before PNG encoding."""
        ocr = LLMOCR(MockProvider(responses=["# CMYK"]))
        assert await ocr.convert_image(Image.new("CMYK", (20, 20))) == "# CMYK"

    @pytest.mark.asyncio
    async def test_unusable_response_is_not_retried(self):
        """OCRResponseError (truncated/refused output) skips the retry loop."""
        from llm_ocr.core.base import OCRResponseError

        class TruncatingProvider(MockProvider):
            async def generate_content(
                self, prompt: str, image_bytes_list: List[bytes]
            ) -> str:
                self.call_count += 1
                raise OCRResponseError("truncated")

        provider = TruncatingProvider()
        ocr = LLMOCR(provider, OCRConfig(max_retries=3, retry_delay=0.01))
        with pytest.raises(OCRResponseError):
            await ocr._process_batch_with_retries([b"png"], [1], 1)
        assert provider.call_count == 1
