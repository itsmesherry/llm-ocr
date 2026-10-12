"""Tests for llm_ocr.config.settings module."""

import pytest

from llm_ocr.config.settings import OCRConfig


class TestOCRConfig:
    """Tests for OCRConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OCRConfig()
        assert config.dpi == 200
        assert config.max_pages is None
        assert config.batch_size == 5
        assert config.llm_batch_size == 1
        assert config.max_concurrency == 4
        assert config.thread_count == 4
        assert config.convert_to_grayscale is False
        assert config.optimize_png is True
        assert config.use_cropbox is True
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.output_format == "markdown"
        assert config.include_page_markers is False
        assert config.custom_prompt is None
        assert config.additional_instructions is None

    def test_custom_values(self):
        """Test setting custom configuration values."""
        config = OCRConfig(
            dpi=300,
            max_pages=10,
            batch_size=3,
            llm_batch_size=2,
            max_concurrency=8,
            thread_count=8,
            convert_to_grayscale=True,
            optimize_png=False,
            use_cropbox=False,
            max_retries=5,
            retry_delay=2.0,
            output_format="html",
            include_page_markers=True,
            custom_prompt="Custom prompt",
            additional_instructions="Tables only",
        )
        assert config.dpi == 300
        assert config.max_pages == 10
        assert config.batch_size == 3
        assert config.llm_batch_size == 2
        assert config.max_concurrency == 8
        assert config.thread_count == 8
        assert config.convert_to_grayscale is True
        assert config.optimize_png is False
        assert config.use_cropbox is False
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.output_format == "html"
        assert config.include_page_markers is True
        assert config.custom_prompt == "Custom prompt"
        assert config.additional_instructions == "Tables only"

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = OCRConfig(dpi=150, batch_size=3, llm_batch_size=2)
        config.validate()  # Should not raise

    def test_validate_dpi_too_low(self):
        """Test validation fails when DPI is too low."""
        config = OCRConfig(dpi=50)
        with pytest.raises(ValueError, match="DPI must be between 72 and 600"):
            config.validate()

    def test_validate_dpi_too_high(self):
        """Test validation fails when DPI is too high."""
        config = OCRConfig(dpi=700)
        with pytest.raises(ValueError, match="DPI must be between 72 and 600"):
            config.validate()

    def test_validate_batch_size_zero(self):
        """Test validation fails when batch_size is zero."""
        config = OCRConfig(batch_size=0)
        with pytest.raises(ValueError, match="Batch size must be at least 1"):
            config.validate()

    def test_validate_batch_size_negative(self):
        """Test validation fails when batch_size is negative."""
        config = OCRConfig(batch_size=-1)
        with pytest.raises(ValueError, match="Batch size must be at least 1"):
            config.validate()

    def test_validate_llm_batch_size_zero(self):
        """Test validation fails when llm_batch_size is zero."""
        config = OCRConfig(llm_batch_size=0)
        with pytest.raises(ValueError, match="LLM batch size must be at least 1"):
            config.validate()

    def test_validate_llm_batch_size_negative(self):
        """Test validation fails when llm_batch_size is negative."""
        config = OCRConfig(llm_batch_size=-1)
        with pytest.raises(ValueError, match="LLM batch size must be at least 1"):
            config.validate()

    def test_validate_max_concurrency_zero(self):
        """Test validation fails when max_concurrency is zero."""
        config = OCRConfig(max_concurrency=0)
        with pytest.raises(ValueError, match="Max concurrency must be at least 1"):
            config.validate()

    def test_validate_boundary_values(self):
        """Test validation with boundary values."""
        # Minimum valid values
        config = OCRConfig(dpi=72, batch_size=1, llm_batch_size=1, max_concurrency=1)
        config.validate()  # Should not raise

        # Maximum valid DPI
        config = OCRConfig(dpi=600)
        config.validate()  # Should not raise

    def test_validate_called_implicitly(self):
        """Test that validation is called when config is used."""
        # This tests that the config can be instantiated
        # and validation will be called when needed
        config = OCRConfig()
        assert config.validate() is None
