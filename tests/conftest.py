"""Shared pytest fixtures for llm-ocr tests."""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def sample_image_bytes():
    """Create sample PNG image bytes for testing."""
    img = Image.new("RGB", (100, 100), color="white")
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return buff.getvalue()


@pytest.fixture
def sample_image_bytes_list(sample_image_bytes):
    """Create a list of sample image bytes."""
    return [sample_image_bytes, sample_image_bytes]


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a sample PDF file path (doesn't actually create the file)."""
    return tmp_path / "sample.pdf"


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI async client."""
    client = MagicMock()
    client.close = AsyncMock()

    # Mock the chat completions response
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "# Test Markdown\nSample content"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    client.chat.completions.create = AsyncMock(return_value=mock_response)

    return client


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini async client."""
    client = MagicMock()

    # Mock the generate_content response
    mock_response = MagicMock()
    mock_response.text = "# Test Markdown\nSample content"

    client.models.generate_content = AsyncMock(return_value=mock_response)

    return client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic async client."""
    client = MagicMock()
    client.close = AsyncMock()

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "# Test Markdown\nSample content"
    mock_response.content = [mock_block]

    client.messages.create = AsyncMock(return_value=mock_response)

    return client


@pytest.fixture
def set_openai_env_key(mock_api_key, monkeypatch):
    """Set OPENAI_API_KEY environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", mock_api_key)


@pytest.fixture
def set_gemini_env_key(mock_api_key, monkeypatch):
    """Set GEMINI_API_KEY environment variable."""
    monkeypatch.setenv("GEMINI_API_KEY", mock_api_key)


@pytest.fixture
def set_anthropic_env_key(mock_api_key, monkeypatch):
    """Set ANTHROPIC_API_KEY environment variable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", mock_api_key)


@pytest.fixture
def unset_env_keys(monkeypatch):
    """Remove API key environment variables."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
