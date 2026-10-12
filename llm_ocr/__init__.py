"""llm-ocr: Convert PDFs to markdown using LLMs."""

from importlib.metadata import PackageNotFoundError, version

from .config.settings import OCRConfig
from .core.base import LLMProvider, OCRResponseError
from .core.processor import LLMOCR

# Import providers
from .providers.anthropic import Anthropic
from .providers.gemini import Gemini
from .providers.openai import OpenAI

# The version lives in pyproject.toml only
try:
    __version__ = version("ocr-llm")
except PackageNotFoundError:  # source tree that has not been installed
    __version__ = "0.0.0"

__all__ = [
    "LLMOCR",
    "OCRConfig",
    "LLMProvider",
    "OCRResponseError",
    "Anthropic",
    "Gemini",
    "OpenAI",
]
