"""llm-ocr: Convert PDFs to markdown using LLMs."""

from .config.settings import OCRConfig
from .core.base import LLMProvider
from .core.processor import LLMOCR

# Import providers
from .providers.gemini import Gemini
from .providers.openai import OpenAI

__version__ = "1.0.0"
__all__ = [
    "LLMOCR",
    "OCRConfig",
    "LLMProvider",
    "Gemini",
    "OpenAI",
]
