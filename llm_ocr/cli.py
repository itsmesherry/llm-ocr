"""Command-line interface for llm-ocr."""

import argparse
import logging
import os
import sys
from typing import List, Optional

from . import __version__
from .config.settings import OCRConfig
from .core.base import LLMProvider
from .core.processor import LLMOCR
from .providers.anthropic import Anthropic
from .providers.gemini import Gemini
from .providers.openai import OpenAI


def _detect_default_provider() -> str:
    """Detect default provider based on set environment variables."""
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="ocr-llm",
        description="Convert PDFs and images to markdown using vision LLMs.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "input",
        help="Path to the input PDF or image file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save the output markdown file (prints to stdout if omitted)",
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=["openai", "gemini", "anthropic"],
        default=None,
        help="LLM provider to use (default: auto-detected from environment)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name to use (default: provider's default model)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the selected provider (prefer the provider's "
        "environment variable: flags are visible in shell history)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom API base URL for OpenAI-compatible servers (e.g. Ollama, vLLM)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion (72-600, default: 200)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of PDF pages to process",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=1,
        help="Number of pages to send to the LLM per request (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum concurrent LLM requests (default: 4)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert images to grayscale before sending to LLM",
    )
    parser.add_argument(
        "--page-markers",
        action="store_true",
        help="Include <!-- Page N --> markers in the markdown output",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Additional instructions to append to the OCR prompt",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def _create_provider(args: argparse.Namespace) -> LLMProvider:
    """Instantiate the requested LLM provider from parsed CLI arguments."""
    # --base-url implies an OpenAI-compatible server
    provider_name = args.provider or (
        "openai" if args.base_url else _detect_default_provider()
    )
    kwargs = {}
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.model:
        kwargs["model"] = args.model

    if provider_name == "gemini":
        return Gemini(**kwargs)
    if provider_name == "anthropic":
        return Anthropic(**kwargs)
    if args.base_url:
        kwargs["base_url"] = args.base_url
    return OpenAI(**kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    if args.base_url and args.provider not in (None, "openai"):
        parser.error("--base-url is only supported with --provider openai")

    try:
        provider = _create_provider(args)
        config = OCRConfig(
            dpi=args.dpi,
            max_pages=args.max_pages,
            llm_batch_size=args.llm_batch_size,
            max_concurrency=args.concurrency,
            convert_to_grayscale=args.grayscale,
            include_page_markers=args.page_markers,
            additional_instructions=args.instructions,
        )
        ocr = LLMOCR(provider=provider, config=config)
        markdown = ocr.convert_sync(args.input, output_path=args.output)
        if not args.output:
            print(markdown)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
