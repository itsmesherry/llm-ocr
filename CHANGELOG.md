# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-10-12

### Added

- Anthropic Claude provider (`Anthropic`), installed with the optional extra
  `pip install "ocr-llm[anthropic]"`.
- OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, OpenRouter, ...) through
  the `base_url` argument of `OpenAI` or the `OPENAI_BASE_URL` environment variable.
- Concurrent page processing, controlled by `OCRConfig.max_concurrency` (default 4).
- Image input: `convert()` accepts image files, and `convert_image()` accepts a
  path, raw bytes or a `PIL.Image.Image`.
- `convert_bytes()` for in-memory PDFs and images.
- `convert_sync()` for scripts and notebooks that do not use `asyncio`.
- `OCRConfig.custom_prompt` and `OCRConfig.additional_instructions`.
- `ocr-llm` / `llm-ocr` command-line interface.
- `Gemini.FLASH_LITE_2_5` model constant.
- `OCRResponseError`, raised (without retries) when a provider returns truncated
  or refused output.

### Changed

- Progress and errors are reported through the `logging` module instead of
  `print()`. Call `logging.basicConfig(level=logging.INFO)` to see progress.
- Pages are sent to the LLM up to 4 at a time by default. Set
  `OCRConfig(max_concurrency=1)` to restore strictly sequential requests, for
  example on a low rate limit.

### Fixed

- `include_page_markers` now labels each chunk with its real page number when a
  page is empty or `llm_batch_size` is greater than 1.

## [1.0.0]

- Initial release.

[Unreleased]: https://github.com/itsmesherry/llm-ocr/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/itsmesherry/llm-ocr/releases/tag/v1.1.0
