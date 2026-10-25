# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- HTTP 4xx errors that fail the same way on every attempt (bad request, invalid
  API key, unknown model, request too large) are raised at once instead of being
  retried `max_retries` times. 408, 409 and 429 are still retried, as are 5xx
  and errors without a status code.
- A page the model returns no text for is logged as a warning (it was logged at
  INFO level), so pages missing from the output no longer go unnoticed.
- `OpenAI` with a `base_url` on this machine (`localhost`, `*.localhost`, `127.0.0.1`, `::1`, `0.0.0.0`)
  uses a 30-minute request timeout instead of the SDK's 10 minutes. Local
  servers such as Ollama queue requests beyond `OLLAMA_NUM_PARALLEL`, and the
  shorter timeout cancelled and silently resent queued pages. A `timeout=`
  passed to `OpenAI` still takes precedence.

### Documentation

- README section on running local models with Ollama: `OLLAMA_CONTEXT_LENGTH`
  and matching `max_concurrency` to `OLLAMA_NUM_PARALLEL`.

## [1.2.0] - 2026-10-12

### Changed

- The OpenAI and Gemini providers raise `OCRResponseError` when the model stops
  at its output token limit, instead of returning the cut-off text as if it were
  complete. The Anthropic provider already did this.

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

## [1.0.0] - 2025-11-04

- Initial release.

[Unreleased]: https://github.com/itsmesherry/llm-ocr/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/itsmesherry/llm-ocr/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/itsmesherry/llm-ocr/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/itsmesherry/llm-ocr/releases/tag/v1.0.0
