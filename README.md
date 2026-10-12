# LLM OCR

[![PyPI](https://img.shields.io/pypi/v/ocr-llm)](https://pypi.org/project/ocr-llm/)
[![CI](https://github.com/itsmesherry/llm-ocr/actions/workflows/ci.yml/badge.svg)](https://github.com/itsmesherry/llm-ocr/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/ocr-llm/)
[![Downloads](https://img.shields.io/pypi/dm/ocr-llm)](https://pypi.org/project/ocr-llm/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/itsmesherry/llm-ocr/blob/main/LICENSE)

Turn PDFs and images into clean Markdown with vision LLMs: Gemini, OpenAI,
Claude, or a local model through Ollama. Tables, headings and lists come out as
Markdown, not as a wall of text.

```bash
pip install ocr-llm
export GEMINI_API_KEY="..."   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
ocr-llm document.pdf -o document.md
```

Or from Python:

```python
from llm_ocr import LLMOCR, Gemini

markdown = LLMOCR(Gemini()).convert_sync("document.pdf")
```

It handles multi-page PDFs concurrently, keeps pages in order, retries failed
requests, and works with any OpenAI-compatible server.

## Installation

```bash
pip install ocr-llm
```

Gemini and OpenAI work out of the box. Claude needs an extra:

```bash
pip install "ocr-llm[anthropic]"
```

PDF input needs [poppler](https://poppler.freedesktop.org/): `brew install poppler`
on macOS, `sudo apt-get install poppler-utils` on Debian/Ubuntu. Images work
without it.

## Providers

| Provider | Class       | API key             | Default model       |
| -------- | ----------- | ------------------- | ------------------- |
| Gemini   | `Gemini`    | `GEMINI_API_KEY`    | `gemini-2.5-flash`  |
| OpenAI   | `OpenAI`    | `OPENAI_API_KEY`    | `gpt-4o-mini`       |
| Claude   | `Anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |

Keys are read from the environment, or passed as `api_key=`. Choose a model
with `model=`: either a constant such as `Gemini.PRO_2_5` or any model name as
a string, so new models work without a library update.

```python
from llm_ocr import Anthropic, Gemini, OpenAI

Gemini(model=Gemini.PRO_2_5)
Anthropic(model="claude-haiku-5-5")

# Ollama, vLLM, LM Studio, OpenRouter: anything that speaks the OpenAI API
OpenAI(base_url="http://localhost:11434/v1", model="llava")
```

## Usage

`convert_sync` is the simplest entry point and also works in notebooks:

```python
from llm_ocr import LLMOCR, OpenAI

ocr = LLMOCR(OpenAI())
markdown = ocr.convert_sync("document.pdf", output_path="document.md")
```

In async code, use `convert` and its siblings:

```python
import asyncio

from llm_ocr import LLMOCR, Gemini


async def main():
    async with LLMOCR(Gemini()) as ocr:
        report = await ocr.convert("report.pdf")
        receipt = await ocr.convert("receipt.png")  # images work too

        # bytes from an upload or object store; PDF or image is detected
        with open("report.pdf", "rb") as f:
            from_bytes = await ocr.convert_bytes(f.read())


asyncio.run(main())
```

`convert_image` also accepts raw bytes or a `PIL.Image.Image`.

## Command line

```bash
# Uses whichever of GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY is set
ocr-llm document.pdf -o document.md

ocr-llm scan.png --provider anthropic --instructions "Tables only"

ocr-llm document.pdf --base-url http://localhost:11434/v1 --model llava
```

Without `-o`, the Markdown goes to stdout. `ocr-llm --help` lists every option.

## Configuration

```python
from llm_ocr import LLMOCR, OCRConfig, OpenAI

config = OCRConfig(
    dpi=300,
    max_concurrency=8,
    additional_instructions="Write formulas in LaTeX.",
)
ocr = LLMOCR(OpenAI(), config=config)
```

| Option                    | Default | Description                                         |
| ------------------------- | ------- | --------------------------------------------------- |
| `dpi`                     | 200     | Resolution for rendering PDF pages (72 to 600)      |
| `max_pages`               | None    | Stop after this many pages                          |
| `llm_batch_size`          | 1       | Pages sent to the model per request                 |
| `max_concurrency`         | 4       | Concurrent requests; use 1 on low rate limits       |
| `max_retries`             | 3       | Attempts per request                                |
| `retry_delay`             | 1.0     | Seconds between attempts, multiplied by the attempt |
| `include_page_markers`    | False   | Insert `<!-- Page N -->` between pages              |
| `custom_prompt`           | None    | Replace the built-in prompt                         |
| `additional_instructions` | None    | Text appended to the prompt                         |
| `convert_to_grayscale`    | False   | Send grayscale images                               |
| `optimize_png`            | True    | Smaller PNGs at the cost of slower encoding         |
| `use_cropbox`             | True    | Render the PDF crop box instead of the media box    |
| `batch_size`              | 5       | Pages rendered per poppler call                     |
| `thread_count`            | 4       | Poppler rendering threads                           |

## Logging

Progress is reported through the standard `logging` module (it was `print()`
before 1.1.0). To see it:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Project

[Changelog](https://github.com/itsmesherry/llm-ocr/blob/main/CHANGELOG.md) · [Contributing](https://github.com/itsmesherry/llm-ocr/blob/main/CONTRIBUTING.md) · [Roadmap](https://github.com/itsmesherry/llm-ocr/blob/main/TODO.md) · [MIT License](https://github.com/itsmesherry/llm-ocr/blob/main/LICENSE)
