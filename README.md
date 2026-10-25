<h1 align="center">LLM OCR</h1>

<p align="center">
  Turn PDFs and images into clean Markdown with vision LLMs:<br>
  Gemini, OpenAI, Claude, or a local model through Ollama.
</p>

<p align="center">
  <a href="https://pypi.org/project/ocr-llm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/ocr-llm"></a>
  <a href="https://github.com/itsmesherry/llm-ocr/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/itsmesherry/llm-ocr/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/ocr-llm/"><img alt="Python" src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue"></a>
  <a href="https://pypi.org/project/ocr-llm/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/ocr-llm"></a>
  <a href="https://github.com/itsmesherry/llm-ocr/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

<p align="center">
  <img alt="A scanned order summary on the left and the Markdown produced by ocr-llm on the right" src="https://raw.githubusercontent.com/itsmesherry/llm-ocr/main/docs/demo.png" width="900">
</p>

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

### Local models with Ollama

Two Ollama server settings matter:

- **Context length.** At the default 200 DPI a page takes about 4,300 tokens
  with `qwen2.5vl`, more than the 4,096-token context Ollama picks on machines
  with little memory, so every page fails. Start the server with
  `OLLAMA_CONTEXT_LENGTH=8192`, or lower `dpi`.
- **Parallel requests.** Ollama works on `OLLAMA_NUM_PARALLEL` requests at a
  time (often 1) and queues the rest, so a `max_concurrency` above that does
  not make it faster. Set them to the same value. Servers built for batching,
  such as vLLM, do benefit from higher concurrency.

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
ollama pull qwen2.5vl:3b
ocr-llm document.pdf --base-url http://localhost:11434/v1 --model qwen2.5vl:3b --concurrency 1
```

For a server on this machine (`localhost`), requests time out after 30 minutes
instead of the OpenAI SDK's 10, so pages waiting in the queue are not cancelled
and sent again. Pass `timeout=` (in seconds) to `OpenAI(...)` to change it.

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

Warnings, such as a page the model returned no text for, appear on stderr even
without this. Such pages are left out of the output, so check for them.

## Project

[Changelog](https://github.com/itsmesherry/llm-ocr/blob/main/CHANGELOG.md) · [Contributing](https://github.com/itsmesherry/llm-ocr/blob/main/CONTRIBUTING.md) · [Roadmap](https://github.com/itsmesherry/llm-ocr/blob/main/TODO.md) · [MIT License](https://github.com/itsmesherry/llm-ocr/blob/main/LICENSE)
