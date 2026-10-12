# LLM OCR

[![PyPI](https://img.shields.io/pypi/v/ocr-llm)](https://pypi.org/project/ocr-llm/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/ocr-llm/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/itsmesherry/llm-ocr/blob/main/LICENSE)

Convert PDFs and images to clean Markdown using Large Language Models (LLMs) with vision capabilities.

## Features

- 🔍 **High-quality OCR** using vision-capable LLMs
- 🔌 **Multiple Providers**: Google Gemini, OpenAI, Anthropic Claude, and OpenAI-compatible local/cloud endpoints (Ollama, vLLM, OpenRouter, Groq)
- ⚡ **Concurrent Processing**: Parallel LLM batch execution (`max_concurrency`) for faster multi-page PDF conversion
- 🖼️ **Flexible Inputs**: Convert PDFs, standalone images (`.png`, `.jpg`, `.webp`, `.tiff`), PIL Images, or in-memory `bytes`
- 💻 **CLI Included**: Convert documents directly from the terminal with `ocr-llm` / `llm-ocr`
- 🎯 **Custom Prompts**: Override the system prompt or append custom instructions (e.g., tables only, LaTeX math, translation)
- 🔄 **Automatic Retries** & synchronous convenience wrapper (`convert_sync`)

## Installation

```bash
pip install ocr-llm
```

Gemini and OpenAI support is included. Anthropic Claude is an optional extra:

```bash
pip install "ocr-llm[anthropic]"
```

Install `poppler` (required for PDF-to-image conversion):

```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# Fedora/RHEL
sudo yum install poppler-utils
```

## CLI Usage

Convert PDFs or images straight from the terminal:

```bash
# Auto-detects provider from GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
ocr-llm document.pdf -o output.md

# Specify provider, model, and concurrency
ocr-llm document.pdf -o output.md --provider gemini --model gemini-2.5-flash --concurrency 4

# Use Anthropic Claude
ocr-llm invoice.png -o invoice.md --provider anthropic --instructions "Extract all tables accurately"

# Use local Ollama vision model
ocr-llm document.pdf -o output.md --provider openai --base-url http://localhost:11434/v1 --model llava
```

## Quick Start (Python API)

### Synchronous Usage (`convert_sync`)

```python
from llm_ocr import LLMOCR, Gemini

provider = Gemini(model=Gemini.FLASH_2_5)  # Reads GEMINI_API_KEY from env
ocr = LLMOCR(provider)

markdown = ocr.convert_sync("document.pdf", output_path="output.md")
print(markdown)
```

### Async Usage with OpenAI

```python
import asyncio
from llm_ocr import LLMOCR, OpenAI


async def main():
    provider = OpenAI(
        api_key="your-api-key",  # Or set OPENAI_API_KEY env var
        model=OpenAI.GPT_4O_MINI,
    )

    async with LLMOCR(provider) as ocr:
        markdown = await ocr.convert("document.pdf", output_path="output.md")
        print(markdown)


asyncio.run(main())
```

### Using Anthropic Claude

```python
import asyncio
from llm_ocr import LLMOCR, Anthropic


async def main():
    provider = Anthropic(
        api_key="your-api-key",  # Or set ANTHROPIC_API_KEY env var
        model=Anthropic.CLAUDE_SONNET_5_5,
    )

    async with LLMOCR(provider) as ocr:
        markdown = await ocr.convert("document.pdf", output_path="output.md")
        print(markdown)


asyncio.run(main())
```

### Using Local / OpenAI-Compatible Models (Ollama, vLLM, OpenRouter)

```python
from llm_ocr import LLMOCR, OpenAI

provider = OpenAI(
    base_url="http://localhost:11434/v1",  # Or set OPENAI_BASE_URL env var
    model="qwen2.5vl",
)

ocr = LLMOCR(provider)
markdown = ocr.convert_sync("document.pdf")
```

## Converting Images & In-Memory Bytes

```python
import asyncio
from llm_ocr import LLMOCR, Gemini


async def main():
    async with LLMOCR(Gemini()) as ocr:
        # 1. Direct image file (.png, .jpg, .webp, .tiff, etc.)
        md_from_image = await ocr.convert("receipt.png")

        # 2. In-memory PDF or image bytes (e.g. FastAPI UploadFile or S3 download)
        with open("document.pdf", "rb") as f:
            pdf_bytes = f.read()
        md_from_bytes = await ocr.convert_bytes(pdf_bytes)


asyncio.run(main())
```

## Available Models

### Gemini
- `Gemini.PRO_2_5` (`gemini-2.5-pro`)
- `Gemini.FLASH_2_5` (`gemini-2.5-flash` - default)
- `Gemini.FLASH_LITE_2_5` (`gemini-2.5-flash-lite`)
- `Gemini.PRO_2_0` (`gemini-2.0-pro`)
- `Gemini.FLASH_2_0` (`gemini-2.0-flash`)

### OpenAI
- `OpenAI.GPT_4O` (`gpt-4o`)
- `OpenAI.GPT_4O_MINI` (`gpt-4o-mini` - default)
- `OpenAI.GPT_5`, `OpenAI.GPT_5_MINI`, `OpenAI.GPT_5_NANO`
- `OpenAI.GPT_4_1`, `OpenAI.GPT_4_1_MINI`, `OpenAI.GPT_4_1_NANO`
- `OpenAI.O1`, `OpenAI.O3`, `OpenAI.O4_MINI`

### Anthropic
- `Anthropic.CLAUDE_OPUS_5_5` (`claude-opus-5-5`)
- `Anthropic.CLAUDE_SONNET_5_5` (`claude-sonnet-5-5` - default)
- `Anthropic.CLAUDE_HAIKU_5_5` (`claude-haiku-5-5`)

Providers retire models over time. Any model identifier can be passed as a plain
string (`model="..."`), so you are never limited to the constants above.

## Configuration

Customize the OCR processing with `OCRConfig`:

```python
from llm_ocr import LLMOCR, OpenAI, OCRConfig

config = OCRConfig(
    dpi=300,  # Higher DPI for better quality
    max_pages=10,  # Limit number of pages to process
    llm_batch_size=2,  # Send 2 pages to LLM at once
    max_concurrency=4,  # Process up to 4 batches concurrently
    convert_to_grayscale=True,  # Convert images to grayscale
    max_retries=3,  # Retry failed requests
    retry_delay=1.0,  # Wait 1 second between retries
    include_page_markers=True,  # Add <!-- Page N --> markers in output
    additional_instructions="Preserve all mathematical formulas in LaTeX.",
)

provider = OpenAI()
ocr = LLMOCR(provider, config=config)
```

### Configuration Options

| Option                    | Default | Description                                               |
| ------------------------- | ------- | --------------------------------------------------------- |
| `dpi`                     | 200     | DPI for PDF to image conversion (72–600)                  |
| `max_pages`               | None    | Maximum number of pages to process                        |
| `batch_size`              | 5       | PDF to image conversion batch size                        |
| `llm_batch_size`          | 1       | Number of pages to send to LLM at once                    |
| `max_concurrency`         | 4       | Maximum concurrent LLM requests                           |
| `thread_count`            | 4       | Number of threads for PDF conversion                      |
| `convert_to_grayscale`    | False   | Convert images to grayscale                               |
| `optimize_png`            | True    | Optimize PNG compression                                  |
| `use_cropbox`             | True    | Use PDF cropbox for conversion                            |
| `max_retries`             | 3       | Maximum retry attempts for failed requests                |
| `retry_delay`             | 1.0     | Delay between retries in seconds                          |
| `include_page_markers`    | False   | Add page markers in markdown output                       |
| `custom_prompt`           | None    | Override the default system prompt                        |
| `additional_instructions` | None    | Extra instructions appended to the prompt                 |

## Progress Output

Since 1.1.0 progress is reported through the standard `logging` module instead of
`print()`. To see it:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Environment Variables

Set API keys via environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Planned work is tracked in [TODO.md](TODO.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

See [LICENSE](LICENSE) file for details.
