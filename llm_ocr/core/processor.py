import asyncio
import io
import tempfile
from pathlib import Path
from typing import AsyncIterable, List, Optional, Tuple, Union

from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

from ..config.settings import OCRConfig
from ..core.base import LLMProvider
from ..prompts.markdown import MARKDOWN_SYSTEM_PROMPT


class LLMOCR:
    """Main processor for converting PDFs to markdown using LLMs."""

    def __init__(self, provider: LLMProvider, config: Optional[OCRConfig] = None):
        """
        Initialize LLM OCR converter.

        Args:
            provider: An instance of an LLM provider (Gemini, OpenAI, etc.)
            config: OCRConfig object for processing settings
        """
        if not isinstance(provider, LLMProvider):
            raise TypeError(
                f"provider must be an instance of LLMProvider, got {type(provider)}"
            )

        self.provider = provider
        self.config = config or OCRConfig()
        self.config.validate()

    def _process_image_to_bytes(self, img_path: str) -> bytes:
        """Convert image file to PNG bytes."""
        with Image.open(img_path) as im:
            if self.config.convert_to_grayscale:
                im = im.convert("L")
            buff = io.BytesIO()
            im.save(buff, format="PNG", optimize=self.config.optimize_png)
            return buff.getvalue()

    async def _iter_pdf_pages_as_png_bytes(
        self, pdf_path: Path
    ) -> AsyncIterable[Tuple[int, bytes]]:
        """Yield (page_number, png_bytes) for each page."""
        info = pdfinfo_from_path(pdf_path)
        total_pages = info["Pages"]
        if self.config.max_pages:
            total_pages = min(total_pages, self.config.max_pages)

        page = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            while page <= total_pages:
                first = page
                last = min(page + self.config.batch_size - 1, total_pages)

                img_paths = convert_from_path(
                    pdf_path,
                    dpi=self.config.dpi,
                    first_page=first,
                    last_page=last,
                    output_folder=tmpdir,
                    fmt="png",
                    paths_only=True,
                    thread_count=self.config.thread_count,
                    use_cropbox=self.config.use_cropbox,
                )

                for p, img_path in enumerate(img_paths, start=first):
                    png_bytes = await asyncio.to_thread(
                        self._process_image_to_bytes, img_path
                    )
                    yield p, png_bytes

                page = last + 1

    async def _process_batch_with_llm(
        self, png_bytes_list: List[bytes], page_nums: List[int], total_pages: int
    ) -> str:
        """Process a batch of pages with the LLM provider."""
        # Format page range for prompt
        if len(page_nums) == 1:
            page_range = f"{page_nums[0]}"
        else:
            page_range = f"{page_nums[0]}-{page_nums[-1]}"

        prompt = MARKDOWN_SYSTEM_PROMPT.format(
            page_range=page_range, total_pages=total_pages
        )

        try:
            content = await self.provider.generate_content(prompt, png_bytes_list)
            return content.strip()
        except Exception as err:
            print(f"[ERROR] Pages {page_range}: Failed to process: {err}")
            raise  # Fail-fast: re-raise to stop processing

    def _combine_markdown_chunks(self, chunks: List[str]) -> str:
        """Combine markdown chunks from multiple pages."""
        combined = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            # Add page separator comment for debugging (optional)
            if i > 0 and self.config.include_page_markers:
                combined.append(f"\n<!-- Page {i + 1} -->\n")

            combined.append(chunk)

        return "\n\n".join(combined)

    async def aclose(self):
        """Close and cleanup resources."""
        await self.provider.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()

    async def convert(
        self, pdf_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Convert PDF to markdown.

        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path to save the markdown file

        Returns:
            Markdown content as string
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Get total pages
        info = pdfinfo_from_path(pdf_path)
        total_pages = info["Pages"]
        if self.config.max_pages:
            total_pages = min(total_pages, self.config.max_pages)

        print(f"[INFO] Converting PDF: {pdf_path.name} ({total_pages} pages)")
        print(f"[INFO] Using provider: {self.provider.name} ({self.provider.model})")
        print(f"[INFO] LLM batch size: {self.config.llm_batch_size}")

        markdown_chunks = []

        # Collect pages into batches according to llm_batch_size
        page_batch = []
        page_num_batch = []

        async for page_num, png_bytes in self._iter_pdf_pages_as_png_bytes(pdf_path):
            page_batch.append(png_bytes)
            page_num_batch.append(page_num)

            # Process batch when it reaches llm_batch_size or this is the last page
            if len(page_batch) >= self.config.llm_batch_size or page_num == total_pages:
                # Retry logic for transient errors
                markdown_content = ""
                for attempt in range(self.config.max_retries):
                    try:
                        markdown_content = await self._process_batch_with_llm(
                            page_batch, page_num_batch, total_pages
                        )
                        break
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            page_range = (
                                f"{page_num_batch[0]}-{page_num_batch[-1]}"
                                if len(page_num_batch) > 1
                                else str(page_num_batch[0])
                            )
                            print(
                                f"[ERROR] Pages {page_range}: Failed after "
                                f"{self.config.max_retries} attempts: {e}"
                            )
                            raise  # Fail-fast: stop processing on failure
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))

                if markdown_content.strip():
                    page_range = (
                        f"{page_num_batch[0]}-{page_num_batch[-1]}"
                        if len(page_num_batch) > 1
                        else str(page_num_batch[0])
                    )
                    print(
                        f"[OK] Pages {page_range}/{total_pages} processed "
                        f"({len(markdown_content)} chars)"
                    )
                    markdown_chunks.append(markdown_content)
                else:
                    page_range = (
                        f"{page_num_batch[0]}-{page_num_batch[-1]}"
                        if len(page_num_batch) > 1
                        else str(page_num_batch[0])
                    )
                    print(f"[OK] Pages {page_range}/{total_pages} - No content")

                # Clear the batch
                page_batch = []
                page_num_batch = []

        # Combine all chunks
        final_markdown = self._combine_markdown_chunks(markdown_chunks)

        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(final_markdown, encoding="utf-8")
            print(f"[DONE] Saved to: {output_path}")

        print(f"[DONE] Generated {len(final_markdown)} characters of markdown")
        return final_markdown
