import asyncio
import io
import logging
import tempfile
import threading
from pathlib import Path
from typing import AsyncIterable, List, Optional, Tuple, Union

from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

from ..config.settings import OCRConfig
from ..core.base import LLMProvider, OCRResponseError
from ..prompts.markdown import MARKDOWN_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
}


class LLMOCR:
    """Main processor for converting PDFs and images to markdown using LLMs."""

    def __init__(self, provider: LLMProvider, config: Optional[OCRConfig] = None):
        """
        Initialize LLM OCR converter.

        Args:
            provider: An instance of an LLM provider (Gemini, OpenAI, Anthropic, etc.)
            config: OCRConfig object for processing settings
        """
        if not isinstance(provider, LLMProvider):
            raise TypeError(
                f"provider must be an instance of LLMProvider, got {type(provider)}"
            )

        self.provider = provider
        self.config = config or OCRConfig()
        self.config.validate()
        self._sync_loop: Optional[asyncio.AbstractEventLoop] = None

    def _pil_image_to_png_bytes(self, im: Image.Image) -> bytes:
        """Convert a PIL Image object to PNG bytes."""
        if getattr(im, "n_frames", 1) > 1:
            logger.warning(
                "Image has %d frames/pages; only the first one is converted",
                im.n_frames,  # type: ignore[attr-defined]
            )
        if self.config.convert_to_grayscale:
            im = im.convert("L")
        elif im.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
            # PNG cannot store CMYK, YCbCr, etc.
            im = im.convert("RGB")
        buff = io.BytesIO()
        im.save(buff, format="PNG", optimize=self.config.optimize_png)
        return buff.getvalue()

    def _process_image_to_bytes(self, img_path: Union[str, Path]) -> bytes:
        """Convert image file to PNG bytes."""
        with Image.open(img_path) as im:
            return self._pil_image_to_png_bytes(im)

    async def _iter_pdf_pages_as_png_bytes(
        self, pdf_path: Path
    ) -> AsyncIterable[Tuple[int, bytes]]:
        """Yield (page_number, png_bytes) for each page."""
        info = pdfinfo_from_path(str(pdf_path))
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
                        self._process_image_to_bytes, str(img_path)
                    )
                    yield p, png_bytes

                page = last + 1

    def _build_prompt(self, page_range: str, total_pages: int) -> str:
        """Build the prompt for a given page range."""
        if self.config.custom_prompt:
            # str.replace, not str.format: custom prompts may contain literal
            # braces (e.g. JSON examples).
            prompt = self.config.custom_prompt.replace(
                "{page_range}", page_range
            ).replace("{total_pages}", str(total_pages))
        else:
            prompt = MARKDOWN_SYSTEM_PROMPT.format(
                page_range=page_range, total_pages=total_pages
            )

        if self.config.additional_instructions:
            prompt = (
                f"{prompt.rstrip()}\n\n"
                f"ADDITIONAL INSTRUCTIONS:\n{self.config.additional_instructions}\n"
            )

        return prompt

    async def _process_batch_with_llm(
        self, png_bytes_list: List[bytes], page_nums: List[int], total_pages: int
    ) -> str:
        """Process a batch of pages with the LLM provider."""
        # Format page range for prompt
        if len(page_nums) == 1:
            page_range = f"{page_nums[0]}"
        else:
            page_range = f"{page_nums[0]}-{page_nums[-1]}"

        prompt = self._build_prompt(page_range=page_range, total_pages=total_pages)

        try:
            content = await self.provider.generate_content(prompt, png_bytes_list)
            return content.strip()
        except Exception as err:
            logger.error("Pages %s: Failed to process: %s", page_range, err)
            raise  # Fail-fast: re-raise to stop processing

    async def _process_batch_with_retries(
        self,
        png_bytes_list: List[bytes],
        page_nums: List[int],
        total_pages: int,
    ) -> str:
        """Process a batch of pages, retrying transient failures."""
        page_range = (
            f"{page_nums[0]}-{page_nums[-1]}"
            if len(page_nums) > 1
            else str(page_nums[0])
        )
        markdown_content = ""
        for attempt in range(self.config.max_retries):
            try:
                markdown_content = await self._process_batch_with_llm(
                    png_bytes_list, page_nums, total_pages
                )
                break
            except OCRResponseError:
                raise  # Deterministic (truncated/refused output): retrying won't help
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        "Pages %s: Failed after %d attempts: %s",
                        page_range,
                        self.config.max_retries,
                        e,
                    )
                    raise  # Fail-fast: stop processing on failure
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        if markdown_content.strip():
            logger.info(
                "Pages %s/%d processed (%d chars)",
                page_range,
                total_pages,
                len(markdown_content),
            )
        else:
            logger.info("Pages %s/%d - No content", page_range, total_pages)

        return markdown_content

    def _combine_markdown_chunks(
        self, chunks: List[str], page_nums: Optional[List[int]] = None
    ) -> str:
        """
        Combine markdown chunks from multiple pages.

        Args:
            chunks: Markdown for each batch, in document order
            page_nums: First page number of each chunk, used for page markers.
                Defaults to the chunk's position (1-based).
        """
        combined = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            # Add page separator comment for debugging (optional)
            if i > 0 and self.config.include_page_markers:
                page = page_nums[i] if page_nums else i + 1
                combined.append(f"\n<!-- Page {page} -->\n")

            combined.append(chunk)

        return "\n\n".join(combined)

    def _save_output(
        self, markdown: str, output_path: Optional[Union[str, Path]]
    ) -> None:
        """Save markdown content to output_path if provided."""
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
            logger.info("Saved to: %s", output_path)

    async def aclose(self):
        """Close and cleanup resources."""
        await self.provider.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()

    async def convert_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Convert a single image (path, bytes, or PIL Image) to markdown.

        Args:
            image_input: Path to image file, raw image bytes, or PIL.Image.Image
            output_path: Optional path to save the markdown file

        Returns:
            Markdown content as string
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            png_bytes = await asyncio.to_thread(self._process_image_to_bytes, img_path)
        elif isinstance(image_input, bytes):

            def _bytes_to_png(raw: bytes) -> bytes:
                with Image.open(io.BytesIO(raw)) as im:
                    return self._pil_image_to_png_bytes(im)

            png_bytes = await asyncio.to_thread(_bytes_to_png, image_input)
        elif isinstance(image_input, Image.Image):
            png_bytes = await asyncio.to_thread(
                self._pil_image_to_png_bytes, image_input
            )
        else:
            raise TypeError(
                "image_input must be a file path, bytes, or PIL.Image.Image"
            )

        markdown_content = await self._process_batch_with_retries([png_bytes], [1], 1)
        final_markdown = self._combine_markdown_chunks([markdown_content])
        self._save_output(final_markdown, output_path)
        return final_markdown

    async def convert_bytes(
        self,
        data: bytes,
        file_type: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Convert in-memory PDF or image bytes to markdown.

        Args:
            data: Raw bytes of a PDF or image
            file_type: Optional hint: 'pdf' or an image format ('png', 'jpg', ...).
                If omitted, PDFs are detected from their header and anything
                else is treated as an image.
            output_path: Optional path to save the markdown file

        Returns:
            Markdown content as string
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")

        normalized_type = (file_type or "").lower().lstrip(".")
        if normalized_type == "pdf" or b"%PDF" in bytes(data[:1024]):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_pdf = Path(tmpdir) / "document.pdf"
                tmp_pdf.write_bytes(data)
                return await self.convert(tmp_pdf, output_path=output_path)
        else:
            return await self.convert_image(bytes(data), output_path=output_path)

    async def convert(
        self, pdf_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Convert a PDF (or image) file to markdown.

        Args:
            pdf_path: Path to the PDF or image file
            output_path: Optional path to save the markdown file

        Returns:
            Markdown content as string
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() in IMAGE_EXTENSIONS:
            return await self.convert_image(pdf_path, output_path=output_path)

        # Get total pages
        info = pdfinfo_from_path(str(pdf_path))
        total_pages = info["Pages"]
        if self.config.max_pages:
            total_pages = min(total_pages, self.config.max_pages)

        logger.info("Converting PDF: %s (%d pages)", pdf_path.name, total_pages)
        logger.info("Using provider: %s (%s)", self.provider.name, self.provider.model)
        logger.info(
            "LLM batch size: %d, max concurrency: %d",
            self.config.llm_batch_size,
            self.config.max_concurrency,
        )

        # Pages are rendered and sent as we go: at most `max_concurrency` batches
        # are in flight (and held in memory) at any time.
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        tasks: List["asyncio.Task[str]"] = []
        first_pages: List[int] = []

        async def run_batch(pages: List[bytes], nums: List[int]) -> str:
            try:
                return await self._process_batch_with_retries(pages, nums, total_pages)
            finally:
                semaphore.release()

        async def submit(pages: List[bytes], nums: List[int]) -> None:
            await semaphore.acquire()
            # Fail-fast: don't start new batches once one has failed
            for task in tasks:
                if task.done() and not task.cancelled() and task.exception():
                    raise task.exception()  # type: ignore[misc]
            tasks.append(asyncio.create_task(run_batch(pages, nums)))
            first_pages.append(nums[0])

        page_batch: List[bytes] = []
        page_num_batch: List[int] = []

        try:
            async for page_num, png_bytes in self._iter_pdf_pages_as_png_bytes(
                pdf_path
            ):
                page_batch.append(png_bytes)
                page_num_batch.append(page_num)

                if len(page_batch) >= self.config.llm_batch_size:
                    await submit(page_batch, page_num_batch)
                    page_batch = []
                    page_num_batch = []

            if page_batch:
                await submit(page_batch, page_num_batch)

            markdown_chunks = await asyncio.gather(*tasks)
        except BaseException:
            # Fail-fast: cancel in-flight batches so they stop costing money
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # Combine all chunks
        final_markdown = self._combine_markdown_chunks(
            list(markdown_chunks), first_pages
        )

        # Save to file if output path provided
        self._save_output(final_markdown, output_path)

        logger.info("Generated %d characters of markdown", len(final_markdown))
        return final_markdown

    def convert_sync(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Synchronous convenience wrapper for `convert()`.

        Safe to call repeatedly and from environments that already run an
        event loop (e.g. Jupyter notebooks): the work runs on a dedicated
        background event loop owned by this instance.

        Args:
            input_path: Path to the PDF or image file
            output_path: Optional path to save the markdown file

        Returns:
            Markdown content as string
        """
        if self._sync_loop is None:
            self._sync_loop = asyncio.new_event_loop()
            threading.Thread(target=self._sync_loop.run_forever, daemon=True).start()

        future = asyncio.run_coroutine_threadsafe(
            self.convert(input_path, output_path=output_path), self._sync_loop
        )
        try:
            return future.result()
        except BaseException:
            future.cancel()  # e.g. KeyboardInterrupt: stop in-flight requests
            raise
