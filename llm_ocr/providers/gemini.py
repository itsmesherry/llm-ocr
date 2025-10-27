import os
from typing import List, Optional

from google.genai import Client, types

from ..core.base import LLMProvider


class Gemini(LLMProvider):
    """
    Google Gemini provider for OCR.

    Supported models:
    - gemini-2.5-pro (recommended, best quality)
    - gemini-2.5-flash (fast)
    - gemini-2.0-pro
    - gemini-2.0-flash
    """

    # Available models with their identifiers
    FLASH_2_5 = "gemini-2.5-flash"
    PRO_2_5 = "gemini-2.5-pro"
    FLASH_2_0 = "gemini-2.0-flash"
    PRO_2_0 = "gemini-2.0-pro"

    def __init__(self, api_key: Optional[str] = None, model: str = FLASH_2_5, **kwargs):
        """
        Initialize Gemini provider.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var
            model: Model to use. Use Gemini.FLASH_2_5, Gemini.PRO_2_5, etc.
            **kwargs: Additional parameters for the Gemini API
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key is required. "
                "Pass it directly or set GEMINI_API_KEY environment variable."
            )

        super().__init__(api_key, model, **kwargs)

        # Initialize the async client
        client = Client(api_key=self.api_key)
        self.aclient = client.aio

    @property
    def name(self) -> str:
        """Return provider name."""
        return "gemini"

    async def generate_content(
        self,
        prompt: str,
        image_bytes_list: List[bytes],
    ) -> str:
        """Generate content using Gemini with one or more images."""
        # Create image parts for all images in the batch
        image_parts = [
            types.Part(inline_data=types.Blob(mime_type="image/png", data=image_bytes))
            for image_bytes in image_bytes_list
        ]

        # Combine prompt and all images
        contents = [prompt] + image_parts

        try:
            resp = await self.aclient.models.generate_content(
                model=self.model, contents=contents, **self.kwargs
            )

            try:
                return resp.text or ""
            except (ValueError, AttributeError):
                print("[WARNING] Gemini returned no content")
                return ""

        except Exception as err:
            print(f"[ERROR] Gemini generation failed: {err}")
            raise

    async def aclose(self):
        """Close the Gemini client."""
        # The Google GenAI AsyncClient doesn't require explicit closing
        # Resources are cleaned up automatically
        pass
