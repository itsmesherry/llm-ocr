import os
from typing import Optional

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
    - gemini-1.5-pro
    - gemini-1.5-flash
    """

    # Available models with their identifiers
    FLASH_2_5 = "gemini-2.5-flash"
    PRO_2_5 = "gemini-2.5-pro"
    FLASH_2_0 = "gemini-2.0-flash"
    PRO_2_0 = "gemini-2.0-pro"
    PRO_1_5 = "gemini-1.5-pro"
    FLASH_1_5 = "gemini-1.5-flash"

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
        image_bytes: bytes,
    ) -> str:
        """Generate content using Gemini."""
        image_part = types.Part(
            inline_data=types.Blob(mime_type="image/png", data=image_bytes)
        )

        try:
            resp = await self.aclient.models.generate_content(
                model=self.model, contents=[prompt, image_part], **self.kwargs
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
        await self.aclient.aclose()
