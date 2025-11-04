"""Tests for llm_ocr.prompts.markdown module."""

from llm_ocr.prompts.markdown import MARKDOWN_SYSTEM_PROMPT


class TestMarkdownPrompt:
    """Tests for markdown system prompt."""

    def test_prompt_exists(self):
        """Test that the prompt is defined."""
        assert MARKDOWN_SYSTEM_PROMPT is not None
        assert isinstance(MARKDOWN_SYSTEM_PROMPT, str)
        assert len(MARKDOWN_SYSTEM_PROMPT) > 0

    def test_prompt_contains_placeholders(self):
        """Test that the prompt contains required format placeholders."""
        assert "{page_range}" in MARKDOWN_SYSTEM_PROMPT
        assert "{total_pages}" in MARKDOWN_SYSTEM_PROMPT

    def test_prompt_formatting_single_page(self):
        """Test prompt formatting with single page."""
        formatted = MARKDOWN_SYSTEM_PROMPT.format(page_range="1", total_pages=5)
        assert "PAGES: 1 of 5" in formatted
        assert "{page_range}" not in formatted
        assert "{total_pages}" not in formatted

    def test_prompt_formatting_page_range(self):
        """Test prompt formatting with page range."""
        formatted = MARKDOWN_SYSTEM_PROMPT.format(page_range="1-3", total_pages=10)
        assert "PAGES: 1-3 of 10" in formatted
        assert "{page_range}" not in formatted
        assert "{total_pages}" not in formatted

    def test_prompt_contains_markdown_instructions(self):
        """Test that prompt contains markdown formatting instructions."""
        prompt_lower = MARKDOWN_SYSTEM_PROMPT.lower()
        assert "markdown" in prompt_lower
        assert "format" in prompt_lower

    def test_prompt_mentions_key_markdown_features(self):
        """Test that prompt mentions key markdown features."""
        prompt_lower = MARKDOWN_SYSTEM_PROMPT.lower()
        # Headers
        assert "#" in MARKDOWN_SYSTEM_PROMPT or "header" in prompt_lower
        # Lists
        assert "list" in prompt_lower
        # Tables
        assert "table" in prompt_lower
        # Bold/italic
        assert "bold" in prompt_lower or "**" in MARKDOWN_SYSTEM_PROMPT

    def test_prompt_contains_critical_instructions(self):
        """Test that prompt contains critical instructions."""
        prompt_lower = MARKDOWN_SYSTEM_PROMPT.lower()
        # Should mention not to add commentary
        assert "do not" in prompt_lower or "don't" in prompt_lower
        # Should mention output format
        assert "output" in prompt_lower

    def test_prompt_mentions_multi_page_handling(self):
        """Test that prompt mentions handling multiple pages."""
        prompt_lower = MARKDOWN_SYSTEM_PROMPT.lower()
        assert "page" in prompt_lower
        assert "multiple" in prompt_lower or "pages" in prompt_lower

    def test_prompt_is_multi_line(self):
        """Test that prompt is multi-line."""
        lines = MARKDOWN_SYSTEM_PROMPT.strip().split("\n")
        assert len(lines) > 5  # Should be a substantial prompt

    def test_prompt_no_extra_braces(self):
        """Test that prompt doesn't have unformatted braces."""
        # After formatting, there should be no remaining braces
        formatted = MARKDOWN_SYSTEM_PROMPT.format(page_range="1", total_pages=1)
        # Allow double braces {{ }} which are escaped
        import re

        # Find single braces (not doubled)
        single_braces = re.findall(r"(?<!\{)\{(?!\{)|(?<!\})\}(?!\})", formatted)
        assert len(single_braces) == 0, "Found unformatted placeholders"
