"""Tests for llm_ocr.cli module."""

from unittest.mock import patch

import pytest
from PIL import Image

from llm_ocr.cli import _create_provider, _detect_default_provider, build_parser, main


class TestCLI:
    """Tests for the command-line interface."""

    def test_detect_default_provider(self, unset_env_keys, monkeypatch):
        """Test provider auto-detection based on environment variables."""
        assert _detect_default_provider() == "openai"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert _detect_default_provider() == "anthropic"

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert _detect_default_provider() == "openai"

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        assert _detect_default_provider() == "gemini"

    def test_build_parser_arguments(self):
        """Test CLI argument parsing."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "input.pdf",
                "-o",
                "output.md",
                "--provider",
                "anthropic",
                "--model",
                "claude-opus-5-5",
                "--dpi",
                "300",
                "--concurrency",
                "8",
                "--grayscale",
                "--page-markers",
                "--instructions",
                "Tables only",
            ]
        )
        assert args.input == "input.pdf"
        assert args.output == "output.md"
        assert args.provider == "anthropic"
        assert args.model == "claude-opus-5-5"
        assert args.dpi == 300
        assert args.concurrency == 8
        assert args.grayscale is True
        assert args.page_markers is True
        assert args.instructions == "Tables only"

    def test_create_provider_variants(self, mock_api_key):
        """Test _create_provider for openai, gemini, and anthropic."""
        parser = build_parser()

        with patch("llm_ocr.providers.openai.AsyncOpenAI"):
            args_oa = parser.parse_args(
                ["doc.pdf", "-p", "openai", "--api-key", mock_api_key]
            )
            prov_oa = _create_provider(args_oa)
            assert prov_oa.name == "openai"

        with patch("llm_ocr.providers.gemini.Client"):
            args_gem = parser.parse_args(
                ["doc.pdf", "-p", "gemini", "--api-key", mock_api_key]
            )
            prov_gem = _create_provider(args_gem)
            assert prov_gem.name == "gemini"

        with patch("llm_ocr.providers.anthropic.AsyncAnthropic"):
            args_ant = parser.parse_args(
                ["doc.pdf", "-p", "anthropic", "--api-key", mock_api_key]
            )
            prov_ant = _create_provider(args_ant)
            assert prov_ant.name == "anthropic"

    def test_main_success_with_output_file(
        self, tmp_path, mock_api_key, mock_openai_client
    ):
        """Test main() converts an image and saves output to file."""
        img_path = tmp_path / "page.png"
        out_path = tmp_path / "out.md"
        Image.new("RGB", (30, 30), color="white").save(img_path)

        with patch(
            "llm_ocr.providers.openai.AsyncOpenAI", return_value=mock_openai_client
        ):
            exit_code = main(
                [
                    str(img_path),
                    "-o",
                    str(out_path),
                    "-p",
                    "openai",
                    "--api-key",
                    mock_api_key,
                ]
            )
            assert exit_code == 0
            assert out_path.exists()
            assert "# Test Markdown" in out_path.read_text()

    def test_main_error_returns_nonzero(self, unset_env_keys, capsys):
        """Test main() returns 1 on error (e.g. missing API key)."""
        exit_code = main(["nonexistent.pdf", "-p", "openai"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_base_url_rejected_for_non_openai_provider(self, capsys):
        """--base-url must not be silently ignored by other providers."""
        with pytest.raises(SystemExit):
            main(["doc.pdf", "-p", "gemini", "--base-url", "http://localhost:1/v1"])
        assert "--base-url" in capsys.readouterr().err

    def test_base_url_implies_openai(self, unset_env_keys, monkeypatch):
        """--base-url selects the OpenAI-compatible provider despite other keys."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        args = build_parser().parse_args(["doc.pdf", "--base-url", "http://x/v1"])
        with patch("llm_ocr.providers.openai.AsyncOpenAI"):
            assert _create_provider(args).name == "openai"
