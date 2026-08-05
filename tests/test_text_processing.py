"""
Tests for Phase 6's text_processing module — the defensive backstop
that strips markdown before anything reaches TTS, plus the rough
speech-duration estimate used for logging.
"""
from app.agents.text_processing import estimate_speech_duration_seconds, sanitize_for_speech


class TestSanitizeForSpeech:
    def test_empty_string_returns_empty(self):
        assert sanitize_for_speech("") == ""

    def test_plain_text_passes_through_unchanged(self):
        text = "Hey, I think the weather looks good today."
        assert sanitize_for_speech(text) == text

    def test_strips_bold_double_asterisk(self):
        assert sanitize_for_speech("This is **important** news") == "This is important news"

    def test_strips_bold_double_underscore(self):
        assert sanitize_for_speech("This is __important__ news") == "This is important news"

    def test_strips_italic_single_asterisk(self):
        assert sanitize_for_speech("This is *quite* nice") == "This is quite nice"

    def test_strips_italic_single_underscore(self):
        assert sanitize_for_speech("This is _quite_ nice") == "This is quite nice"

    def test_does_not_mangle_words_with_internal_underscores(self):
        # e.g. variable names or snake_case shouldn't be treated as
        # italics markers when there's no word boundary.
        text = "the file is named my_file_name"
        assert "my_file_name" in sanitize_for_speech(text)

    def test_strips_markdown_headers(self):
        text = "# Big Heading\nSome content here"
        result = sanitize_for_speech(text)
        assert "#" not in result
        assert "Big Heading" in result

    def test_strips_bullet_list_markers(self):
        text = "- first item\n- second item\n- third item"
        result = sanitize_for_speech(text)
        assert "-" not in result.replace("first", "").replace("second", "").replace("third", "")
        assert "first item" in result
        assert "second item" in result

    def test_strips_numbered_list_markers(self):
        text = "1. do this\n2. then that"
        result = sanitize_for_speech(text)
        assert "1." not in result
        assert "2." not in result
        assert "do this" in result

    def test_strips_markdown_links_keeps_link_text(self):
        text = "Check out [the podcast](https://example.com) for more"
        result = sanitize_for_speech(text)
        assert "the podcast" in result
        assert "https://" not in result
        assert "[" not in result and "]" not in result

    def test_strips_inline_code(self):
        text = "Run the `pytest` command"
        assert sanitize_for_speech(text) == "Run the pytest command"

    def test_strips_code_fences_entirely(self):
        text = "Here's the code:\n```python\nprint('hi')\n```\nThat's it"
        result = sanitize_for_speech(text)
        assert "print" not in result
        assert "```" not in result

    def test_collapses_multiple_newlines_into_space(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = sanitize_for_speech(text)
        assert "\n" not in result
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_collapses_extra_whitespace(self):
        text = "Too    many     spaces"
        result = sanitize_for_speech(text)
        assert "  " not in result

    def test_strips_leading_trailing_whitespace(self):
        assert sanitize_for_speech("   hello there   ") == "hello there"

    def test_is_idempotent(self):
        text = "**Bold** and _italic_ and a [link](url) and `code`"
        once = sanitize_for_speech(text)
        twice = sanitize_for_speech(once)
        assert once == twice

    def test_combined_realistic_markdown_response(self):
        text = (
            "# Weekend Plan\n\n"
            "Here's what I'd suggest:\n"
            "- Go for a **hike** in the morning\n"
            "- Grab lunch at that place we talked about\n"
            "- Check out the [new exhibit](https://museum.example.com)\n\n"
            "Let me know if that works!"
        )
        result = sanitize_for_speech(text)
        assert "#" not in result
        assert "**" not in result
        assert "[" not in result
        assert "\n" not in result
        assert "hike" in result
        assert "new exhibit" in result


class TestEstimateSpeechDuration:
    def test_empty_text_is_zero_seconds(self):
        assert estimate_speech_duration_seconds("") == 0.0

    def test_whitespace_only_is_zero_seconds(self):
        assert estimate_speech_duration_seconds("   ") == 0.0

    def test_longer_text_takes_longer(self):
        short = estimate_speech_duration_seconds("Hello there.")
        long = estimate_speech_duration_seconds(" ".join(["word"] * 100))
        assert long > short

    def test_returns_positive_float_for_normal_text(self):
        duration = estimate_speech_duration_seconds("This is a normal sentence to say aloud.")
        assert isinstance(duration, float)
        assert duration > 0
