"""
Text processing for spoken-word output (Phase 6).

Ava's system prompt (app/agents/prompts.py) already instructs the LLM
not to use markdown/bullets/emoji — but prompts are guidance, not a
guarantee. This module is the defensive backstop: it runs on every
response before it's stored or sent to TTS (Phase 7), so an occasional
"**Sure!**" or stray bullet list doesn't get spoken aloud as literal
asterisks or dashes.
"""
import re

# Average adult speaking rate. Used only for a rough duration estimate
# (e.g. for logging/telemetry) — not for cutting anything off.
_WORDS_PER_MINUTE = 150


def sanitize_for_speech(text: str) -> str:
    """Strip markdown formatting and normalize whitespace so `text` is
    safe to hand to a TTS engine. Idempotent — running it twice gives
    the same result as running it once."""
    if not text:
        return ""

    cleaned = text

    # Bold/italic emphasis: **text**, __text__, *text*, _text_
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", cleaned)

    # Headers: "# Heading" -> "Heading"
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

    # Bullet/numbered list markers at line start: "- ", "* ", "1. "
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)

    # Markdown links: [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

    # Inline code / code fences: `text` or ```text``` -> text
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

    # Collapse multi-line structure into natural spoken flow: multiple
    # newlines/blank lines become a single space, not a pause TTS has
    # no good way to represent.
    cleaned = re.sub(r"\n{2,}", " ", cleaned)
    cleaned = re.sub(r"\n", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


def estimate_speech_duration_seconds(text: str) -> float:
    """Rough estimate of how long `text` will take to speak aloud, at
    an average conversational pace. Used for logging/telemetry now;
    Phase 7's TTS step will have the actual audio duration, at which
    point this becomes a sanity-check rather than the source of truth."""
    if not text.strip():
        return 0.0
    word_count = len(text.split())
    return round((word_count / _WORDS_PER_MINUTE) * 60, 1)
