"""
Sentence chunker (Phase 8).

Buffers text deltas as they stream in from the LLM and yields complete
sentence-groups as soon as they're ready, instead of waiting for the
entire response. This is what lets TTS synthesis start on the
beginning of Ava's reply while the LLM is still generating the end of
it — the core latency win of streaming.

Known limitation: sentence-boundary detection is a simple regex on
`. ! ?` followed by whitespace — not real NLP sentence segmentation.
It will over-split on abbreviations ("Dr. Smith") and decimal numbers
("3.14"). That's an acceptable tradeoff here: an early split just means
one extra short TTS clip, not incorrect output. It would NOT be
acceptable if this logic were ever reused somewhere that needs precise
sentence boundaries (e.g. NLP analysis) — worth remembering if this
module gets reused elsewhere later.
"""
import re

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker:
    """Stateful, incremental chunker: call `feed()` with each new text
    delta as it arrives, call `flush()` once at the end of the stream
    to get any leftover text that never hit a sentence boundary.

    `min_chunk_chars` batches short sentences together before flushing
    (e.g. "Sure. Yes. Got it." as one chunk instead of three) — each
    flushed chunk becomes a separate TTS request, and TTS has a fixed
    per-request latency overhead, so over-chunking on short sentences
    can hurt more than it helps.
    """

    def __init__(self, min_chunk_chars: int = 20):
        self.min_chunk_chars = min_chunk_chars
        self._buffer = ""  # text not yet confirmed to end in a complete sentence
        self._pending = ""  # complete sentence(s) accumulated but still under min_chunk_chars

    def feed(self, delta: str) -> list[str]:
        """Add a text delta. Returns zero or more chunks now ready to
        flush (e.g. to a TTS call)."""
        self._buffer += delta

        parts = _SENTENCE_BOUNDARY_RE.split(self._buffer)
        # Everything except the last part is a complete sentence — the
        # last part might still be mid-sentence, so it stays buffered.
        complete_sentences, self._buffer = parts[:-1], parts[-1]

        chunks: list[str] = []
        for sentence in complete_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            self._pending = f"{self._pending} {sentence}".strip()
            if len(self._pending) >= self.min_chunk_chars:
                chunks.append(self._pending)
                self._pending = ""

        return chunks

    def flush(self) -> str | None:
        """Call once at the end of the stream. Returns any remaining
        buffered text (pending short sentences + an incomplete trailing
        fragment) as one final chunk, or None if nothing's left."""
        leftover = f"{self._pending} {self._buffer}".strip()
        self._pending = ""
        self._buffer = ""
        return leftover or None
