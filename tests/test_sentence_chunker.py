"""
Tests for the Phase 8 sentence chunker — the piece that decides when
enough streamed text has accumulated to flush as one TTS-ready chunk.
"""
from app.services.sentence_chunker import SentenceChunker


def _feed_all(chunker: SentenceChunker, text: str) -> list[str]:
    """Feed a string one character at a time (simulating real
    token-by-token LLM streaming) and collect every chunk produced,
    including the final flush."""
    chunks: list[str] = []
    for ch in text:
        chunks.extend(chunker.feed(ch))
    leftover = chunker.flush()
    if leftover:
        chunks.append(leftover)
    return chunks


class TestBasicSentenceSplitting:
    def test_single_sentence(self):
        chunker = SentenceChunker(min_chunk_chars=1)
        assert _feed_all(chunker, "Hello there.") == ["Hello there."]

    def test_multiple_sentences_split_correctly(self):
        chunker = SentenceChunker(min_chunk_chars=1)
        text = "This is one. This is two! Is this three?"
        assert _feed_all(chunker, text) == ["This is one.", "This is two!", "Is this three?"]

    def test_no_terminal_punctuation_yields_one_final_chunk(self):
        chunker = SentenceChunker(min_chunk_chars=1)
        assert _feed_all(chunker, "no punctuation here") == ["no punctuation here"]

    def test_empty_input_yields_nothing(self):
        chunker = SentenceChunker()
        assert _feed_all(chunker, "") == []

    def test_whitespace_only_input_yields_nothing(self):
        chunker = SentenceChunker()
        assert _feed_all(chunker, "   ") == []


class TestMinChunkCharsBatching:
    def test_short_sentences_batch_together(self):
        chunker = SentenceChunker(min_chunk_chars=20)
        text = "Sure. Yes. Got it."  # each sentence alone is under 20 chars
        chunks = _feed_all(chunker, text)
        assert chunks == ["Sure. Yes. Got it."]

    def test_batch_flushes_once_threshold_crossed(self):
        chunker = SentenceChunker(min_chunk_chars=10)
        text = "Sure. Yes. Got it. This part is much longer than the rest of it."
        chunks = _feed_all(chunker, text)
        assert chunks[0] == "Sure. Yes."
        assert "Got it." in chunks[1]

    def test_single_long_sentence_flushes_immediately(self):
        chunker = SentenceChunker(min_chunk_chars=5)
        text = "This one sentence alone is already long enough to flush."
        chunks = _feed_all(chunker, text)
        assert chunks == [text]


class TestIncrementalFeeding:
    def test_feed_can_be_called_with_multi_char_deltas(self):
        """Real streaming doesn't always arrive one character at a
        time — some models yield multi-token chunks. Chunker should
        handle deltas of any size."""
        chunker = SentenceChunker(min_chunk_chars=1)
        chunks = []
        for delta in ["This is ", "one sentence. ", "This is ", "another!"]:
            chunks.extend(chunker.feed(delta))
        leftover = chunker.flush()
        if leftover:
            chunks.append(leftover)
        assert chunks == ["This is one sentence.", "This is another!"]

    def test_sentence_split_exactly_at_delta_boundary(self):
        """The period lands as the very last character of one delta —
        make sure the boundary is still detected correctly once the
        next delta (starting with whitespace) arrives."""
        chunker = SentenceChunker(min_chunk_chars=1)
        chunks = []
        chunks.extend(chunker.feed("Sentence one."))
        chunks.extend(chunker.feed(" Sentence two."))
        leftover = chunker.flush()
        if leftover:
            chunks.append(leftover)
        assert chunks == ["Sentence one.", "Sentence two."]


class TestFlushBehavior:
    def test_flush_on_fresh_chunker_returns_none(self):
        chunker = SentenceChunker()
        assert chunker.flush() is None

    def test_flush_returns_none_after_everything_already_flushed(self):
        chunker = SentenceChunker(min_chunk_chars=1)
        chunker.feed("Complete sentence.")
        chunker.flush()  # nothing left after a fully-punctuated sentence already flushed
        assert chunker.flush() is None

    def test_flush_combines_pending_and_buffer(self):
        """If there's both a short pending sentence AND an incomplete
        trailing fragment when the stream ends, flush should combine
        both into one final chunk rather than losing one of them."""
        chunker = SentenceChunker(min_chunk_chars=100)  # high threshold so nothing auto-flushes
        chunker.feed("Short.")  # becomes _pending (under threshold)
        chunker.feed(" trailing fragment without punctuation")  # stays in _buffer
        leftover = chunker.flush()
        assert leftover == "Short. trailing fragment without punctuation"


class TestRealisticResponses:
    def test_typical_short_ava_reply(self):
        chunker = SentenceChunker(min_chunk_chars=20)
        text = "Yeah, I know that podcast. It's really good."
        chunks = _feed_all(chunker, text)
        assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

    def test_preserves_all_content_regardless_of_chunking(self):
        """Whatever the chunk boundaries end up being, no words should
        ever be dropped or duplicated across chunks."""
        chunker = SentenceChunker(min_chunk_chars=15)
        text = "First sentence here. Second one follows. Third is the last!"
        chunks = _feed_all(chunker, text)
        reconstructed = " ".join(chunks)
        for word in text.replace(".", "").replace("!", "").split():
            assert word in reconstructed
