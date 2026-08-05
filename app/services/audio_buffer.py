"""
Audio Buffer — the stage between chunking and Whisper transcription in
the architecture diagram (step 2, feeding step 3).

Why a buffer class instead of just a list of chunks: it (a) exports
each chunk as WAV bytes ready for the Groq Whisper API, which expects
a file-like object, and (b) tracks consumption state, so Phase 3 can
pull chunks one at a time — this is what will let Phase 8 (streaming)
start transcribing chunk 1 while chunk 2 is still being buffered,
instead of waiting for the whole voice note.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

from pydub import AudioSegment


@dataclass
class BufferedChunk:
    index: int
    segment: AudioSegment
    consumed: bool = False

    def to_wav_bytes(self) -> bytes:
        """Export as WAV bytes — the format Groq's Whisper endpoint
        (and most STT APIs) accept as a file upload."""
        buf = io.BytesIO()
        self.segment.export(buf, format="wav")
        return buf.getvalue()


@dataclass
class AudioBuffer:
    """Holds ordered chunks for one voice note and hands them out for
    transcription. One AudioBuffer instance = one in-flight voice note.
    """

    conversation_id: str
    chunks: list[BufferedChunk] = field(default_factory=list)

    @classmethod
    def from_segments(cls, conversation_id: str, segments: list[AudioSegment]) -> "AudioBuffer":
        chunks = [BufferedChunk(index=i, segment=s) for i, s in enumerate(segments)]
        return cls(conversation_id=conversation_id, chunks=chunks)

    def total_duration_ms(self) -> int:
        return sum(len(c.segment) for c in self.chunks)

    def next_unconsumed(self) -> BufferedChunk | None:
        """Get the next chunk not yet sent to Whisper, without removing
        it — lets a caller retry a failed transcription without losing
        buffer state."""
        for chunk in self.chunks:
            if not chunk.consumed:
                return chunk
        return None

    def mark_consumed(self, index: int) -> None:
        for chunk in self.chunks:
            if chunk.index == index:
                chunk.consumed = True
                return
        raise IndexError(f"No chunk with index {index} in buffer")

    def is_fully_consumed(self) -> bool:
        return all(c.consumed for c in self.chunks)

    def iter_wav_bytes(self):
        """Convenience generator for Phase 3: yields (index, wav_bytes)
        for every chunk in order, without mutating consumed state."""
        for chunk in self.chunks:
            yield chunk.index, chunk.to_wav_bytes()
