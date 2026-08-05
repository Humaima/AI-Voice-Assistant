"""
Streaming domain models (Phase 8) — used by the metrics-only and
latency-comparison debug endpoints in app/api/agent.py. These exist
purely to make Phase 8's latency improvement *measurable*, since
listening to streamed vs. non-streamed audio sounds identical — the
difference is in timing, not content.
"""
from pydantic import BaseModel


class SentenceTiming(BaseModel):
    text: str
    elapsed_seconds: float


class StreamMetricsResult(BaseModel):
    sentences: list[SentenceTiming]
    full_text: str
    total_elapsed_seconds: float


class LatencyComparisonResult(BaseModel):
    blocking_total_ms: float
    streaming_time_to_first_audio_ms: float
    streaming_total_ms: float
    speedup_to_first_audio: float
