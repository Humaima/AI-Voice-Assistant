"""
Basic observability (Phase 10): a handful of counters/histograms for
the metrics that actually matter for this project's cost/reliability
profile — pipeline stage failures (since Groq/ElevenLabs errors cost
money and should be visible) and end-to-end WhatsApp turn latency
(since Phase 10 specifically re-architected around Twilio's webhook
timeout). Intentionally a small, targeted set, not instrumentation of
every function — add more where you actually feel the need for
visibility, rather than pre-instrumenting everything speculatively.

Lives in app/core (not app/api) so services can import these directly
without creating a service -> API -> service dependency cycle;
app/api/metrics.py just exposes them over HTTP for Prometheus to
scrape.
"""
from prometheus_client import Counter, Histogram

whatsapp_messages_total = Counter(
    "whatsapp_messages_total",
    "WhatsApp messages received, by outcome",
    ["outcome"],  # "voice_note_processed" | "no_media_reply" | "pipeline_error"
)

pipeline_stage_errors_total = Counter(
    "pipeline_stage_errors_total",
    "Errors from each pipeline stage",
    ["stage"],  # "media_download" | "audio_validation" | "transcription" | "agent" | "tts"
)

whatsapp_reply_duration_seconds = Histogram(
    "whatsapp_reply_duration_seconds",
    "End-to-end time from receiving a voice note to sending the reply via Twilio's REST API",
    buckets=(1, 2, 5, 10, 15, 20, 30, 60),
)
