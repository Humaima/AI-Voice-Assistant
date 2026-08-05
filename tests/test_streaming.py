"""
Tests for the Phase 8 streaming orchestration (app/services/streaming.py).

Uses FakeListChatModel (real streaming support built into
langchain-core, character-by-character) plus a fake ElevenLabs client
via TTSService's injectable client — no network calls. Wall-clock
timing is inherently non-deterministic in CI so exact durations aren't
asserted, but ORDER and CONTENT of what gets streamed, synthesized, and
persisted are fully testable.
"""
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.services.streaming import SentenceEvent, stream_response_sentences, stream_voice_reply
from app.services.tts import TTSService


class FakeConversationMemory:
    def __init__(self):
        self.store: dict[str, list[BaseMessage]] = {}

    async def get_recent_messages(self, conversation_id: str, limit: int) -> list[BaseMessage]:
        return self.store.get(conversation_id, [])[-limit:]

    async def append_messages(self, conversation_id: str, messages: list[BaseMessage]) -> None:
        self.store.setdefault(conversation_id, []).extend(messages)


class _FakeTextToSpeechAPI:
    def __init__(self, outer: "FakeTTSClient"):
        self._outer = outer

    def convert(self, voice_id: str, *, text: str, model_id: Any, output_format: Any):
        self._outer.synthesized_texts.append(text)
        return iter([f"audio:{text}".encode()])


class FakeTTSClient:
    """Records every text it was asked to synthesize, in order — lets
    tests assert exactly which sentences triggered TTS calls."""

    def __init__(self):
        self.synthesized_texts: list[str] = []
        self.text_to_speech = _FakeTextToSpeechAPI(self)


class TestStreamResponseSentences:
    async def test_yields_each_sentence_as_it_completes(self):
        llm = FakeListChatModel(responses=["First sentence. Second sentence! Third one?"])
        events = [
            event
            async for event in stream_response_sentences(llm, [HumanMessage(content="hi")], min_chunk_chars=1)
        ]
        texts = [e.text for e in events]
        assert texts == ["First sentence.", "Second sentence!", "Third one?"]

    async def test_events_carry_nondecreasing_elapsed_time(self):
        llm = FakeListChatModel(responses=["One. Two. Three."])
        events = [
            event
            async for event in stream_response_sentences(llm, [HumanMessage(content="hi")], min_chunk_chars=1)
        ]
        elapsed_times = [e.elapsed_seconds for e in events]
        assert elapsed_times == sorted(elapsed_times)

    async def test_returns_sentence_event_instances(self):
        llm = FakeListChatModel(responses=["Just one sentence."])
        events = [
            event
            async for event in stream_response_sentences(llm, [HumanMessage(content="hi")], min_chunk_chars=1)
        ]
        assert all(isinstance(e, SentenceEvent) for e in events)

    async def test_whitespace_only_response_yields_no_events(self):
        llm = FakeListChatModel(responses=["   "])
        events = [
            event
            async for event in stream_response_sentences(llm, [HumanMessage(content="hi")], min_chunk_chars=1)
        ]
        assert events == []


class TestStreamVoiceReply:
    async def test_yields_one_audio_chunk_per_sentence(self):
        llm = FakeListChatModel(
            responses=["This is the first complete sentence. This is the second one here!", "A short note"]
        )
        tts = TTSService(client=FakeTTSClient())

        chunks = [chunk async for chunk in stream_voice_reply("c1", "hello", llm, tts)]

        assert len(chunks) == 2
        assert chunks[0] == b"audio:This is the first complete sentence."
        assert chunks[1] == b"audio:This is the second one here!"

    async def test_synthesizes_sentences_in_order(self):
        llm = FakeListChatModel(
            responses=[
                "Alpha sentence comes first here. Beta sentence comes second. Gamma sentence is the third one.",
                "note",
            ]
        )
        client = FakeTTSClient()
        tts = TTSService(client=client)

        async for _ in stream_voice_reply("c1", "hello", llm, tts, memory_store=None):
            pass

        assert client.synthesized_texts == [
            "Alpha sentence comes first here.",
            "Beta sentence comes second.",
            "Gamma sentence is the third one.",
        ]

    async def test_persists_full_turn_to_memory_store_after_streaming(self):
        llm = FakeListChatModel(responses=["Part one. Part two.", "A note"])
        tts = TTSService(client=FakeTTSClient())
        memory_store = FakeConversationMemory()

        async for _ in stream_voice_reply("c1", "the question", llm, tts, memory_store=memory_store):
            pass

        stored = memory_store.store["c1"]
        assert len(stored) == 2
        assert isinstance(stored[0], HumanMessage)
        assert stored[0].content == "the question"
        assert isinstance(stored[1], AIMessage)
        assert "Part one" in stored[1].content and "Part two" in stored[1].content

    async def test_loads_prior_history_before_streaming(self):
        """Confirms streaming reuses graph.py's _load_memory_node —
        prior conversation history should be visible to the LLM call,
        same as the non-streaming path."""
        captured_prompts = []

        class CapturingModel(FakeListChatModel):
            async def astream(self, input, *args, **kwargs):
                captured_prompts.append(input)
                async for chunk in super().astream(input, *args, **kwargs):
                    yield chunk

        llm = CapturingModel(responses=["A reply.", "note"])
        tts = TTSService(client=FakeTTSClient())
        memory_store = FakeConversationMemory()
        memory_store.store["c1"] = [
            HumanMessage(content="earlier question"),
            AIMessage(content="earlier answer"),
        ]

        async for _ in stream_voice_reply("c1", "follow up", llm, tts, memory_store=memory_store):
            pass

        sent_messages = captured_prompts[0]
        assert isinstance(sent_messages[0], SystemMessage)
        assert sent_messages[1].content == "earlier question"
        assert sent_messages[2].content == "earlier answer"
        assert sent_messages[3].content == "follow up"

    async def test_empty_sentence_after_sanitization_is_skipped(self):
        """A sentence that becomes empty after markdown-stripping
        (e.g. just a stray formatting artifact) shouldn't trigger a
        pointless TTS call."""
        llm = FakeListChatModel(responses=["Real content here.", "note"])
        client = FakeTTSClient()
        tts = TTSService(client=client)

        async for _ in stream_voice_reply("c1", "hi", llm, tts, memory_store=None):
            pass

        # Only the real sentence should have triggered synthesis.
        assert client.synthesized_texts == ["Real content here."]

    async def test_works_with_no_stores_configured(self):
        """Mirrors Phase 4/5's optional-store pattern: streaming should
        work fine with memory_store=None, vector_store=None (the
        defaults) — used by the side-effect-free latency comparison
        endpoint."""
        llm = FakeListChatModel(responses=["A reply with no persistence.", "note"])
        tts = TTSService(client=FakeTTSClient())

        chunks = [chunk async for chunk in stream_voice_reply("c1", "hi", llm, tts)]

        assert len(chunks) == 1
