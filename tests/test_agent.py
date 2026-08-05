"""
Tests for the LangGraph agent — Phase 4's graph wiring, Phase 5's
memory integration, and Phase 6's response sanitization + note
generation.

Uses LangChain's built-in FakeListChatModel — a real BaseChatModel
implementation that returns preset responses in order, with no network
calls — plus small in-memory fakes for ConversationMemory/VectorMemory
so these tests never touch a real database or ChromaDB instance. The
real store implementations have their own dedicated test files:
tests/test_conversation_store.py and tests/test_vector_store.py.

IMPORTANT: since Phase 6, every turn makes TWO LLM calls — one for
generate_response, one for generate_note. FakeListChatModel consumes
its `responses` list in call order, so every responses=[...] list below
must supply a note-response immediately after each turn's actual reply,
e.g. responses=["turn 1 reply", "turn 1 note", "turn 2 reply", "turn 2
note"] for a two-turn conversation.
"""
import re

from typing import Any, cast

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agents.graph import build_graph
from app.agents.prompts import NOTE_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.agents.state import AgentState
from app.models.agent import ChatMessageModel
from app.services.agent_service import (
    AgentService,
    messages_from_models,
    messages_to_models,
)


def _state(**overrides: Any) -> AgentState:
    """Builds a complete AgentState with sensible defaults, so
    individual tests only need to specify what's relevant to them."""
    base: dict[str, Any] = {
        "conversation_id": "c1",
        "transcript": "hello",
        "messages": [],
        "response": "",
        "note": "",
    }
    base.update(overrides)
    return cast(AgentState, base)


class FakeConversationMemory:
    """In-memory stand-in for ConversationMemory — a dict of
    conversation_id -> message list, no database involved."""

    def __init__(self):
        self.store: dict[str, list[BaseMessage]] = {}

    async def get_recent_messages(self, conversation_id: str, limit: int) -> list[BaseMessage]:
        return self.store.get(conversation_id, [])[-limit:]

    async def append_messages(self, conversation_id: str, messages: list[BaseMessage]) -> None:
        self.store.setdefault(conversation_id, []).extend(messages)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


class FakeVectorMemory:
    """In-memory stand-in for VectorMemory — no real embeddings, just
    word-overlap matching, enough to test that the graph calls
    search/add_memory correctly without needing real ChromaDB."""

    def __init__(self):
        self.entries: list[tuple[str, str, str]] = []  # (conversation_id, role, content)

    def add_memory(self, conversation_id: str, role: str, content: str) -> None:
        self.entries.append((conversation_id, role, content))

    def search(self, conversation_id: str, query: str, k: int) -> list[str]:
        query_words = _words(query)
        matches = [
            content
            for cid, _role, content in self.entries
            if cid == conversation_id and _words(content) & query_words
        ]
        return matches[:k]


class TestGraphBasics:
    async def test_single_turn_produces_response(self):
        llm = FakeListChatModel(responses=["Hi! How can I help?", "Offers help"])
        graph = build_graph(llm)

        result = await graph.ainvoke(_state(transcript="hello"))

        assert result["response"] == "Hi! How can I help?"

    async def test_messages_include_user_turn_and_ai_reply(self):
        llm = FakeListChatModel(responses=["got it", "Confirms understanding"])
        graph = build_graph(llm)

        result = await graph.ainvoke(_state(transcript="remember this"))

        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["messages"][0].content == "remember this"
        assert isinstance(result["messages"][1], AIMessage)
        assert result["messages"][1].content == "got it"

    async def test_prior_history_is_preserved_and_extended(self):
        llm = FakeListChatModel(responses=["second reply", "Replies again"])
        graph = build_graph(llm)

        prior = [HumanMessage(content="first message"), AIMessage(content="first reply")]
        result = await graph.ainvoke(_state(transcript="second message", messages=prior))

        # add_messages reducer should append, not replace
        assert len(result["messages"]) == 4
        assert result["messages"][0].content == "first message"
        assert result["messages"][1].content == "first reply"
        assert result["messages"][2].content == "second message"
        assert result["messages"][3].content == "second reply"

    async def test_note_field_is_populated(self):
        llm = FakeListChatModel(responses=["Sure, I can help with that", "Offers to help"])
        graph = build_graph(llm)

        result = await graph.ainvoke(_state(transcript="can you help me"))

        assert result["note"] == "Offers to help"


class TestPromptConstruction:
    async def test_system_prompt_and_history_sent_to_response_call(self):
        """FakeListChatModel doesn't expose what it was called with
        directly, so we use a tiny local subclass to capture every
        message list passed to .ainvoke(). The graph makes 2 calls per
        turn (response, then note) — this checks the FIRST call, which
        should be the response-generation call using SYSTEM_PROMPT."""

        captured_calls = []

        class CapturingFakeModel(FakeListChatModel):
            async def ainvoke(self, input, *args, **kwargs):
                captured_calls.append(input)
                return await super().ainvoke(input, *args, **kwargs)

        llm = CapturingFakeModel(responses=["ok", "A short note"])
        graph = build_graph(llm)

        prior = [HumanMessage(content="earlier question"), AIMessage(content="earlier answer")]
        await graph.ainvoke(_state(transcript="follow up", messages=prior))

        assert len(captured_calls) == 2

        response_call = captured_calls[0]
        assert isinstance(response_call[0], SystemMessage)
        assert response_call[0].content == SYSTEM_PROMPT
        assert response_call[1].content == "earlier question"
        assert response_call[2].content == "earlier answer"
        assert response_call[3].content == "follow up"

    async def test_note_call_uses_note_prompt_and_the_generated_response(self):
        """The SECOND call (note generation) should use NOTE_SYSTEM_PROMPT
        and receive the response text as its input, not the original
        transcript or full history."""

        captured_calls = []

        class CapturingFakeModel(FakeListChatModel):
            async def ainvoke(self, input, *args, **kwargs):
                captured_calls.append(input)
                return await super().ainvoke(input, *args, **kwargs)

        llm = CapturingFakeModel(responses=["The actual spoken reply", "A short note"])
        graph = build_graph(llm)

        await graph.ainvoke(_state(transcript="some question"))

        note_call = captured_calls[1]
        assert isinstance(note_call[0], SystemMessage)
        assert note_call[0].content == NOTE_SYSTEM_PROMPT
        assert note_call[1].content == "The actual spoken reply"


class TestResponseSanitization:
    async def test_markdown_in_llm_output_gets_stripped_from_response(self):
        """Defensive backstop: even if the LLM ignores the "no
        markdown" instruction, sanitize_for_speech should clean it up
        before it's stored in state['response']."""
        llm = FakeListChatModel(responses=["Here's the **plan**: - do this - then that", "Shares a plan"])
        graph = build_graph(llm)

        result = await graph.ainvoke(_state(transcript="what's the plan"))

        assert "**" not in result["response"]
        assert "do this" in result["response"]

    async def test_note_is_also_sanitized(self):
        llm = FakeListChatModel(responses=["reply", "**Bold** note"])
        graph = build_graph(llm)

        result = await graph.ainvoke(_state(transcript="hi"))

        assert "**" not in result["note"]


class TestNoteFallback:
    async def test_note_generation_failure_falls_back_to_truncated_response(self):
        """If the note-generation LLM call raises, the turn shouldn't
        fail entirely — fall back to a simple truncation of the actual
        response instead of losing the whole result."""

        class FailingNoteModel(FakeListChatModel):
            call_count: int = 0

            async def ainvoke(self, input, *args, **kwargs):
                self.call_count += 1
                if self.call_count == 2:  # the note-generation call
                    raise RuntimeError("simulated note generation failure")
                return await super().ainvoke(input, *args, **kwargs)

        llm = FailingNoteModel(responses=["one two three four five six seven eight nine ten eleven twelve thirteen"])
        graph = build_graph(llm)

        result = await graph.ainvoke(_state(transcript="hi"))

        # Response generation succeeded and is untouched by the note failure.
        assert result["response"].startswith("one two three")
        # Note falls back to a truncated version of the response (max 12 words + "...").
        assert result["note"].startswith("one two three")
        assert result["note"].endswith("...")

    async def test_empty_response_produces_empty_note_without_calling_llm(self):
        """Edge case: if generate_response somehow produced an empty
        string, generate_note shouldn't bother calling the LLM at all."""
        llm = FakeListChatModel(responses=[""])
        graph = build_graph(llm)

        result = await graph.ainvoke(_state(transcript="hi"))

        assert result["response"] == ""
        assert result["note"] == ""


class TestAgentService:
    async def test_run_returns_response_note_and_history_models(self):
        llm = FakeListChatModel(responses=["Ava's reply", "Gives a reply"])
        service = AgentService(llm=llm)

        result = await service.run("c1", "hi there")

        assert result.response_text == "Ava's reply"
        assert result.note == "Gives a reply"
        assert len(result.history) == 2
        assert result.history[0] == ChatMessageModel(role="user", content="hi there")
        assert result.history[1] == ChatMessageModel(role="assistant", content="Ava's reply")

    async def test_multi_turn_conversation_via_service_without_persistence(self):
        """No memory_store configured: history must be carried by the
        caller, exactly like Phase 4."""
        llm = FakeListChatModel(responses=["first reply", "Note 1", "second reply", "Note 2"])
        service = AgentService(llm=llm)

        first = await service.run("c1", "first message")
        second = await service.run(
            "c1", "second message", history=messages_from_models(first.history)
        )

        assert second.response_text == "second reply"
        assert len(second.history) == 4
        assert second.history[0].content == "first message"
        assert second.history[2].content == "second message"


class TestAgentServiceWithMemory:
    """Phase 5: with a memory_store configured, conversation history
    persists across separate .run() calls automatically — the caller
    no longer needs to pass `history` back in."""

    async def test_history_persists_across_calls_without_caller_passing_it(self):
        llm = FakeListChatModel(responses=["first reply", "Note 1", "second reply", "Note 2"])
        memory_store = FakeConversationMemory()
        service = AgentService(llm=llm, memory_store=memory_store)

        first = await service.run("c1", "first message")
        # Note: no `history=` passed here — the store should supply it.
        second = await service.run("c1", "second message")

        assert first.response_text == "first reply"
        assert second.response_text == "second reply"
        # The store should now hold all 4 messages for this conversation.
        assert len(memory_store.store["c1"]) == 4
        assert memory_store.store["c1"][0].content == "first message"
        assert memory_store.store["c1"][2].content == "second message"

    async def test_explicit_history_is_ignored_when_memory_store_configured(self):
        """Passing `history` explicitly shouldn't double up with what
        the store already has — the store is authoritative."""
        llm = FakeListChatModel(responses=["reply", "Note"])
        memory_store = FakeConversationMemory()
        service = AgentService(llm=llm, memory_store=memory_store)

        stray_history = [HumanMessage(content="ignored"), AIMessage(content="also ignored")]
        result = await service.run("c1", "actual message", history=stray_history)

        assert "ignored" not in result.history[0].content
        assert result.history[0].content == "actual message"

    async def test_conversations_are_isolated_by_id(self):
        llm = FakeListChatModel(responses=["reply to c1", "Note 1", "reply to c2", "Note 2"])
        memory_store = FakeConversationMemory()
        service = AgentService(llm=llm, memory_store=memory_store)

        await service.run("c1", "hello from c1")
        await service.run("c2", "hello from c2")

        assert len(memory_store.store["c1"]) == 2
        assert len(memory_store.store["c2"]) == 2
        assert memory_store.store["c1"][0].content == "hello from c1"
        assert memory_store.store["c2"][0].content == "hello from c2"

    async def test_vector_store_receives_new_turn(self):
        llm = FakeListChatModel(responses=["reply", "Note"])
        vector_store = FakeVectorMemory()
        service = AgentService(llm=llm, vector_store=vector_store)

        await service.run("c1", "my dog is named Rex")

        stored_contents = [content for _cid, _role, content in vector_store.entries]
        assert "my dog is named Rex" in stored_contents
        assert "reply" in stored_contents

    async def test_semantic_recall_surfaces_relevant_older_context(self):
        """Simulates: something mentioned in turn 1 gets surfaced by
        vector search when asked about much later, even without it
        being in the (small) recent-history window."""
        llm = FakeListChatModel(
            responses=["Rex is a good name!", "Note 1", "Yes, your dog is named Rex.", "Note 2"]
        )
        memory_store = FakeConversationMemory()
        vector_store = FakeVectorMemory()
        service = AgentService(llm=llm, memory_store=memory_store, vector_store=vector_store)

        await service.run("c1", "my dog is named Rex")
        second = await service.run("c1", "what's my dog's name again?")

        # The graph should have found "my dog is named Rex" via vector
        # search and folded it into a SystemMessage before this turn.
        recall_messages = [m for m in second.history if m.role == "system"]
        assert any("Rex" in m.content for m in recall_messages)


class TestMessageConversion:
    def test_models_to_messages_roundtrip(self):
        models = [
            ChatMessageModel(role="user", content="hi"),
            ChatMessageModel(role="assistant", content="hello"),
        ]
        messages = messages_from_models(models)
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

        back = messages_to_models(messages)
        assert back == models

    def test_unknown_role_falls_back_to_human(self):
        models = [ChatMessageModel(role="mystery", content="???")]
        messages = messages_from_models(models)
        assert isinstance(messages[0], HumanMessage)
