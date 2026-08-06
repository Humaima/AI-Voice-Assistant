"""
Agent domain models (Phase 4).

`ChatMessageModel` is the JSON-friendly mirror of LangChain's
BaseMessage subclasses (HumanMessage/AIMessage/SystemMessage) — used at
the API boundary so callers don't need to import LangChain types.
"""
from pydantic import BaseModel, Field


class ChatMessageModel(BaseModel):
    role: str = Field(description="'user', 'assistant', or 'system'")
    content: str


class AgentChatRequest(BaseModel):
    conversation_id: str
    transcript: str
    history: list[ChatMessageModel] = Field(
        default_factory=list,
        description="Prior turns for this conversation. Phase 4 has no "
        "server-side memory, so the caller carries history between calls; "
        "Phase 5 makes this optional once memory is persisted server-side.",
    )


class AgentResult(BaseModel):
    response_text: str
    note: str = Field(
        default="",
        description="Short caption summarizing response_text (Phase 6's "
        "'Generate Ava's Note' step) — meant to accompany the spoken reply "
        "as WhatsApp message text once Phase 9 wires up sending it.",
    )
    history: list[ChatMessageModel]
