"""
LangGraph state schema (Phase 4).

This is the shared state every node in the graph reads from and writes
to — it's the "shape" of the LangGraph box in the architecture diagram.

`messages` uses LangGraph's `add_messages` reducer: instead of a node
returning the full message list, it returns only the *new* messages to
append, and LangGraph merges them in. This is the standard LangGraph
pattern (matches `MessagesState`) and matters once we add branching
nodes in later phases — each node can add messages without needing to
know about or overwrite what other nodes added.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    conversation_id: str
    transcript: str  # the user's message for this turn, from Phase 3
    messages: Annotated[list[BaseMessage], add_messages]  # full conversation history
    response: str  # the assistant's spoken reply for this turn, set by generate_response
    note: str  # short text caption summarizing `response`, set by generate_note (Phase 6)
