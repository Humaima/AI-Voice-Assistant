"""
LLM factory (Phase 4).

Centralizes construction of the Groq-backed chat model so every node
that needs an LLM gets the same configuration, and so tests can swap in
a fake model in one place (see build_graph's `llm` parameter in
app/agents/graph.py) instead of patching this module.
"""
from langchain_groq import ChatGroq
from pydantic import SecretStr

from app.core.config import get_settings

settings = get_settings()


def get_llm(temperature: float = 0.4) -> ChatGroq:
    """Llama-3.3-70B on Groq — the 'Conversational Chain' box in the
    diagram. temperature=0.4 balances natural-sounding replies against
    staying on-topic; lower than a general chatbot default since
    responses get spoken aloud and rambling is more noticeable there
    than in text."""
    return ChatGroq(
        model=settings.groq_llm_model,
        api_key=SecretStr(settings.groq_api_key) if settings.groq_api_key else None,
        temperature=temperature,
    )
