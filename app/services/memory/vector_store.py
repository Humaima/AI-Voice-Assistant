"""
Semantic (long-term) memory store — Phase 5, the ChromaDB side of the
diagram's "Memory" box.

Where PostgresConversationStore gives Ava the last N turns verbatim,
this gives her similarity-based recall across a whole conversation —
so if someone mentions their dog's name in turn 2 and asks about it in
turn 40, it can surface even though it's long outside the recent-N
window Postgres alone would return.

`VectorMemory` is a structural Protocol, same pattern as
ConversationMemory and TranscriptionClient elsewhere in this project —
tests inject a fake in-process store instead of exercising real
ChromaDB + embedding infrastructure.
"""
from __future__ import annotations

from typing import Any, Protocol
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VectorMemory(Protocol):
    def add_memory(self, conversation_id: str, role: str, content: str) -> None: ...

    def search(self, conversation_id: str, query: str, k: int) -> list[str]: ...


def build_chroma_client() -> ClientAPI:
    """Real client, pointed at the ChromaDB service from docker-compose.
    Telemetry is disabled — no reason to phone home from a self-hosted
    memory store, and it avoids noisy network warnings in restricted
    environments."""
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


class ChromaVectorStore:
    """Wraps one Chroma collection. `embedding_function` is injectable:
    production leaves it unset and gets Chroma's bundled default
    (all-MiniLM-L6-v2, downloaded from Hugging Face on first use — a
    one-time internet requirement, same as Groq needing a real API key);
    tests inject a tiny deterministic function instead so they don't
    need network access or a large model download."""

    def __init__(
        self,
        client: ClientAPI,
        collection_name: str | None = None,
        embedding_function: Any = None,
    ):
        self._client = client
        name = collection_name or settings.chroma_collection
        kwargs: dict[str, Any] = {}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self._collection = client.get_or_create_collection(name=name, **kwargs)

    def add_memory(self, conversation_id: str, role: str, content: str) -> None:
        if not content.strip():
            return
        doc_id = f"{conversation_id}:{role}:{uuid4().hex}"
        self._collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"conversation_id": conversation_id, "role": role}],
        )
        logger.debug("Indexed memory | conversation=%s | role=%s | chars=%d", conversation_id, role, len(content))

    def search(self, conversation_id: str, query: str, k: int = 3) -> list[str]:
        """Returns up to `k` past messages from this conversation most
        similar to `query`, ranked by relevance. Empty list if nothing
        indexed yet or the query is empty."""
        if not query.strip():
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            where={"conversation_id": conversation_id},
        )
        documents = results.get("documents") or [[]]
        return list(documents[0]) if documents else []
