"""
Tests for ChromaVectorStore (Phase 5).

Uses chromadb.EphemeralClient (fully in-memory, no server/Docker
needed) with a small deterministic embedding function instead of
Chroma's real default (which downloads a model from Hugging Face on
first use — not something a test suite should depend on). The fake
embedding is a simple bag-of-words hash vector: good enough to make
"similar text -> similar vector" hold for these tests, without any ML
model or network access.
"""
import re

import chromadb
import numpy as np
import pytest

from app.services.memory.vector_store import ChromaVectorStore


class HashBagOfWordsEmbedding:
    """Deterministic, dependency-free stand-in for a real embedding
    model. Each word hashes to a fixed dimension and gets added into a
    shared vector — texts sharing more words end up with more similar
    vectors. Not remotely state-of-the-art, but sufficient to test that
    ChromaVectorStore's add/search plumbing works correctly."""

    _DIM = 64

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def name(self) -> str:
        return "hash-bag-of-words-test-embedding"

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self._DIM, dtype=float)
        for word in re.findall(r"[a-z]+", text.lower()):
            idx = hash(word) % self._DIM
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


@pytest.fixture
def vector_store():
    client = chromadb.EphemeralClient()
    # Unique collection name per test run avoids collisions if
    # EphemeralClient's process-level state is ever reused.
    import uuid

    collection_name = f"test-{uuid.uuid4().hex}"
    return ChromaVectorStore(client, collection_name=collection_name, embedding_function=HashBagOfWordsEmbedding())


class TestChromaVectorStore:
    def test_search_on_empty_store_returns_empty_list(self, vector_store):
        results = vector_store.search("conv-1", "anything", k=3)
        assert results == []

    def test_search_empty_query_returns_empty_list_without_querying(self, vector_store):
        vector_store.add_memory("conv-1", "user", "my dog is named Rex")
        results = vector_store.search("conv-1", "   ", k=3)
        assert results == []

    def test_finds_relevant_memory_by_similarity(self, vector_store):
        vector_store.add_memory("conv-1", "user", "my dog is named Rex")
        vector_store.add_memory("conv-1", "assistant", "Rex sounds like a great name!")
        vector_store.add_memory("conv-1", "user", "I had pasta for dinner")

        results = vector_store.search("conv-1", "what is my dog's name", k=2)

        assert any("Rex" in r for r in results)

    def test_conversations_are_isolated(self, vector_store):
        vector_store.add_memory("conv-1", "user", "my dog is named Rex")
        vector_store.add_memory("conv-2", "user", "my cat is named Whiskers")

        results = vector_store.search("conv-1", "dog Rex", k=5)

        assert all("Whiskers" not in r for r in results)

    def test_empty_content_is_not_stored(self, vector_store):
        vector_store.add_memory("conv-1", "user", "   ")
        results = vector_store.search("conv-1", "anything", k=5)
        assert results == []

    def test_respects_k_limit(self, vector_store):
        for i in range(5):
            vector_store.add_memory("conv-1", "user", f"fact number {i} about dogs")

        results = vector_store.search("conv-1", "dogs", k=2)

        assert len(results) == 2
