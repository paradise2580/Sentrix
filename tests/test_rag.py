"""
tests/test_rag.py — unit tests for the RAG layer (indexer, retriever, chain).

Chunking logic is pure and always tested. Retrieval against the live
ChromaDB index is an integration-flavored test that skips cleanly if the
index hasn't been built yet, rather than failing the suite.
"""

import pytest

from src.rag.indexer import chunk_text


def test_chunk_text_returns_single_chunk_for_short_text():
    text = "Short supplier update."
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_long_text_with_overlap():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # every chunk should be at most chunk_size long
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_text_covers_full_text():
    text = "word " * 200  # 1000 chars
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    # the last character of the original text must appear in the last chunk
    assert text[-5:] in chunks[-1]


@pytest.fixture
def chroma_index_available():
    try:
        from src.rag.indexer import get_chroma_client, _COLLECTION_NAME
        client = get_chroma_client()
        client.get_collection(_COLLECTION_NAME)
        return True
    except Exception:
        return False


def test_retrieve_returns_ranked_results(chroma_index_available):
    if not chroma_index_available:
        pytest.skip("ChromaDB index not built in this environment — run src.rag.indexer first")

    from src.rag.retriever import retrieve
    results = retrieve("supplier risk in Rotterdam", top_k=3)

    assert len(results) <= 3
    assert all("text" in r and "distance" in r for r in results)
    # results should be sorted by ascending distance (most relevant first)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_answer_question_always_returns_grounded_field(chroma_index_available):
    if not chroma_index_available:
        pytest.skip("ChromaDB index not built in this environment — run src.rag.indexer first")

    from src.rag.chain import answer_question
    result = answer_question("What suppliers are at risk?")

    assert "answer" in result
    assert "sources" in result
    assert "grounded" in result
