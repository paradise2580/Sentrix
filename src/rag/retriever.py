"""
src/rag/retriever.py

Role
----
Given a user question, embed it into the same vector space the index was
built in, and return the top-k most relevant document chunks from
ChromaDB. This is the "R" (retrieval) in RAG.
"""

from src.config_loader import load_config
from src.rag.indexer import get_chroma_client, _COLLECTION_NAME
from src.preprocessing.embedder import embed_single
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def retrieve(query: str, top_k: int | None = None, filter_metadata: dict | None = None) -> list[dict]:
    """
    Semantic search over the indexed knowledge base.

    Returns a list of {text, metadata, distance} dicts, ranked by
    relevance (lowest distance first).
    """
    try:
        cfg = load_config()["rag"]
        top_k = top_k or cfg["top_k"]

        client = get_chroma_client()
        collection = client.get_collection(_COLLECTION_NAME)

        query_embedding = embed_single(query)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata,
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        logger.info(f"Retrieved {len(retrieved)} chunks for query: '{query[:60]}...'")
        return retrieved
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    results = retrieve("Which sellers have critical delivery risk?")
    for r in results:
        print(f"[{r['distance']:.4f}] {r['text']}")
