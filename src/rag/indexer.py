"""
src/rag/indexer.py

Role
----
Builds the searchable knowledge base: chunks documents into passages,
embeds them (via src/preprocessing/embedder.py — sentence-transformers
or TF-IDF fallback), and stores them in ChromaDB.

Document sources indexed
-------------------------
- News/signal text generated during ingestion
- SHAP explanations from evaluation (turned into readable sentences), so the
  RAG chat can cite the MODEL's own reasoning, not just raw news
- Supplier metadata (region, category, reliability) for grounding
"""

import json
import chromadb

from src.config_loader import load_config, get_project_root
from src.preprocessing.embedder import embed_texts
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

_COLLECTION_NAME = "sentrix_knowledge_base"


def get_chroma_client():
    cfg = load_config()
    persist_dir = get_project_root() / cfg["paths"]["chroma_db"]
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window character chunking with overlap."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def build_documents_from_predictions() -> list[dict]:
    """
    Turns each seller's stored prediction (the predictions table in MySQL)
    into a readable text document — this is what lets the RAG chat answer
    "why is Supplier X risky?" grounded in the model's actual SHAP output.
    """
    try:
        from src.ingestion.loader import DataLoader
        loader = DataLoader()
        predictions = loader.read_table("predictions")
        sellers = loader.read_table("sellers")

        merged = predictions.merge(sellers, on="seller_id", how="left")

        documents = []
        for _, row in merged.iterrows():
            top_features = json.loads(row["top_features"]) if row["top_features"] else []
            feature_text = "; ".join(
                f"{f['feature']} ({'raises' if f['contribution'] > 0 else 'lowers'} risk, "
                f"impact {abs(f['contribution']):.2f})"
                for f in top_features
            )
            text = (
                f"Seller {row['seller_id']} in {row['seller_city']}, "
                f"state {row['seller_state']}, has a current delivery-risk score of "
                f"{row['risk_score']:.2f} ({row['risk_band']} risk). "
                f"Model used: {row['model_name']}. "
                f"Key contributing factors: {feature_text}."
            )
            documents.append({
                "id": f"prediction_{row['seller_id']}",
                "text": text,
                "metadata": {
                    "seller_id": str(row["seller_id"]),
                    "source": "prediction",
                    "risk_band": row["risk_band"],
                },
            })
        return documents
    except Exception as e:
        raise SentrixException(e, sys)


def build_documents_from_reviews(limit: int = 2000) -> list[dict]:
    """
    Turns REAL customer reviews into documents, so the RAG chat can ground
    answers in what customers actually wrote about a seller's deliveries.
    """
    try:
        from src.ingestion.loader import DataLoader
        loader = DataLoader()
        rows = loader.read_query(f"""
            SELECT oi.seller_id, s.seller_state, r.review_score,
                   r.review_comment_message, DATE(r.review_creation_date) AS review_date
            FROM reviews r
            JOIN order_items oi ON oi.order_id = r.order_id
            JOIN sellers s      ON s.seller_id = oi.seller_id
            WHERE r.review_comment_message IS NOT NULL
              AND r.review_comment_message <> ''
            ORDER BY r.review_creation_date DESC
            LIMIT {limit}
        """)

        documents = []
        for i, row in rows.iterrows():
            sentiment = ("negative" if row["review_score"] <= 2
                         else "positive" if row["review_score"] >= 4 else "neutral")
            text = (
                f"Customer review for seller {row['seller_id']} (state {row['seller_state']}) "
                f"on {row['review_date']}: {row['review_score']}/5 stars ({sentiment}). "
                f"Comment: {str(row['review_comment_message'])[:300]}"
            )
            documents.append({
                "id": f"review_{i}_{row['seller_id']}",
                "text": text,
                "metadata": {"seller_id": str(row["seller_id"]), "source": "customer_review"},
            })
        return documents
    except Exception as e:
        raise SentrixException(e, sys)


def build_index(documents: list[dict] = None, reset: bool = True) -> chromadb.Collection:
    """
    Embed and store documents in ChromaDB. If no documents are passed,
    builds the default knowledge base from predictions + recent signals.
    """
    try:
        cfg = load_config()["rag"]
        client = get_chroma_client()

        if reset:
            try:
                client.delete_collection(_COLLECTION_NAME)
            except Exception:
                pass
        collection = client.get_or_create_collection(_COLLECTION_NAME)

        if documents is None:
            documents = build_documents_from_predictions() + build_documents_from_reviews()

        # Chunk long documents (predictions/signals here are short, but this
        # keeps the pipeline correct for longer real-world documents too)
        all_chunks, all_ids, all_metadata = [], [], []
        for doc in documents:
            chunks = chunk_text(doc["text"], cfg["chunk_size"], cfg["chunk_overlap"])
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{doc['id']}_chunk{i}")
                all_metadata.append(doc["metadata"])

        logger.info(f"Embedding {len(all_chunks)} chunks...")
        embeddings = embed_texts(all_chunks, fit=True)

        collection.add(
            ids=all_ids,
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadata,
        )
        logger.info(f"Indexed {len(all_chunks)} chunks into ChromaDB collection '{_COLLECTION_NAME}'")
        return collection
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    build_index()
