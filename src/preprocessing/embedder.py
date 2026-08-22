"""
src/preprocessing/embedder.py

Role
----
Converts text documents (news articles, port bulletins, disruption
reports) into vector embeddings for the RAG layer (the ChromaDB
index and retriever).

Dual backend
------------
Primary:  sentence-transformers ("all-MiniLM-L6-v2", set in config.yaml)
          — the production path. Requires downloading model weights from
          Hugging Face on first use.
Fallback: a locally-fit TF-IDF vectorizer — zero network dependency.
          Used automatically if the sentence-transformers model can't be
          reached (e.g. a network-restricted environment). This keeps
          RAG genuinely functional end-to-end even without external
          model access — the retrieval mechanics (chunk, embed, index,
          similarity search) are identical either way; only the quality
          of the semantic representation differs.

Which backend is active is logged once at first use and never silently
swapped mid-session, so indexing and querying always use the same
vector space.
"""

from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np

from src.config_loader import load_config, get_project_root
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

_backend_choice: str | None = None   # "sentence_transformers" | "tfidf", decided once
_tfidf_vectorizer = None


def _tfidf_artifact_path() -> Path:
    cfg = load_config()
    return get_project_root() / cfg["paths"]["artifacts"] / "tfidf_vectorizer.joblib"


@lru_cache(maxsize=1)
def _try_load_sentence_transformer():
    """
    Attempt to load the configured sentence-transformers model. Returns
    None (rather than raising) if it can't be reached — the caller
    decides what to do with that, and the failed attempt is cached so we
    don't retry a slow network timeout on every single call.
    """
    try:
        from sentence_transformers import SentenceTransformer
        cfg = load_config()
        model_name = cfg["rag"]["embedding_model"]
        model = SentenceTransformer(model_name)
        logger.info(f"Embedding backend: sentence-transformers ('{model_name}')")
        return model
    except Exception as e:
        logger.warning(
            f"sentence-transformers model unavailable ({e.__class__.__name__}: {e}). "
            "Falling back to local TF-IDF embeddings — retrieval mechanics are "
            "identical, semantic quality is lower. Resolves automatically once "
            "the environment has normal access to huggingface.co."
        )
        return None


def _get_backend() -> str:
    global _backend_choice
    if _backend_choice is None:
        _backend_choice = "sentence_transformers" if _try_load_sentence_transformer() else "tfidf"
        if _backend_choice == "tfidf":
            logger.info("Embedding backend: TF-IDF (offline fallback)")
    return _backend_choice


def _fit_tfidf(texts: list[str]):
    """Fit (or re-fit) the TF-IDF vectorizer on a corpus and persist it to disk."""
    global _tfidf_vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    _tfidf_vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
    _tfidf_vectorizer.fit(texts)

    path = _tfidf_artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_tfidf_vectorizer, path)
    logger.info(f"TF-IDF vectorizer fit on {len(texts)} documents, saved to {path}")


def _load_tfidf():
    global _tfidf_vectorizer
    if _tfidf_vectorizer is not None:
        return _tfidf_vectorizer
    path = _tfidf_artifact_path()
    if path.exists():
        _tfidf_vectorizer = joblib.load(path)
        return _tfidf_vectorizer
    return None


def embed_texts(texts: list[str], fit: bool = False) -> np.ndarray:
    """
    Embed a list of strings into an (n_texts, embedding_dim) array.

    Parameters
    ----------
    fit : bool
        Only relevant for the TF-IDF fallback. Pass True when indexing a
        new corpus (fits the vectorizer on these texts). Pass False at
        query time, reusing the vectorizer already fit during indexing —
        a query must land in the same vector space as the documents.
    """
    try:
        backend = _get_backend()

        if backend == "sentence_transformers":
            model = _try_load_sentence_transformer()
            embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(embeddings)

        # TF-IDF fallback
        if fit:
            _fit_tfidf(texts)
        vectorizer = _load_tfidf()
        if vectorizer is None:
            raise RuntimeError(
                "TF-IDF vectorizer not yet fit. Call embed_texts(docs, fit=True) "
                "on the document corpus before embedding queries."
            )
        return vectorizer.transform(texts).toarray()

    except Exception as e:
        raise SentrixException(e, sys)


def embed_single(text: str) -> np.ndarray:
    """Embed a single string — convenience wrapper for query-time embedding."""
    return embed_texts([text], fit=False)[0]


if __name__ == "__main__":
    docs = ["Port congestion rising near Shanghai.", "Factory reopens after flood."]
    doc_vectors = embed_texts(docs, fit=True)
    print("Backend in use:", _get_backend())
    print("Doc embeddings shape:", doc_vectors.shape)

    query_vector = embed_single("Is the Shanghai port affected?")
    print("Query embedding shape:", query_vector.shape)
