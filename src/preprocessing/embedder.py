"""
src/preprocessing/embedder.py

Role
----
Turns text into vectors for the RAG layer — seller prediction summaries
and real Olist customer reviews, indexed in ChromaDB and searched at
query time.

Three backends, tried in order
------------------------------
1. **ONNX MiniLM** (`all-MiniLM-L6-v2`, via chromadb's bundled runtime).
   The default. Same model as sentence-transformers, but it runs on
   onnxruntime — tens of megabytes instead of the ~800 MB PyTorch drags
   in. That difference is what makes the API deployable on a free tier at
   all, so the light path is the primary one rather than a compromise.
2. **sentence-transformers**. Used if it is already installed and ONNX is
   unavailable. Identical vectors, much heavier dependency.
3. **TF-IDF**, fit locally on the corpus. Zero network dependency, so RAG
   still works end to end in a sandbox with no model download. Retrieval
   mechanics are identical; semantic quality is plainly worse — it matches
   on shared words, not shared meaning.

Why the backend is written to disk
-----------------------------------
Indexing and querying happen in different processes. If indexing ran with
MiniLM (384-d) and a later query process silently fell back to TF-IDF,
the query vector would land in a different space from the documents.
ChromaDB would either raise on the dimension mismatch or, if the widths
happened to agree, return confidently ranked nonsense. So the backend
chosen at index time is persisted and asserted at query time.

A note on stop words
--------------------
The TF-IDF path deliberately does NOT use `stop_words="english"`. Roughly
half this corpus is Brazilian Portuguese review text, so an English stop
list strips almost nothing from it while removing useful English tokens
from the prediction summaries — the worst of both. Filtering by document
frequency (`max_df`) drops boilerplate in whatever language it appears.
"""

from functools import lru_cache
from pathlib import Path
import json
import joblib
import numpy as np

from src.config_loader import load_config, get_project_root
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

_backend_choice: str | None = None
_tfidf_vectorizer = None
_onnx_fn = None

ONNX = "onnx_minilm"
SBERT = "sentence_transformers"
TFIDF = "tfidf"


def _artifacts_dir() -> Path:
    return get_project_root() / load_config()["paths"]["artifacts"]


def _tfidf_artifact_path() -> Path:
    return _artifacts_dir() / "tfidf_vectorizer.joblib"


def _backend_manifest_path() -> Path:
    return _artifacts_dir() / "embedding_backend.json"


# --------------------------------------------------------------- backends
@lru_cache(maxsize=1)
def _try_onnx():
    """
    chromadb ships an ONNX build of all-MiniLM-L6-v2. Downloads once on
    first use, then runs locally with no torch.
    """
    try:
        from chromadb.utils import embedding_functions
        fn = embedding_functions.ONNXMiniLM_L6_V2()
        fn(["warmup"])                      # force the download/init now, not mid-index
        logger.info("Embedding backend: ONNX MiniLM (all-MiniLM-L6-v2, no torch)")
        return fn
    except Exception as e:
        logger.warning(f"ONNX MiniLM unavailable ({e.__class__.__name__}: {e})")
        return None


@lru_cache(maxsize=1)
def _try_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        model_name = load_config()["rag"]["embedding_model"]
        model = SentenceTransformer(model_name)
        logger.info(f"Embedding backend: sentence-transformers ('{model_name}')")
        return model
    except Exception as e:
        logger.warning(f"sentence-transformers unavailable ({e.__class__.__name__}: {e})")
        return None


def _get_backend() -> str:
    """Decide once per process, cheapest capable backend first."""
    global _backend_choice, _onnx_fn
    if _backend_choice is not None:
        return _backend_choice

    _onnx_fn = _try_onnx()
    if _onnx_fn is not None:
        _backend_choice = ONNX
    elif _try_sentence_transformer() is not None:
        _backend_choice = SBERT
    else:
        logger.warning(
            "No neural embedding backend reachable — falling back to TF-IDF. "
            "Retrieval mechanics are identical; semantic quality is lower. "
            "Resolves automatically once the environment can reach huggingface.co."
        )
        _backend_choice = TFIDF
    return _backend_choice


# ----------------------------------------------------------------- tfidf
def _fit_tfidf(texts: list[str]):
    global _tfidf_vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    # No English stop list — see the module docstring. max_df drops
    # boilerplate that appears in nearly every document (the shared
    # sentence templates) regardless of language.
    _tfidf_vectorizer = TfidfVectorizer(max_features=4096, max_df=0.6, min_df=2)
    _tfidf_vectorizer.fit(texts)

    path = _tfidf_artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_tfidf_vectorizer, path)
    logger.info(f"TF-IDF vectorizer fit on {len(texts):,} documents "
                f"({len(_tfidf_vectorizer.vocabulary_):,} terms), saved to {path}")


def _load_tfidf():
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None and _tfidf_artifact_path().exists():
        _tfidf_vectorizer = joblib.load(_tfidf_artifact_path())
    return _tfidf_vectorizer


# ------------------------------------------------------------- manifest
def _record_backend(backend: str, dim: int) -> None:
    path = _backend_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"backend": backend, "dim": dim}), encoding="utf-8")
    logger.info(f"Index built with backend '{backend}' ({dim}-d) — recorded at {path}")


def indexed_backend() -> dict | None:
    """The backend the current ChromaDB index was built with, if known."""
    path = _backend_manifest_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------- public
def embed_texts(texts: list[str], fit: bool = False) -> np.ndarray:
    """
    Embed a list of strings into an (n_texts, dim) array.

    fit=True is index time: it fits the TF-IDF vectorizer (if that is the
    active backend) and records which backend built the index. fit=False
    is query time and reuses whatever indexing chose.
    """
    try:
        backend = _get_backend()

        if not fit:
            recorded = indexed_backend()
            if recorded and recorded["backend"] != backend:
                raise RuntimeError(
                    f"The index was built with '{recorded['backend']}' but this process "
                    f"resolved '{backend}'. A query embedded in a different vector space "
                    f"from the documents returns meaningless rankings. Rebuild the index "
                    f"with: python -m src.rag.indexer"
                )

        if backend == ONNX:
            vectors = np.asarray(_onnx_fn(texts), dtype=np.float32)
        elif backend == SBERT:
            model = _try_sentence_transformer()
            vectors = np.asarray(
                model.encode(texts, show_progress_bar=False, normalize_embeddings=True))
        else:
            if fit:
                _fit_tfidf(texts)
            vectorizer = _load_tfidf()
            if vectorizer is None:
                raise RuntimeError(
                    "TF-IDF vectorizer not yet fit. Build the index first: "
                    "python -m src.rag.indexer"
                )
            vectors = vectorizer.transform(texts).toarray()

        if fit:
            _record_backend(backend, int(vectors.shape[1]))
        return vectors

    except Exception as e:
        raise SentrixException(e, sys)


def embed_single(text: str) -> np.ndarray:
    """Embed one string — the query-time convenience wrapper."""
    return embed_texts([text], fit=False)[0]


if __name__ == "__main__":
    docs = [
        "Seller 3442f8 in sao paulo, state SP, has a delivery-risk score of 0.71 (critical).",
        "Customer review: produto chegou com atraso de duas semanas, muito insatisfeito.",
    ]
    doc_vectors = embed_texts(docs, fit=True)
    print("Backend in use:", _get_backend())
    print("Doc embeddings shape:", doc_vectors.shape)
    print("Query embedding shape:", embed_single("which sellers are late?").shape)
