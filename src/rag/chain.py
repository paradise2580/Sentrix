"""
src/rag/chain.py

Role
----
Ties retrieval + the LLM together into the full RAG flow:

    User question
        -> embed + retrieve top-k relevant chunks from ChromaDB
        -> inject chunks into a grounded prompt
        -> Groq (Llama 3) generates a cited answer
        -> return {answer, sources}

No-key behaviour
-----------------
If GROQ_API_KEY isn't set, this returns the retrieved context directly
with a clear notice instead of crashing — so the retrieval half of RAG
(the part that doesn't need a paid/keyed API) is still fully demonstrable
without a key. Add a real key to .env and the exact same function starts
returning LLM-generated answers with no code changes.
"""

import os
from dotenv import load_dotenv

from src.config_loader import load_config
from src.rag.retriever import retrieve
from src.logger import get_logger
from src.exception import SentrixException
import sys

load_dotenv()
logger = get_logger(__name__)


def _groq_key() -> str:
    """
    Read the key at CALL time, not import time.

    Reading it into a module constant means the value is frozen at the
    first import. That is fine locally, where .env is loaded before
    anything else, and wrong in a container or a test that sets the
    variable after the module graph is already loaded — the key would be
    present in the environment and the code would still report it missing.
    """
    return os.getenv("GROQ_API_KEY", "")


_SYSTEM_PROMPT = """You are SENTRIX, a supply chain risk intelligence assistant.
Answer the user's question using ONLY the context provided below. Be specific,
cite seller IDs, cities, and numbers from the context, and if the context doesn't
contain enough information to answer confidently, say so rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"- {c['text']}" for c in chunks)
    return _SYSTEM_PROMPT.format(context=context, question=question)


def _generate(prompt: str, cfg: dict, groq_key: str):
    """
    Call Groq, trying each configured model in turn.

    Hosted model catalogues are not stable. `llama-3.3-70b-versatile` was
    the configured model here and started returning 404 model_not_found
    when Groq retired it — a dependency that changed underneath the project
    without a single line of code changing. Pinning one model name means
    the demo dies silently the day the provider moves on.

    So `llm_model` is the preference and `llm_fallback_models` is the queue
    behind it. A 404/model-not-found moves to the next candidate; any other
    error (auth, rate limit, network) is raised, because those are real
    failures the caller needs to see rather than route around.
    """
    from langchain_groq import ChatGroq

    candidates = [cfg["llm_model"], *cfg.get("llm_fallback_models", [])]
    for model in candidates:
        try:
            llm = ChatGroq(model=model, temperature=cfg["temperature"], api_key=groq_key)
            return llm.invoke(prompt), model
        except Exception as exc:                                    # noqa: BLE001
            text = str(exc).lower()
            if "model_not_found" in text or "does not exist" in text:
                logger.warning(f"Groq model '{model}' unavailable — trying the next candidate")
                continue
            raise
    return None, None


def answer_question(question: str, top_k: int | None = None) -> dict:
    """
    The full RAG flow. Always retrieves real context from ChromaDB; only
    the generation step depends on a Groq key being present.
    """
    try:
        chunks = retrieve(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "No relevant information found in the knowledge base for this question.",
                "sources": [],
                "grounded": False,
            }

        groq_key = _groq_key()
        if not groq_key:
            logger.warning("GROQ_API_KEY not set — returning retrieved context without LLM generation")
            preview = "\n".join(f"- {c['text']}" for c in chunks)
            return {
                "answer": (
                    "[LLM generation unavailable — GROQ_API_KEY not set in .env. "
                    "Showing the raw retrieved context that WOULD be sent to the LLM:]\n\n"
                    f"{preview}"
                ),
                "sources": chunks,
                "grounded": True,
                "llm_used": False,
            }

        cfg = load_config()["rag"]
        prompt = _build_prompt(question, chunks)
        response, model_used = _generate(prompt, cfg, groq_key)

        if response is None:
            # Every candidate model was rejected by the provider. Retrieval
            # still worked, so degrade to the same grounded-context answer
            # the no-key path returns rather than failing the request.
            logger.warning("No configured Groq model was available — "
                           "returning retrieved context without generation")
            preview = "\n".join(f"- {c['text']}" for c in chunks)
            return {
                "answer": ("[LLM generation unavailable — none of the configured Groq "
                           "models are currently served. Showing the retrieved context "
                           f"that WOULD be sent to the LLM:]\n\n{preview}"),
                "sources": chunks,
                "grounded": True,
                "llm_used": False,
            }

        logger.info(f"RAG answer generated by '{model_used}' for: '{question[:60]}...'")
        return {
            "answer": response.content,
            "sources": chunks,
            "grounded": True,
            "llm_used": True,
            "llm_model": model_used,
        }
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    result = answer_question("Which sellers have critical delivery risk, and why?")
    print("ANSWER:\n", result["answer"])
    print(f"\nGrounded: {result['grounded']}, LLM used: {result.get('llm_used', False)}")
    print(f"Sources: {len(result['sources'])} chunks retrieved")
