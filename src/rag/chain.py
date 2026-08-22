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

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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

        if not _GROQ_API_KEY:
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

        from langchain_groq import ChatGroq

        cfg = load_config()["rag"]
        llm = ChatGroq(
            model=cfg["llm_model"], temperature=cfg["temperature"], api_key=_GROQ_API_KEY,
        )

        prompt = _build_prompt(question, chunks)
        response = llm.invoke(prompt)

        logger.info(f"RAG answer generated for: '{question[:60]}...'")
        return {
            "answer": response.content,
            "sources": chunks,
            "grounded": True,
            "llm_used": True,
        }
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    result = answer_question("Why might Supplier_048 be at risk?")
    print("ANSWER:\n", result["answer"])
    print(f"\nGrounded: {result['grounded']}, LLM used: {result.get('llm_used', False)}")
    print(f"Sources: {len(result['sources'])} chunks retrieved")
