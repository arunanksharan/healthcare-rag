import logging
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
try:
    reranker = CrossEncoder(MODEL)
except Exception as e:
    logger.error(f"Failed to load CrossEncoder {MODEL}: {e}")
    reranker = None

def rerank_documents(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: int = None
) -> List[Dict[str, Any]]:
    """
    Batch-scores and sorts docs by relevance to the query.
    Expects each doc to have a 'content' field.
    """
    if not reranker or not docs:
        return docs

    texts = [doc["content"] for doc in docs]  # directly use 'content'

    # Pair query with each document text
    pairs = [(query, text) for text in texts]

    try:
        scores = reranker.predict(pairs, batch_size=32)
    except Exception as e:
        logger.error(f"Reranker inference failed: {e}")
        return docs

    # Sort docs by descending score
    ranked_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    ranked_docs = [doc for doc, _ in ranked_pairs]

    return ranked_docs[:top_k] if top_k else ranked_docs