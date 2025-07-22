import logging
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from data_retrieval.core.settings import settings

logger = logging.getLogger(__name__)

try:
    client = QdrantClient(url=settings.qdrant_url)
except Exception as e:
    logger.error(f"Error initializing Qdrant client with URL {settings.qdrant_url}: {e}")
    raise


def search_with_metadata_and_embedding(
    embedding: list[float],
    metadata: Optional[dict[str, Any]],
    collection_name: str = settings.qdrant_collection_name,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """
    1) Optionally builds metadata Filter if metadata is provided.
    2) Vector-searches Qdrant.
    3) Returns for each hit:
        - content: the chunk text
        - embedding_score: the similarity score
        - metadata: the full payload you upserted (page, bbox + all the ingestion metadata)
    """
    try:
        must_conditions = []
        if metadata:
            for key in [
                "type",
            ]:
                value = metadata.get(key)
                if not value:
                    continue

                elif isinstance(value, list):
                    cond = models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value) if len(value) > 1 else models.MatchValue(value=value[0]),
                    )
                else:
                    cond = models.FieldCondition(key=key, match=models.MatchValue(value=value))
                must_conditions.append(cond)

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        hits = client.search(
            collection_name=collection_name,
            query_vector=embedding,
            query_filter=query_filter,
            limit=top_k,
        )

        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            # pull the text out
            text = payload.get("text", "")
            # keep the entire payload as metadata
            metadata_full = payload.copy()
            if "text" in metadata_full:
                del metadata_full["text"]
            results.append(
                {
                    "content": text,
                    "embedding_score": hit.score,
                    "metadata": metadata_full,
                }
            )

        return results

    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        return []
