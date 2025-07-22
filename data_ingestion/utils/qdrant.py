import logging
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ..core.settings import settings

logger = logging.getLogger(__name__)

try:
    client = QdrantClient(url=settings.qdrant_url)
except Exception as e:
    logger.error(f"Error initializing Qdrant client with URL {settings.qdrant_url}: {e}")
    raise e


def init_qdrant_collection(collection_name: str):
    """
    Create the collection (idempotent).
    """
    try:
        names = [c.name for c in client.get_collections().collections]
        if collection_name not in names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            logger.info(f"Qdrant: created collection {collection_name} using provided client.")
        else:
            logger.debug(f"Qdrant: collection {collection_name} already exists.")
    except Exception as e:
        logger.error(f"Failed to init Qdrant collection {collection_name}: {e}")
        raise


def store_embeddings_in_qdrant(
    chunks: list,
    embeddings: list,
    metadata: dict,
    collection_name: str = settings.qdrant_collection_name,
):
    """
    Upsert each chunk+embedding into Qdrant with payload:
    - text: the chunk text
    - plus all user metadata keys
    """
    if len(chunks) != len(embeddings):
        logger.error(
            f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch for collection '{collection_name}'."  # noqa: E501
        )
        raise ValueError("Mismatched lengths between chunks and embeddings.")

    points = []
    for i, (chunk_data, emb) in enumerate(zip(chunks, embeddings, strict=False)):
        try:
            # Validate chunk_data structure
            if (
                not isinstance(chunk_data, dict)
                or not all(key in chunk_data for key in ["chunk", "page", "bbox"])
                or not isinstance(chunk_data["bbox"], dict)
            ):
                logger.warning(
                    f"Skipping invalid chunk data at index {i} for collection '{collection_name}': {chunk_data}"
                )
                continue

            text_content = chunk_data.get("chunk", "").strip()
            if not text_content:
                logger.warning(
                    f"Skipping chunk with empty text content at index {i} for collection '{collection_name}'."
                )
                continue

            payload = {
                "text": text_content,
                "page": chunk_data.get("page"),
                "bbox": chunk_data.get("bbox"),
                "parse_type": chunk_data.get("parse_type"),
                "page_width": chunk_data.get("page_width"),
                "page_height": chunk_data.get("page_height"),
                **metadata,  # Spread the original metadata
            }
            points.append(PointStruct(id=str(uuid4()), vector=emb, payload=payload))

        except Exception as inner_exc:
            logger.error(
                f"Error building point for chunk at index {i} for collection '{collection_name}': {inner_exc}",
                exc_info=True,
            )

    if not points:
        logger.warning(f"No valid points were generated to upsert into Qdrant collection '{collection_name}'.")
        return

    try:
        response = client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} points into '{collection_name}'")
        return response
    except Exception as e:
        logger.error(f"Failed to upsert points into Qdrant: {e}")
        raise


def chunk_exists_by_metadata(
    metadata_conditions: dict[str, Any],
    collection_name: str = settings.qdrant_collection_name,
) -> bool:
    """
    Returns True if at least one chunk exists in Qdrant that matches the given metadata conditions.
    """
    must_conditions = []

    for key, value in metadata_conditions.items():
        must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    filters = Filter(must=must_conditions)

    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=filters,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points) > 0
    except Exception as e:
        logger.error(f"Failed to check existence of chunks in Qdrant: {e}", exc_info=True)
        return False
