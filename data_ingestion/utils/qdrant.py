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
from shared.embeddings import EmbeddingType, EMBEDDING_CONFIGS, get_collection_name

logger = logging.getLogger(__name__)

try:
    client = QdrantClient(url=settings.qdrant_url)
except Exception as e:
    logger.error(f"Error initializing Qdrant client with URL {settings.qdrant_url}: {e}")
    raise e


def init_qdrant_collection(collection_name: str, embedding_type: EmbeddingType):
    """
    Create the collection with appropriate dimensions for the embedding type.
    """
    try:
        # Get the actual collection name (with embedding type suffix)
        actual_collection_name = get_collection_name(collection_name, embedding_type)
        
        # Get embedding dimensions from config
        embedding_config = EMBEDDING_CONFIGS[embedding_type]
        vector_size = embedding_config.embedding_dim
        
        names = [c.name for c in client.get_collections().collections]
        if actual_collection_name not in names:
            client.create_collection(
                collection_name=actual_collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Qdrant: created collection {actual_collection_name} with vector size {vector_size}")
        else:
            logger.debug(f"Qdrant: collection {actual_collection_name} already exists.")
    except Exception as e:
        logger.error(f"Failed to init Qdrant collection {actual_collection_name}: {e}")
        raise


def store_embeddings_in_qdrant(
    chunks: list,
    embeddings: list,
    metadata: dict,
    embedding_type: EmbeddingType,
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
            if not isinstance(chunk_data, dict) or "chunk" not in chunk_data:
                logger.warning(
                    f"Skipping invalid chunk data at index {i} for collection '{collection_name}': {chunk_data}"
                )
                continue
            
            # bbox is optional, but if present must be dict
            bbox = chunk_data.get("bbox")
            if bbox is not None and not isinstance(bbox, dict):
                logger.warning(
                    f"Invalid bbox at index {i}, setting to None: {bbox}"
                )
                bbox = None

            text_content = chunk_data.get("chunk", "").strip()
            if not text_content:
                logger.warning(
                    f"Skipping chunk with empty text content at index {i} for collection '{collection_name}'."
                )
                continue

            payload = {
                "text": text_content,
                "page": chunk_data.get("page", 0),
                "bbox": bbox,  # Use validated bbox
                "parse_type": chunk_data.get("parse_type"),
                "page_width": chunk_data.get("page_width"),
                "page_height": chunk_data.get("page_height"),
                # Add chunk-specific metadata for retrieval
                "chunk_type": chunk_data.get("chunk_type"),
                "answer_types": chunk_data.get("answer_types", []),
                # Lowercase all entity lists for consistent case-insensitive matching
                "drugs": [drug.lower() for drug in (chunk_data.get("drugs") or [])],
                "diseases": [disease.lower() for disease in (chunk_data.get("diseases") or [])],
                "procedures": [procedure.lower() for procedure in (chunk_data.get("procedures") or [])],
                "boost_section": chunk_data.get("boost_section"),
                "section_title": chunk_data.get("section_title"),
                "section_type": chunk_data.get("section_type"),
                "has_medical_content": chunk_data.get("has_medical_content", False),
                **metadata,  # Spread the original metadata
                "embedding_type": embedding_type.value,  # Store embedding type in metadata
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
        # Get the actual collection name (with embedding type suffix)
        actual_collection_name = get_collection_name(collection_name, embedding_type)
        
        response = client.upsert(collection_name=actual_collection_name, points=points)
        logger.info(f"Upserted {len(points)} points into '{actual_collection_name}'")
        return response
    except Exception as e:
        logger.error(f"Failed to upsert points into Qdrant: {e}")
        raise


def chunk_exists_by_metadata(
    metadata_conditions: dict[str, Any],
    embedding_type: EmbeddingType,
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
        # Get the actual collection name (with embedding type suffix)
        actual_collection_name = get_collection_name(collection_name, embedding_type)
        
        # Check if collection exists first
        names = [c.name for c in client.get_collections().collections]
        if actual_collection_name not in names:
            return False
            
        points, _ = client.scroll(
            collection_name=actual_collection_name,
            scroll_filter=filters,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points) > 0
    except Exception as e:
        logger.error(f"Failed to check existence of chunks in Qdrant: {e}", exc_info=True)
        return False
