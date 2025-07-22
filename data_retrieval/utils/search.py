import logging
from typing import Any, Optional, List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http import models

from data_retrieval.core.settings import settings
from shared.embeddings import EmbeddingType, get_collection_name
from shared.query_analysis import QueryIntent

logger = logging.getLogger(__name__)

try:
    client = QdrantClient(url=settings.qdrant_url)
except Exception as e:
    logger.error(f"Error initializing Qdrant client with URL {settings.qdrant_url}: {e}")
    raise


def search_single_collection(
    embedding: list[float],
    collection_name: str,
    metadata: Optional[dict[str, Any]] = None,
    top_k: int = 50,
    intent_filters: Optional[Dict[str, Any]] = None
) -> list[dict[str, Any]]:
    """
    Search a single Qdrant collection with the given embedding.
    Returns list of results with content, score, and metadata.
    
    Args:
        embedding: Query embedding vector
        collection_name: Name of the collection to search
        metadata: Optional metadata filters
        top_k: Number of results to retrieve
        intent_filters: Optional intent-based filters (chunk_types, boost_sections)
    """
    try:
        must_conditions = []
        should_conditions = []
        
        # Add metadata filters
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
        
        # Add intent-based filters
        if intent_filters:
            # Filter by chunk types if specified
            if "chunk_types" in intent_filters:
                chunk_type_cond = models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchAny(any=intent_filters["chunk_types"])
                )
                should_conditions.append(chunk_type_cond)
            
            # Note: We don't filter by boost_sections, only use them for scoring
            # This ensures we don't exclude relevant content
        
        # Build query filter
        filter_conditions = []
        if must_conditions:
            filter_conditions.extend(must_conditions)
        
        # For should conditions, we want to boost but not exclude
        # So we'll handle them differently in scoring
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

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
            
            # Apply intent-based score boosting
            score = hit.score
            if intent_filters and should_conditions:
                # Boost score if chunk matches preferred types or sections
                if "chunk_types" in intent_filters:
                    if metadata_full.get("chunk_type") in intent_filters["chunk_types"]:
                        score *= 1.2  # 20% boost for matching chunk type
                
                if "boost_sections" in intent_filters:
                    # Check multiple fields for section matching
                    boost_section = metadata_full.get("boost_section")
                    answer_types = metadata_full.get("answer_types", [])
                    
                    # Apply boost if any field matches
                    for section in intent_filters["boost_sections"]:
                        if boost_section == section or section in answer_types:
                            score *= 1.3  # 30% boost for matching section
                            break  # Only apply boost once
            
            results.append(
                {
                    "content": text,
                    "embedding_score": score,
                    "original_score": hit.score,  # Keep original for debugging
                    "metadata": metadata_full,
                    "collection_name": collection_name,  # Track which collection this came from
                }
            )
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x["embedding_score"], reverse=True)

        return results

    except Exception as e:
        logger.error(f"Error during search in collection {collection_name}: {e}", exc_info=True)
        return []


def search_with_metadata_and_embedding(
    embedding: list[float],
    metadata: Optional[dict[str, Any]],
    embedding_type: Optional[EmbeddingType] = None,
    collection_name: str = settings.qdrant_collection_name,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """
    Search with a specific embedding type.
    For backward compatibility, this wraps search_single_collection.
    """
    if embedding_type:
        actual_collection = get_collection_name(collection_name, embedding_type)
    else:
        # Default to searching the base collection (backward compatibility)
        actual_collection = collection_name
    
    return search_single_collection(embedding, actual_collection, metadata, top_k)


def search_multiple_collections(
    query_embeddings: dict[EmbeddingType, list[float]],
    metadata: Optional[dict[str, Any]] = None,
    base_collection_name: str = settings.qdrant_collection_name,
    top_k_per_collection: int = 50,
) -> list[dict[str, Any]]:
    """
    Search across multiple collections with different embeddings.
    
    Args:
        query_embeddings: Dict mapping embedding type to its query embedding
        metadata: Optional metadata filters
        base_collection_name: Base collection name (without embedding suffix)
        top_k_per_collection: Number of results to retrieve from each collection
        
    Returns:
        Combined and sorted results from all collections
    """
    all_results = []
    
    # Get existing collections
    try:
        existing_collections = {c.name for c in client.get_collections().collections}
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        return []
    
    for embedding_type, query_embedding in query_embeddings.items():
        collection_name = get_collection_name(base_collection_name, embedding_type)
        
        # Skip if collection doesn't exist
        if collection_name not in existing_collections:
            logger.info(f"Collection {collection_name} does not exist, skipping")
            continue
        
        # Search this collection
        results = search_single_collection(
            query_embedding,
            collection_name,
            metadata,
            top_k_per_collection
        )
        
        # Add embedding type to metadata for tracking
        for result in results:
            result["metadata"]["embedding_type"] = embedding_type.value
        
        all_results.extend(results)
    
    # Sort all results by score (descending)
    all_results.sort(key=lambda x: x["embedding_score"], reverse=True)
    
    # Return top results across all collections
    return all_results[:top_k_per_collection]
