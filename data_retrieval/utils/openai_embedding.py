import logging
from typing import Dict, List, Optional

from openai import OpenAI

from data_retrieval.core.settings import settings
from shared.embeddings import EmbeddingType, get_embedding_model

logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.openai_api_key)


def get_openai_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    """
    Generate an embedding for the input text using OpenAI's embedding model.
    Kept for backward compatibility.
    """
    try:
        response = client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error("Error generating OpenAI embedding: %s", e)
        raise e


def get_embedding_for_query(text: str, embedding_type: EmbeddingType) -> List[float]:
    """
    Generate an embedding for the query text using the specified embedding model.
    
    Args:
        text: Query text to embed
        embedding_type: Type of embedding model to use
        
    Returns:
        Embedding vector as list of floats
    """
    try:
        embedding_model = get_embedding_model(embedding_type)
        return embedding_model.embed_text(text)
    except Exception as e:
        logger.error(f"Error generating {embedding_type.value} embedding: {e}")
        raise


def get_embeddings_for_multiple_types(
    text: str, 
    embedding_types: Optional[List[EmbeddingType]] = None
) -> Dict[EmbeddingType, List[float]]:
    """
    Generate embeddings for the same text using multiple embedding models.
    
    Args:
        text: Query text to embed
        embedding_types: List of embedding types to use. If None, uses all available.
        
    Returns:
        Dict mapping embedding type to its embedding vector
    """
    if embedding_types is None:
        # Default to all embedding types
        embedding_types = list(EmbeddingType)
    
    embeddings = {}
    for embedding_type in embedding_types:
        try:
            embedding = get_embedding_for_query(text, embedding_type)
            embeddings[embedding_type] = embedding
        except Exception as e:
            logger.warning(f"Could not generate {embedding_type.value} embedding: {e}")
            # Continue with other embeddings
    
    return embeddings
