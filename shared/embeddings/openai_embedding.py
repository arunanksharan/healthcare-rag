"""
OpenAI embedding implementation.
"""
import logging
from typing import List, Optional
from openai import OpenAI

from .base import BaseEmbedding, EmbeddingConfig

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI text-embedding-ada-002 implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.client: Optional[OpenAI] = None
        self.model_name = config.model_path  # For OpenAI, this is the model identifier
    
    def initialize(self) -> None:
        """Initialize OpenAI client."""
        if self._initialized:
            return
        
        try:
            # Try to import from data_ingestion first
            try:
                from data_ingestion.core.settings import settings
            except ImportError:
                # Fall back to data_retrieval if ingestion not available
                from data_retrieval.core.settings import settings
            
            self.client = OpenAI(api_key=settings.openai_api_key)
            self._initialized = True
            logger.info(f"Initialized OpenAI embedding model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI API."""
        if not self._initialized:
            self.initialize()
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.config.model_name
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not self._initialized:
            self.initialize()
        
        try:
            # OpenAI API supports batch embedding
            response = self.client.embeddings.create(
                input=texts,
                model=self.config.model_name
            )
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch OpenAI embeddings: {e}")
            raise
