"""
Embedding model registry for managing different embedding implementations.
"""
import logging
from typing import Dict, Optional, Type
from threading import Lock

from .base import BaseEmbedding, EmbeddingType, EMBEDDING_CONFIGS
from .openai_embedding import OpenAIEmbedding
from .transformer_embeddings import (
    PubMedBERTEmbedding,
    BioBERTEmbedding,
    SciBERTEmbedding,
    ClinicalBERTEmbedding,
    BioLinkBERTEmbedding,
)

logger = logging.getLogger(__name__)


class EmbeddingRegistry:
    """
    Singleton registry for managing embedding model instances.
    Provides lazy loading and caching of models.
    """
    
    _instance: Optional["EmbeddingRegistry"] = None
    _lock = Lock()
    
    def __new__(cls) -> "EmbeddingRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._embedding_classes: Dict[EmbeddingType, Type[BaseEmbedding]] = {
                EmbeddingType.OPENAI: OpenAIEmbedding,
                EmbeddingType.PUBMEDBERT: PubMedBERTEmbedding,
                EmbeddingType.BIOBERT: BioBERTEmbedding,
                EmbeddingType.SCIBERT: SciBERTEmbedding,
                EmbeddingType.CLINICALBERT: ClinicalBERTEmbedding,
                EmbeddingType.BIOLINKBERT: BioLinkBERTEmbedding,
            }
            self._embedding_instances: Dict[EmbeddingType, BaseEmbedding] = {}
            self._initialized = True
            logger.info("EmbeddingRegistry initialized")
    
    def get_embedding(self, embedding_type: EmbeddingType) -> BaseEmbedding:
        """
        Get or create an embedding model instance.
        Models are lazily loaded and cached.
        """
        if embedding_type not in self._embedding_instances:
            with self._lock:
                if embedding_type not in self._embedding_instances:
                    self._create_embedding_instance(embedding_type)
        
        return self._embedding_instances[embedding_type]
    
    def _create_embedding_instance(self, embedding_type: EmbeddingType) -> None:
        """Create and initialize an embedding model instance."""
        if embedding_type not in self._embedding_classes:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        if embedding_type not in EMBEDDING_CONFIGS:
            raise ValueError(f"No configuration found for embedding type: {embedding_type}")
        
        logger.info(f"Creating embedding instance for {embedding_type.value}")
        
        config = EMBEDDING_CONFIGS[embedding_type]
        embedding_class = self._embedding_classes[embedding_type]
        
        # Create instance
        instance = embedding_class(config)
        
        # Initialize (download models, etc.)
        try:
            instance.initialize()
            self._embedding_instances[embedding_type] = instance
            logger.info(f"Successfully initialized {embedding_type.value} embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize {embedding_type.value} embedding model: {e}")
            raise
    
    def preload_models(self, embedding_types: Optional[list[EmbeddingType]] = None) -> None:
        """
        Preload specified embedding models into memory.
        If no types specified, preloads the default model.
        """
        if embedding_types is None:
            embedding_types = [EmbeddingType.get_default()]
        
        for embedding_type in embedding_types:
            try:
                self.get_embedding(embedding_type)
                logger.info(f"Preloaded {embedding_type.value} embedding model")
            except Exception as e:
                logger.error(f"Failed to preload {embedding_type.value}: {e}")
    
    @classmethod
    def get_instance(cls) -> "EmbeddingRegistry":
        """Get the singleton instance of the embedding registry."""
        return cls()


# Convenience function
def get_embedding_model(embedding_type: EmbeddingType) -> BaseEmbedding:
    """Get an embedding model instance from the registry."""
    return EmbeddingRegistry.get_instance().get_embedding(embedding_type)
