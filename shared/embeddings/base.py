"""
Base embedding interface and types for the RAG system.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingType(str, Enum):
    """Supported embedding model types."""
    OPENAI = "openai"
    PUBMEDBERT = "pubmedbert"
    BIOBERT = "biobert"
    SCIBERT = "scibert"
    CLINICALBERT = "clinicalbert"
    BIOLINKBERT = "biolinkbert"
    
    @classmethod
    def get_default(cls) -> "EmbeddingType":
        """Get the default embedding type for healthcare documents."""
        return cls.PUBMEDBERT


@dataclass
class EmbeddingConfig:
    """Configuration for an embedding model."""
    model_name: str
    model_path: str  # HuggingFace model path or identifier
    embedding_dim: int
    max_length: int
    requires_api_key: bool = False
    pooling_strategy: str = "mean"  # mean, cls, max
    normalize: bool = True


# Embedding model configurations
EMBEDDING_CONFIGS: Dict[EmbeddingType, EmbeddingConfig] = {
    EmbeddingType.OPENAI: EmbeddingConfig(
        model_name="text-embedding-ada-002",
        model_path="openai",
        embedding_dim=1536,
        max_length=8191,
        requires_api_key=True,
    ),
    EmbeddingType.PUBMEDBERT: EmbeddingConfig(
        model_name="PubMedBERT",
        model_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        embedding_dim=768,
        max_length=512,
        pooling_strategy="mean",
    ),
    EmbeddingType.BIOBERT: EmbeddingConfig(
        model_name="BioBERT",
        model_path="dmis-lab/biobert-v1.1",
        embedding_dim=768,
        max_length=512,
        pooling_strategy="mean",
    ),
    EmbeddingType.SCIBERT: EmbeddingConfig(
        model_name="SciBERT",
        model_path="allenai/scibert_scivocab_uncased",
        embedding_dim=768,
        max_length=512,
        pooling_strategy="mean",
    ),
    EmbeddingType.CLINICALBERT: EmbeddingConfig(
        model_name="ClinicalBERT",
        model_path="emilyalsentzer/Bio_ClinicalBERT",
        embedding_dim=768,
        max_length=512,
        pooling_strategy="mean",
    ),
    EmbeddingType.BIOLINKBERT: EmbeddingConfig(
        model_name="BioLinkBERT",
        model_path="michiyasunaga/BioLinkBERT-base",
        embedding_dim=768,
        max_length=512,
        pooling_strategy="mean",
    ),
}


class BaseEmbedding(ABC):
    """Abstract base class for all embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model (download if needed, load into memory)."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        return self.config.embedding_dim
    
    @property
    def max_length(self) -> int:
        """Return the maximum input length for this model."""
        return self.config.max_length
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name}, dim={self.embedding_dim})"


def get_collection_name(base_name: str, embedding_type: EmbeddingType) -> str:
    """
    Generate collection name based on base name and embedding type.
    This ensures different embedding types are stored in separate collections.
    """
    return f"{base_name}_{embedding_type.value}"


def parse_collection_name(collection_name: str) -> tuple[str, Optional[EmbeddingType]]:
    """
    Parse collection name to extract base name and embedding type.
    Returns (base_name, embedding_type) or (collection_name, None) if no embedding type found.
    """
    for embedding_type in EmbeddingType:
        suffix = f"_{embedding_type.value}"
        if collection_name.endswith(suffix):
            base_name = collection_name[:-len(suffix)]
            return base_name, embedding_type
    
    return collection_name, None
