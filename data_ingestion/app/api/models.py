from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from shared.embeddings import EmbeddingType

class CustomDocumentType(str, Enum):
    """Defines the types of documents that can be ingested."""
    GUIDELINE = "GUIDELINE"
    POLICY = "POLICY"
    FAQ = "FAQ"
    DOCUMENT = "DOCUMENT"


class ChunkerType(str, Enum):
    """Available chunking strategies."""
    GENERIC = "generic"
    HEALTHCARE = "healthcare"
    
    @classmethod
    def get_default(cls) -> "ChunkerType":
        """Get the default chunker type for healthcare documents."""
        return cls.HEALTHCARE

class NewDocumentMetadata(BaseModel):
    """
    Defines the structure for the metadata dictionary accompanying an uploaded file.
    This metadata is expected to be part of a multipart/form-data request,
    likely as a JSON string in a 'metadata' form field, or as individual form fields.
    """
    title: str
    type: CustomDocumentType
    description: str
    date: str  # Example: "2024-06-15"
    embedding_type: Optional[EmbeddingType] = Field(
        default=None,
        description="Embedding model to use. If not specified, uses default (PubMedBERT for healthcare)"
    )
    chunker_type: Optional[ChunkerType] = Field(
        default=None,
        description="Chunking strategy to use. If not specified, uses default (healthcare)"
    )
    enable_ner: Optional[bool] = Field(
        default=False,
        description="Enable NER-based medical entity extraction during chunking"
    )
