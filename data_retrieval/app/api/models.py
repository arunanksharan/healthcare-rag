from typing import Optional, List
from pydantic import BaseModel, Field

from shared.embeddings import EmbeddingType


class SearchRequest(BaseModel):
    query: str
    embedding_types: Optional[List[EmbeddingType]] = Field(
        default=None,
        description="List of embedding types to search across. If not specified, searches all available collections."
    )
