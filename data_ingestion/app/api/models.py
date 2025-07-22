from enum import Enum
from pydantic import BaseModel

class CustomDocumentType(str, Enum):
    """Defines the types of documents that can be ingested."""
    GUIDELINE = "GUIDELINE"
    POLICY = "POLICY"
    FAQ = "FAQ"
    DOCUMENT = "DOCUMENT"

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
