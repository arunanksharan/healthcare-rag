"""
Base classes and types for medical NER.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class MedicalEntityType(str, Enum):
    """Types of medical entities recognized by NER models."""
    DRUG = "drug"
    DISEASE = "disease"
    SYMPTOM = "symptom"
    PROCEDURE = "procedure"
    TEST = "test"
    ANATOMY = "anatomy"
    DOSAGE = "dosage"
    FREQUENCY = "frequency"
    DURATION = "duration"
    ROUTE = "route"
    STRENGTH = "strength"
    GENE_PROTEIN = "gene_protein"
    CHEMICAL = "chemical"
    OTHER = "other"

    @classmethod
    def from_bio_tag(cls, tag: str) -> "MedicalEntityType":
        """Convert common BIO tags to our entity types."""
        tag_lower = tag.lower()
        
        # Common biomedical NER tag mappings
        mappings = {
            "drug": cls.DRUG,
            "medication": cls.DRUG,
            "medicine": cls.DRUG,
            "disease": cls.DISEASE,
            "disorder": cls.DISEASE,
            "condition": cls.DISEASE,
            "problem": cls.DISEASE,
            "symptom": cls.SYMPTOM,
            "sign": cls.SYMPTOM,
            "procedure": cls.PROCEDURE,
            "treatment": cls.PROCEDURE,
            "test": cls.TEST,
            "lab": cls.TEST,
            "anatomy": cls.ANATOMY,
            "body_part": cls.ANATOMY,
            "organ": cls.ANATOMY,
            "dosage": cls.DOSAGE,
            "dose": cls.DOSAGE,
            "frequency": cls.FREQUENCY,
            "duration": cls.DURATION,
            "route": cls.ROUTE,
            "strength": cls.STRENGTH,
            "gene": cls.GENE_PROTEIN,
            "protein": cls.GENE_PROTEIN,
            "chemical": cls.CHEMICAL,
        }
        
        for key, entity_type in mappings.items():
            if key in tag_lower:
                return entity_type
        
        return cls.OTHER


@dataclass
class MedicalEntity:
    """Represents a medical entity found in text."""
    text: str
    entity_type: MedicalEntityType
    start_char: int
    end_char: int
    confidence: float
    # Optional fields for enriched information
    normalized_text: Optional[str] = None
    umls_cui: Optional[str] = None  # UMLS Concept Unique Identifier
    synonyms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for storage."""
        return {
            "text": self.text,
            "type": self.entity_type.value,
            "start": self.start_char,
            "end": self.end_char,
            "confidence": self.confidence,
            "normalized": self.normalized_text,
            "umls_cui": self.umls_cui,
            "synonyms": self.synonyms,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicalEntity":
        """Create entity from dictionary."""
        return cls(
            text=data["text"],
            entity_type=MedicalEntityType(data["type"]),
            start_char=data["start"],
            end_char=data["end"],
            confidence=data["confidence"],
            normalized_text=data.get("normalized"),
            umls_cui=data.get("umls_cui"),
            synonyms=data.get("synonyms", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NERResult:
    """Result of NER processing on a text."""
    text: str
    entities: List[MedicalEntity]
    processing_time_ms: float
    model_name: str
    model_version: Optional[str] = None
    
    def get_entities_by_type(self, entity_type: MedicalEntityType) -> List[MedicalEntity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]
    
    def get_unique_entities(self) -> List[MedicalEntity]:
        """Get unique entities (by normalized text or original text)."""
        seen = set()
        unique = []
        
        for entity in self.entities:
            key = entity.normalized_text or entity.text.lower()
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for storage."""
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


@dataclass
class NERConfig:
    """Configuration for NER models."""
    model_name: str
    model_path: str
    confidence_threshold: float = 0.7
    max_length: int = 512
    batch_size: int = 32
    device: str = "cpu"  # "cpu" or "cuda"
    cache_dir: Optional[str] = None
    include_positions: bool = True
    aggregate_subwords: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
