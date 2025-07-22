"""
Medical Named Entity Recognition (NER) module for healthcare document enrichment.
"""
from .base import MedicalEntity, MedicalEntityType, NERResult
from .model_registry import MedicalNERRegistry, get_medical_ner_model
from .biobert_ner import BioBERTNER
from .entity_processor import MedicalEntityProcessor

__all__ = [
    "MedicalEntity",
    "MedicalEntityType",
    "NERResult",
    "MedicalNERRegistry",
    "get_medical_ner_model",
    "BioBERTNER",
    "MedicalEntityProcessor",
]
