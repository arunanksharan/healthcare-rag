"""
Medical query analysis with ACTUAL NER integration.
This is a fixed version that uses the BioBERT NER model.
"""
import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

from shared.medical_ner import (
    get_medical_ner_model,
    MedicalEntityProcessor,
    MedicalEntityType as NEREntityType,
    MedicalEntity as NERMedicalEntity
)

logger = logging.getLogger(__name__)


# Use the same MedicalEntityType from the existing code for compatibility
from .medical_analyzer import (
    MedicalEntityType,
    MedicalEntity,
    QueryAnalysisResult,
    MedicalQueryAnalyzer as BaseMedicalQueryAnalyzer
)


class MedicalQueryAnalyzerWithNER(BaseMedicalQueryAnalyzer):
    """
    Enhanced medical query analyzer that uses actual BioBERT NER.
    """
    
    def __init__(self):
        super().__init__()
        self._ner_model = None
        self._entity_processor = None
        self._ner_initialized = False
    
    def _initialize_ner(self):
        """Initialize NER model lazily."""
        if not self._ner_initialized:
            try:
                logger.info("Initializing BioBERT NER for query analysis...")
                self._ner_model = get_medical_ner_model("biomedical-ner-all")
                self._entity_processor = MedicalEntityProcessor()
                self._ner_initialized = True
                logger.info("BioBERT NER initialized successfully for query analysis")
            except Exception as e:
                logger.error(f"Failed to initialize NER for query analysis: {e}")
                self._ner_initialized = False
    
    def _extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities using actual NER model."""
        entities = []
        
        # Initialize NER if needed
        self._initialize_ner()
        
        if self._ner_initialized and self._ner_model:
            try:
                # Use the actual BioBERT NER model
                ner_result = self._ner_model.extract_entities(text)
                
                # Process entities
                if self._entity_processor:
                    ner_result = self._entity_processor.process_entities(ner_result)
                
                # Convert NER entities to query analyzer format
                for ner_entity in ner_result.entities:
                    # Map NER entity types to query analyzer types
                    entity_type_map = {
                        NEREntityType.DRUG: MedicalEntityType.DRUG,
                        NEREntityType.DISEASE: MedicalEntityType.DISEASE,
                        NEREntityType.SYMPTOM: MedicalEntityType.SYMPTOM,
                        NEREntityType.PROCEDURE: MedicalEntityType.PROCEDURE,
                        NEREntityType.ANATOMY: MedicalEntityType.ANATOMY,
                        NEREntityType.LAB_TEST: MedicalEntityType.LAB_TEST,
                    }
                    
                    # Skip if entity type not in mapping
                    if ner_entity.entity_type not in entity_type_map:
                        continue
                    
                    entity = MedicalEntity(
                        text=ner_entity.text,
                        entity_type=entity_type_map[ner_entity.entity_type],
                        normalized_form=ner_entity.normalized_form or ner_entity.text,
                        confidence=ner_entity.confidence,
                        synonyms=list(ner_entity.synonyms) if ner_entity.synonyms else []
                    )
                    entities.append(entity)
                
                logger.info(f"Extracted {len(entities)} entities using BioBERT NER from query")
                
            except Exception as e:
                logger.error(f"NER extraction failed, falling back to pattern matching: {e}")
                # Fall back to parent class pattern matching
                entities = super()._extract_entities(text)
        else:
            # Fall back to pattern matching if NER not available
            logger.warning("NER not initialized, using pattern matching")
            entities = super()._extract_entities(text)
        
        return entities
