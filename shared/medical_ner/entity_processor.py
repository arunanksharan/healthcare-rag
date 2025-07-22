"""
Entity post-processing and enrichment utilities.
"""
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict
import re

from .base import MedicalEntity, MedicalEntityType, NERResult

logger = logging.getLogger(__name__)


class MedicalEntityProcessor:
    """
    Post-processes and enriches medical entities extracted by NER.
    
    Features:
    - Entity merging (handling split entities)
    - Normalization
    - Abbreviation expansion
    - Confidence adjustment
    - Relationship detection
    """
    
    def __init__(self):
        # Common medical abbreviations for expansion
        self.abbreviations = {
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "mi": "myocardial infarction",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "cad": "coronary artery disease",
            "ckd": "chronic kidney disease",
            "gerd": "gastroesophageal reflux disease",
            "bid": "twice daily",
            "tid": "three times daily",
            "qid": "four times daily",
            "prn": "as needed",
            "po": "by mouth",
            "iv": "intravenous",
        }
        
        # Normalization patterns
        self.normalization_rules = {
            MedicalEntityType.DOSAGE: [
                (r"(\d+)\s*mg", r"\1mg"),
                (r"(\d+)\s*ml", r"\1ml"),
                (r"(\d+)\s*mcg", r"\1mcg"),
            ],
            MedicalEntityType.FREQUENCY: [
                (r"twice\s+(?:a\s+)?day", "twice daily"),
                (r"three\s+times\s+(?:a\s+)?day", "three times daily"),
                (r"every\s+(\d+)\s+hours?", r"q\1h"),
            ],
        }
    
    def process_entities(self, ner_result: NERResult) -> NERResult:
        """
        Process and enrich entities from NER result.
        
        Args:
            ner_result: Raw NER result
            
        Returns:
            Processed NER result with enriched entities
        """
        # Copy entities to avoid modifying original
        entities = [e for e in ner_result.entities]
        
        # Step 1: Merge adjacent entities
        entities = self._merge_adjacent_entities(entities, ner_result.text)
        
        # Step 2: Expand abbreviations
        entities = self._expand_abbreviations(entities)
        
        # Step 3: Normalize entities
        entities = self._normalize_entities(entities)
        
        # Step 4: Detect relationships
        entities = self._detect_relationships(entities, ner_result.text)
        
        # Step 5: Adjust confidence based on context
        entities = self._adjust_confidence(entities, ner_result.text)
        
        # Create new result with processed entities
        return NERResult(
            text=ner_result.text,
            entities=entities,
            processing_time_ms=ner_result.processing_time_ms,
            model_name=ner_result.model_name,
            model_version=ner_result.model_version,
        )
    
    def _merge_adjacent_entities(
        self, 
        entities: List[MedicalEntity], 
        text: str
    ) -> List[MedicalEntity]:
        """Merge entities that should be together."""
        if not entities:
            return entities
        
        merged = []
        i = 0
        
        while i < len(entities):
            current = entities[i]
            
            # Check if next entity should be merged
            if i + 1 < len(entities):
                next_entity = entities[i + 1]
                
                # Merge if they're close and related
                if self._should_merge(current, next_entity, text):
                    # Create merged entity
                    merged_text = text[current.start_char:next_entity.end_char]
                    merged_entity = MedicalEntity(
                        text=merged_text,
                        entity_type=self._determine_merged_type(current, next_entity),
                        start_char=current.start_char,
                        end_char=next_entity.end_char,
                        confidence=min(current.confidence, next_entity.confidence),
                        metadata={
                            "merged_from": [current.text, next_entity.text],
                            "original_types": [current.entity_type.value, next_entity.entity_type.value],
                        }
                    )
                    merged.append(merged_entity)
                    i += 2  # Skip next entity
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _should_merge(
        self, 
        entity1: MedicalEntity, 
        entity2: MedicalEntity,
        text: str
    ) -> bool:
        """Determine if two entities should be merged."""
        # Check distance between entities
        gap_text = text[entity1.end_char:entity2.start_char]
        
        # Merge if gap is small and contains only spaces or connectors
        if len(gap_text) <= 3 and gap_text.strip() in ["", "-", "/"]:
            # Merge dosage-related entities
            if (entity1.entity_type in [MedicalEntityType.DOSAGE, MedicalEntityType.STRENGTH] and
                entity2.entity_type in [MedicalEntityType.DOSAGE, MedicalEntityType.STRENGTH]):
                return True
            
            # Merge drug name parts
            if (entity1.entity_type == MedicalEntityType.DRUG and 
                entity2.entity_type == MedicalEntityType.DRUG):
                return True
        
        return False
    
    def _determine_merged_type(
        self, 
        entity1: MedicalEntity, 
        entity2: MedicalEntity
    ) -> MedicalEntityType:
        """Determine the type for merged entity."""
        # If same type, keep it
        if entity1.entity_type == entity2.entity_type:
            return entity1.entity_type
        
        # Dosage combinations
        if (MedicalEntityType.DOSAGE in [entity1.entity_type, entity2.entity_type] or
            MedicalEntityType.STRENGTH in [entity1.entity_type, entity2.entity_type]):
            return MedicalEntityType.DOSAGE
        
        # Default to first entity type
        return entity1.entity_type
    
    def _expand_abbreviations(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Expand known medical abbreviations."""
        expanded = []
        
        for entity in entities:
            entity_lower = entity.text.lower()
            
            if entity_lower in self.abbreviations:
                # Create expanded version
                entity.normalized_text = self.abbreviations[entity_lower]
                entity.metadata["abbreviation_expanded"] = True
                entity.synonyms.append(entity.text)  # Keep original as synonym
            
            expanded.append(entity)
        
        return expanded
    
    def _normalize_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Normalize entity text based on type."""
        normalized = []
        
        for entity in entities:
            # Apply normalization rules for entity type
            if entity.entity_type in self.normalization_rules:
                normalized_text = entity.normalized_text or entity.text
                
                for pattern, replacement in self.normalization_rules[entity.entity_type]:
                    normalized_text = re.sub(
                        pattern, 
                        replacement, 
                        normalized_text, 
                        flags=re.IGNORECASE
                    )
                
                if normalized_text != entity.text:
                    entity.normalized_text = normalized_text
            
            normalized.append(entity)
        
        return normalized
    
    def _detect_relationships(
        self, 
        entities: List[MedicalEntity], 
        text: str
    ) -> List[MedicalEntity]:
        """Detect relationships between entities."""
        # Sort entities by position
        entities.sort(key=lambda x: x.start_char)
        
        # Look for drug-dosage relationships
        for i, entity in enumerate(entities):
            if entity.entity_type == MedicalEntityType.DRUG:
                # Look for nearby dosage
                for j in range(i + 1, min(i + 4, len(entities))):
                    nearby = entities[j]
                    
                    if nearby.entity_type in [MedicalEntityType.DOSAGE, MedicalEntityType.STRENGTH]:
                        # Check if they're related (within reasonable distance)
                        distance = nearby.start_char - entity.end_char
                        if distance < 50:  # Arbitrary threshold
                            entity.metadata["related_dosage"] = nearby.text
                            nearby.metadata["related_drug"] = entity.text
                            break
        
        return entities
    
    def _adjust_confidence(
        self, 
        entities: List[MedicalEntity], 
        text: str
    ) -> List[MedicalEntity]:
        """Adjust entity confidence based on context."""
        adjusted = []
        
        for entity in entities:
            # Boost confidence for entities with clear context
            context_start = max(0, entity.start_char - 50)
            context_end = min(len(text), entity.end_char + 50)
            context = text[context_start:context_end].lower()
            
            # Boost drug confidence if prescription context
            if entity.entity_type == MedicalEntityType.DRUG:
                if any(word in context for word in ["prescribed", "medication", "take", "dose"]):
                    entity.confidence = min(1.0, entity.confidence * 1.1)
            
            # Boost disease confidence if diagnosis context
            elif entity.entity_type == MedicalEntityType.DISEASE:
                if any(word in context for word in ["diagnosed", "diagnosis", "history of"]):
                    entity.confidence = min(1.0, entity.confidence * 1.1)
            
            adjusted.append(entity)
        
        return adjusted
    
    def extract_entity_summary(self, entities: List[MedicalEntity]) -> Dict[str, List[str]]:
        """
        Extract a summary of entities by type.
        
        Returns:
            Dictionary mapping entity type to list of unique entity texts
        """
        summary = defaultdict(list)
        seen = defaultdict(set)
        
        for entity in entities:
            # Use normalized text if available
            text = entity.normalized_text or entity.text
            text_lower = text.lower()
            
            # Avoid duplicates
            if text_lower not in seen[entity.entity_type]:
                seen[entity.entity_type].add(text_lower)
                summary[entity.entity_type.value].append(text)
        
        return dict(summary)
