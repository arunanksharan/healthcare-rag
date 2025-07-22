"""
Query enhancement module that integrates medical analysis with retrieval optimization.
"""
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .medical_analyzer import MedicalQueryAnalyzer, QueryAnalysisResult
from .intent_detection import QueryIntent, detect_medical_intent

# Import NER components for query analysis
from shared.medical_ner import (
    get_medical_ner_model,
    MedicalEntityProcessor,
    MedicalEntityType as NEREntityType,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQuery:
    """Represents an enhanced query with all processing applied."""
    original_query: str
    cleaned_query: str
    query_variants: List[str]
    entities: Dict[str, List[str]]  # entity_type -> list of entities
    intent: QueryIntent
    intent_confidence: float
    filters: Dict[str, Any]
    boost_params: Dict[str, Any]
    
    def get_all_query_texts(self) -> List[str]:
        """Get all query variations for embedding."""
        all_queries = [self.original_query, self.cleaned_query]
        all_queries.extend(self.query_variants)
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for q in all_queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique


class QueryEnhancer:
    """
    Enhances medical queries for improved retrieval.
    Integrates NER, intent detection, and query expansion.
    """
    
    def __init__(self):
        self.medical_analyzer = MedicalQueryAnalyzer()
        
        # Initialize NER components for query analysis
        self._ner_model = None
        self._entity_processor = None
        self._ner_initialized = False
        self._init_ner_for_queries()
        
        # Intent to search configuration mapping
        self.intent_configs = {
            QueryIntent.DOSAGE: {
                "answer_types": ["dosage", "definition"],  # Match chunker output
                "boost_sections": ["dosage", "dosage_administration", "administration"],
                "entity_types": ["drug", "dosage"],
                "boost_weight": 1.3
            },
            QueryIntent.SIDE_EFFECTS: {
                "answer_types": ["side_effects"],  # Match chunker output
                "boost_sections": ["adverse_reactions", "side_effects", "warnings"],
                "entity_types": ["drug", "symptom"],
                "boost_weight": 1.3
            },
            QueryIntent.CONTRAINDICATIONS: {
                "answer_types": ["contraindications"],  # Match chunker output
                "boost_sections": ["contraindications", "warnings", "precautions"],
                "entity_types": ["drug", "disease"],
                "boost_weight": 1.3
            },
            QueryIntent.DIAGNOSIS: {
                "answer_types": ["diagnosis", "definition"],  # Match chunker output
                "boost_sections": ["diagnosis", "clinical_features", "symptoms"],
                "entity_types": ["disease", "symptom"],
                "boost_weight": 1.2
            },
            QueryIntent.TREATMENT: {
                "answer_types": ["treatment"],  # Match chunker output
                "boost_sections": ["treatment", "management", "therapy", "guidelines"],
                "entity_types": ["disease", "drug", "procedure"],
                "boost_weight": 1.2
            },
            QueryIntent.COMPARISON: {
                "answer_types": ["comparison"],  # Match chunker output
                "boost_sections": ["comparison", "differences"],
                "entity_types": ["drug", "disease"],
                "boost_weight": 1.2
            },
            QueryIntent.PROCEDURE: {
                "answer_types": ["procedure"],  # Match chunker output
                "boost_sections": ["procedure", "technique", "method"],
                "entity_types": ["procedure"],
                "boost_weight": 1.2
            },
            QueryIntent.GENERAL: {
                "answer_types": ["general"],
                "boost_sections": [],
                "entity_types": [],
                "boost_weight": 1.0
            }
        }
    
    def _init_ner_for_queries(self):
        """Initialize NER model for query analysis."""
        try:
            logger.info("Initializing BioBERT NER for query analysis...")
            self._ner_model = get_medical_ner_model("biomedical-ner-all")
            self._entity_processor = MedicalEntityProcessor()
            self._ner_initialized = True
            logger.info("BioBERT NER initialized successfully for query analysis")
        except Exception as e:
            logger.warning(f"Could not initialize NER for queries: {e}")
            self._ner_initialized = False
    
    def _extract_entities_with_ner(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using BioBERT NER."""
        if not self._ner_initialized or not self._ner_model:
            return {}
        
        try:
            # Extract entities using NER
            ner_result = self._ner_model.extract_entities(text)
            
            # Process entities
            if self._entity_processor:
                ner_result = self._entity_processor.process_entities(ner_result)
            
            # Group by type
            entities_by_type = {}
            for entity in ner_result.entities:
                entity_type = entity.entity_type.value
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                
                # Add both original and normalized forms (all lowercase for matching)
                entities_by_type[entity_type].append(entity.text.lower())
                if entity.normalized_form:
                    entities_by_type[entity_type].append(entity.normalized_form.lower())
                
                # Add synonyms
                for synonym in entity.synonyms:
                    entities_by_type[entity_type].append(synonym.lower())
            
            # Remove duplicates
            for entity_type in entities_by_type:
                entities_by_type[entity_type] = list(set(entities_by_type[entity_type]))
            
            return entities_by_type
            
        except Exception as e:
            logger.error(f"NER extraction failed for query: {e}")
            return {}
    
    def enhance_query(self, query: str) -> EnhancedQuery:
        """
        Enhance a medical query for optimal retrieval.
        
        Args:
            query: The user's query
            
        Returns:
            EnhancedQuery with all enhancements applied
        """
        # Step 1: Analyze query (NER, abbreviations, corrections)
        analysis = self.medical_analyzer.analyze_query(query)
        
        # Step 1b: Extract entities using BioBERT NER if available
        ner_entities = self._extract_entities_with_ner(query)
        
        # Merge NER entities with pattern-based entities
        if ner_entities:
            logger.info(f"NER extracted entities: {ner_entities}")
            for entity_type, entities in ner_entities.items():
                if entity_type in ["drug", "disease", "procedure", "symptom"]:
                    # Update analysis with NER entities
                    existing = {e.text.lower() for e in analysis.entities if e.entity_type.value == entity_type}
                    for ner_entity in entities:
                        if ner_entity not in existing:
                            # Add as MedicalEntity for compatibility
                            from .medical_analyzer import MedicalEntity, MedicalEntityType
                            try:
                                entity_type_enum = MedicalEntityType(entity_type)
                                analysis.entities.append(MedicalEntity(
                                    text=ner_entity,
                                    entity_type=entity_type_enum,
                                    normalized_form=ner_entity,
                                    confidence=0.9,
                                    synonyms=[]
                                ))
                            except ValueError:
                                # Skip if entity type not in enum
                                logger.debug(f"Skipping entity type {entity_type} not in MedicalEntityType")
        
        # Step 2: Detect intent
        intent, confidence = detect_medical_intent(query, analysis)
        
        # Step 3: Get intent-specific configuration
        intent_config = self.intent_configs.get(intent, self.intent_configs[QueryIntent.GENERAL])
        
        # Step 4: Build entity filters
        entity_filters = self._build_entity_filters(analysis, intent_config)
        
        # Step 5: Build boost parameters
        boost_params = self._build_boost_params(intent_config)
        
        # Step 6: Create query variants
        query_variants = self._create_query_variants(analysis, intent)
        
        return EnhancedQuery(
            original_query=query,
            cleaned_query=analysis.cleaned_query,
            query_variants=query_variants,
            entities=self._group_entities_by_type(analysis),
            intent=intent,
            intent_confidence=confidence,
            filters=entity_filters,
            boost_params=boost_params
        )
    
    def _build_entity_filters(
        self, 
        analysis: QueryAnalysisResult, 
        intent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build filters based on extracted entities and intent."""
        filters = {}
        
        # Extract entities by type
        entities_by_type = self._group_entities_by_type(analysis)
        
        # Add entity filters based on intent
        relevant_entity_types = intent_config.get("entity_types", [])
        
        for entity_type in relevant_entity_types:
            # Only add filters for entity types we actually extract and store
            if entity_type not in ["drug", "disease", "procedure"]:
                continue
                
            if entity_type in entities_by_type:
                # Use normalized forms if available
                entities = entities_by_type[entity_type]
                if entities:
                    # Add both original and normalized forms
                    all_forms = set()
                    for entity in analysis.entities:
                        if entity.entity_type.value == entity_type:
                            all_forms.add(entity.text.lower())
                            if entity.normalized_form:
                                all_forms.add(entity.normalized_form.lower())
                            all_forms.update(entity.synonyms)
                    
                    if all_forms:
                        filters[f"{entity_type}s"] = list(all_forms)  # pluralize for field name
        
        # Add intent-based filters
        if intent_config.get("answer_types"):
            filters["answer_types"] = intent_config["answer_types"]
        
        return filters
    
    def _build_boost_params(self, intent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build boosting parameters based on intent."""
        boost_params = {
            "boost_sections": intent_config.get("boost_sections", []),
            "boost_weight": intent_config.get("boost_weight", 1.0),
            "answer_types": intent_config.get("answer_types", [])
        }
        
        # Don't add chunk_types - it's a different concept from answer_types
        # chunk_type is the structural type (text, table, etc.)
        # answer_types is what questions it can answer
        
        return boost_params
    
    def _create_query_variants(
        self, 
        analysis: QueryAnalysisResult, 
        intent: QueryIntent
    ) -> List[str]:
        """Create query variants for better retrieval."""
        variants = []
        
        # Add the base variants from analysis
        variants.extend(analysis.query_variants)
        
        # Add intent-specific variants
        if intent == QueryIntent.DOSAGE:
            # Add dosage-specific terms
            for entity in analysis.entities:
                if entity.entity_type.value == "drug":
                    variants.append(f"{entity.text} dosage")
                    variants.append(f"{entity.text} dose")
                    variants.append(f"how much {entity.text}")
        
        elif intent == QueryIntent.SIDE_EFFECTS:
            # Add side effect queries
            for entity in analysis.entities:
                if entity.entity_type.value == "drug":
                    variants.append(f"{entity.text} side effects")
                    variants.append(f"{entity.text} adverse reactions")
                    variants.append(f"{entity.text} adverse effects")
        
        elif intent == QueryIntent.CONTRAINDICATIONS:
            # Add contraindication queries
            for entity in analysis.entities:
                if entity.entity_type.value == "drug":
                    variants.append(f"{entity.text} contraindications")
                    variants.append(f"when not to use {entity.text}")
                    variants.append(f"{entity.text} warnings")
        
        # Remove duplicates
        return list(set(variants))
    
    def _group_entities_by_type(self, analysis: QueryAnalysisResult) -> Dict[str, List[str]]:
        """Group entities by their type."""
        grouped = {}
        
        for entity in analysis.entities:
            entity_type = entity.entity_type.value
            if entity_type not in grouped:
                grouped[entity_type] = []
            
            # Add normalized form if available, otherwise original
            text = entity.normalized_form or entity.text
            if text not in grouped[entity_type]:
                grouped[entity_type].append(text)
        
        return grouped
    
    def get_search_strategy(self, enhanced_query: EnhancedQuery) -> Dict[str, Any]:
        """
        Get the complete search strategy for a query.
        
        Returns:
            Dictionary with search configuration
        """
        return {
            "query_texts": enhanced_query.get_all_query_texts(),
            "filters": enhanced_query.filters,
            "boost_params": enhanced_query.boost_params,
            "intent": enhanced_query.intent.value,
            "intent_confidence": enhanced_query.intent_confidence,
            "entities": enhanced_query.entities,
            "use_entity_filtering": len(enhanced_query.entities) > 0,
            "use_hybrid_search": True  # Always use hybrid approach
        }
