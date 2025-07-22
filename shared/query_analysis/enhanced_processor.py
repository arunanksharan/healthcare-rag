"""
Enhanced query processor that combines intent detection and medical analysis.
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .intent_detection import MedicalIntentDetector, QueryIntent, IntentDetectionResult
from .medical_analyzer import MedicalQueryAnalyzer, QueryAnalysisResult, MedicalEntity

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQueryResult:
    """Complete result of enhanced query processing."""
    original_query: str
    primary_intent: QueryIntent
    intent_confidence: float
    analysis: QueryAnalysisResult
    enhanced_queries: List[str]
    retrieval_strategy: Dict[str, any]
    metadata_filters: Dict[str, any]


class EnhancedQueryProcessor:
    """
    Combines intent detection and medical analysis for comprehensive query processing.
    """
    
    def __init__(self):
        self.intent_detector = MedicalIntentDetector()
        self.medical_analyzer = MedicalQueryAnalyzer()
        
    def process_query(self, query: str) -> EnhancedQueryResult:
        """
        Process a medical query through the full enhancement pipeline.
        
        Args:
            query: User's original query
            
        Returns:
            EnhancedQueryResult with all processing results
        """
        # Step 1: Detect intent
        intent_result = self.intent_detector.detect_intent(query)
        
        # Step 2: Analyze medical content
        analysis_result = self.medical_analyzer.analyze_query(query)
        
        # Step 3: Generate enhanced queries based on intent and analysis
        enhanced_queries = self._generate_enhanced_queries(
            intent_result,
            analysis_result
        )
        
        # Step 4: Get retrieval strategy
        retrieval_strategy = self.intent_detector.get_retrieval_strategy(
            intent_result.primary_intent
        )
        
        # Step 5: Generate metadata filters
        metadata_filters = self._generate_metadata_filters(
            intent_result,
            analysis_result
        )
        
        return EnhancedQueryResult(
            original_query=query,
            primary_intent=intent_result.primary_intent,
            intent_confidence=intent_result.confidence,
            analysis=analysis_result,
            enhanced_queries=enhanced_queries,
            retrieval_strategy=retrieval_strategy,
            metadata_filters=metadata_filters
        )
    
    def _generate_enhanced_queries(
        self,
        intent_result: IntentDetectionResult,
        analysis_result: QueryAnalysisResult
    ) -> List[str]:
        """
        Generate enhanced queries based on intent and analysis.
        """
        enhanced_queries = []
        
        # Start with the cleaned, corrected query
        base_query = analysis_result.cleaned_query
        enhanced_queries.append(base_query)
        
        # Add query variants
        enhanced_queries.extend(analysis_result.query_variants)
        
        # Add intent-specific enhancements
        intent_enhancements = self._get_intent_specific_enhancements(
            intent_result.primary_intent,
            base_query,
            analysis_result.entities
        )
        enhanced_queries.extend(intent_enhancements)
        
        # Add entity-focused queries
        for entity in analysis_result.entities:
            entity_queries = self._generate_entity_queries(
                entity,
                intent_result.primary_intent
            )
            enhanced_queries.extend(entity_queries)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in enhanced_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:10]  # Limit to top 10 queries
    
    def _get_intent_specific_enhancements(
        self,
        intent: QueryIntent,
        base_query: str,
        entities: List[MedicalEntity]
    ) -> List[str]:
        """
        Add intent-specific query enhancements.
        """
        enhancements = []
        
        if intent == QueryIntent.DEFINITION:
            enhancements.extend([
                f"what is {base_query}",
                f"{base_query} definition medical",
                f"{base_query} overview",
                f"{base_query} explanation"
            ])
        
        elif intent == QueryIntent.DIAGNOSIS:
            enhancements.extend([
                f"{base_query} diagnostic criteria",
                f"{base_query} signs symptoms",
                f"how to diagnose {base_query}",
                f"{base_query} clinical features"
            ])
        
        elif intent == QueryIntent.TREATMENT:
            enhancements.extend([
                f"{base_query} treatment guidelines",
                f"{base_query} management protocol",
                f"{base_query} therapy options",
                f"treating {base_query}"
            ])
        
        elif intent == QueryIntent.DOSAGE:
            # For dosage, we need specific drug names
            drug_entities = [e for e in entities if e.entity_type.value == "drug"]
            for drug in drug_entities:
                enhancements.extend([
                    f"{drug.text} dosage administration",
                    f"{drug.text} dose mg",
                    f"{drug.text} dosing schedule",
                    f"{drug.text} recommended dose"
                ])
        
        elif intent == QueryIntent.SIDE_EFFECTS:
            enhancements.extend([
                f"{base_query} adverse reactions",
                f"{base_query} side effects complications",
                f"{base_query} safety warnings",
                f"{base_query} toxicity"
            ])
        
        elif intent == QueryIntent.CONTRAINDICATIONS:
            enhancements.extend([
                f"{base_query} contraindications warnings",
                f"{base_query} drug interactions",
                f"{base_query} when not to use",
                f"{base_query} precautions"
            ])
        
        elif intent == QueryIntent.PROCEDURE:
            enhancements.extend([
                f"{base_query} procedure steps",
                f"{base_query} surgical technique",
                f"how to perform {base_query}",
                f"{base_query} protocol guidelines"
            ])
        
        return enhancements
    
    def _generate_entity_queries(
        self,
        entity: MedicalEntity,
        intent: QueryIntent
    ) -> List[str]:
        """
        Generate queries focused on specific entities.
        """
        queries = []
        
        # Add entity with its type
        queries.append(f"{entity.text} {entity.entity_type.value}")
        
        # Add entity with intent context
        if intent == QueryIntent.TREATMENT and entity.entity_type.value == "disease":
            queries.append(f"treatment for {entity.text}")
            queries.append(f"{entity.text} management")
        
        elif intent == QueryIntent.DOSAGE and entity.entity_type.value == "drug":
            queries.append(f"{entity.text} dosing")
            queries.append(f"{entity.text} administration")
        
        # Add synonyms
        for synonym in entity.synonyms[:2]:  # Limit synonyms
            queries.append(synonym)
        
        return queries
    
    def _generate_metadata_filters(
        self,
        intent_result: IntentDetectionResult,
        analysis_result: QueryAnalysisResult
    ) -> Dict[str, any]:
        """
        Generate metadata filters for retrieval based on intent and entities.
        """
        filters = {}
        
        # Add chunk type filters based on intent
        strategy = self.intent_detector.get_retrieval_strategy(intent_result.primary_intent)
        if strategy.get("chunk_types"):
            filters["chunk_types"] = strategy["chunk_types"]
        
        # Add section filters
        if strategy.get("boost_sections"):
            filters["boost_sections"] = strategy["boost_sections"]
        
        # Add entity-based filters
        if analysis_result.entities:
            filters["medical_entities"] = [
                {
                    "text": entity.text,
                    "type": entity.entity_type.value,
                    "normalized": entity.normalized_form
                }
                for entity in analysis_result.entities
            ]
        
        # Add confidence threshold
        filters["min_confidence"] = strategy.get("min_confidence", 0.5)
        
        # Add source requirements
        if intent_result.requires_specific_source:
            filters["require_authoritative_source"] = True
        
        return filters
