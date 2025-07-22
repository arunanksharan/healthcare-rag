"""
Medical query intent detection for healthcare RAG system.
"""
import re
from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Types of medical query intents."""
    DEFINITION = "definition"          # What is X?
    DIAGNOSIS = "diagnosis"            # How to diagnose X? Symptoms of X?
    TREATMENT = "treatment"            # How to treat X? Treatment for X?
    DOSAGE = "dosage"                 # Dosage of X? How much X?
    SIDE_EFFECTS = "side_effects"      # Side effects of X? Adverse reactions?
    CONTRAINDICATIONS = "contraindications"  # When not to use X? X interactions?
    COMPARISON = "comparison"          # X vs Y? Difference between X and Y?
    PROCEDURE = "procedure"            # How to perform X? X procedure?
    PREVENTION = "prevention"          # How to prevent X?
    GENERAL = "general"               # General query without specific intent
    
    @classmethod
    def get_default(cls) -> "QueryIntent":
        """Get default intent for ambiguous queries."""
        return cls.GENERAL


@dataclass
class IntentDetectionResult:
    """Result of intent detection analysis."""
    primary_intent: QueryIntent
    confidence: float
    secondary_intents: List[Tuple[QueryIntent, float]]
    intent_keywords: List[str]
    requires_specific_source: bool  # True for dosage, contraindications


class MedicalIntentDetector:
    """Detects intent in medical queries using pattern matching and keywords."""
    
    def __init__(self):
        # Define patterns for each intent type
        self.intent_patterns = {
            QueryIntent.DEFINITION: [
                r'\b(what|define|definition|meaning)\s+(is|are|of)\b',
                r'\b(what\s+is|what\s+are)\s+\w+',
                r'\b(explain|describe)\s+\w+',
                r'^\w+\?$',  # Single word with question mark
                r'\b(tell\s+me\s+about)\b',
            ],
            
            QueryIntent.DIAGNOSIS: [
                r'\b(diagnos[ei]|diagnostic|detect|identify|test)\b',
                r'\b(symptoms?|signs?|manifestations?|presentations?)\s+(of|for)\b',
                r'\b(how\s+to\s+diagnose|diagnosis\s+of)\b',
                r'\b(screening|evaluation)\s+(for|of)\b',
                r'\b(clinical\s+features?|diagnostic\s+criteria)\b',
            ],
            
            QueryIntent.TREATMENT: [
                r'\b(treat|treatment|therapy|management|cure)\b',
                r'\b(how\s+to\s+treat|treatment\s+for|manage)\b',
                r'\b(therapeutic|intervention|protocol)\b',
                r'\b(first[\s-]line|second[\s-]line|treatment\s+options?)\b',
                r'\b(guidelines?\s+for\s+treating)\b',
            ],
            
            QueryIntent.DOSAGE: [
                r'\b(dosage?|dose|dosing|amount|quantity)\b',
                r'\b(how\s+much|mg|mcg|ml|unit[s]?)\b',
                r'\b(daily\s+dose|maximum\s+dose|recommended\s+dose)\b',
                r'\b(administration|frequency|schedule)\b',
                r'\b(pediatric\s+dose|adult\s+dose|geriatric\s+dose)\b',
            ],
            
            QueryIntent.SIDE_EFFECTS: [
                r'\b(side[\s-]?effects?|adverse[\s-]?reactions?|complications?)\b',
                r'\b(toxicity|safety|risks?|warnings?)\b',
                r'\b(adverse\s+events?|untoward\s+effects?)\b',
                r'\b(harmful\s+effects?|negative\s+effects?)\b',
            ],
            
            QueryIntent.CONTRAINDICATIONS: [
                r'\b(contraindications?|when\s+not\s+to)\b',
                r'\b(interactions?|drug[\s-]?interactions?)\b',
                r'\b(avoid|should\s+not|must\s+not)\b',
                r'\b(incompatible|conflicts?\s+with)\b',
                r'\b(precautions?|warnings?)\s+for\b',
            ],
            
            QueryIntent.COMPARISON: [
                r'\b(\w+)\s+vs\.?\s+(\w+)\b',
                r'\b(difference|comparison|compare)\s+between\b',
                r'\b(better|worse|versus|compared\s+to)\b',
                r'\b(advantages?\s+of|disadvantages?\s+of)\b',
                r'\b(which\s+is\s+better)\b',
            ],
            
            QueryIntent.PROCEDURE: [
                r'\b(procedure|protocol|technique|method)\b',
                r'\b(how\s+to\s+perform|steps?\s+for|process\s+of)\b',
                r'\b(surgical|operative|intervention)\s+technique\b',
                r'\b(guidelines?\s+for\s+performing)\b',
            ],
            
            QueryIntent.PREVENTION: [
                r'\b(prevent|prevention|prophylaxis|avoid)\b',
                r'\b(how\s+to\s+prevent|prevention\s+of)\b',
                r'\b(risk\s+factors?|reduce\s+risk)\b',
                r'\b(preventive\s+measures?|precautions?)\b',
            ],
        }
        
        # Keywords that boost confidence for each intent
        self.intent_keywords = {
            QueryIntent.DEFINITION: ['what', 'is', 'are', 'define', 'definition', 'meaning'],
            QueryIntent.DIAGNOSIS: ['diagnose', 'diagnosis', 'symptoms', 'signs', 'test', 'screening'],
            QueryIntent.TREATMENT: ['treat', 'treatment', 'therapy', 'management', 'cure', 'protocol'],
            QueryIntent.DOSAGE: ['dose', 'dosage', 'mg', 'ml', 'mcg', 'units', 'daily', 'frequency'],
            QueryIntent.SIDE_EFFECTS: ['side effects', 'adverse', 'reactions', 'complications', 'toxicity'],
            QueryIntent.CONTRAINDICATIONS: ['contraindications', 'interactions', 'avoid', 'not use'],
            QueryIntent.COMPARISON: ['vs', 'versus', 'compare', 'difference', 'better', 'between'],
            QueryIntent.PROCEDURE: ['procedure', 'how to perform', 'technique', 'steps', 'protocol'],
            QueryIntent.PREVENTION: ['prevent', 'prevention', 'avoid', 'prophylaxis', 'reduce risk'],
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_patterns[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def detect_intent(self, query: str) -> IntentDetectionResult:
        """
        Detect the primary intent of a medical query.
        
        Args:
            query: The user's query string
            
        Returns:
            IntentDetectionResult with primary intent, confidence, and metadata
        """
        query_lower = query.lower().strip()
        
        # Score each intent
        intent_scores = {}
        matched_keywords = {}
        
        for intent, patterns in self.compiled_patterns.items():
            score = 0.0
            keywords = []
            
            # Check patterns
            for pattern in patterns:
                if pattern.search(query_lower):
                    score += 1.0
                    # Extract matched portion
                    match = pattern.search(query_lower)
                    if match:
                        keywords.append(match.group(0))
            
            # Check keywords
            for keyword in self.intent_keywords[intent]:
                if keyword.lower() in query_lower:
                    score += 0.5
                    keywords.append(keyword)
            
            intent_scores[intent] = score
            matched_keywords[intent] = list(set(keywords))  # Remove duplicates
        
        # Sort intents by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Determine primary intent
        if sorted_intents[0][1] > 0:
            primary_intent = sorted_intents[0][0]
            
            # Calculate confidence based on score difference
            if len(sorted_intents) > 1 and sorted_intents[1][1] > 0:
                # Confidence based on how much higher the primary score is
                confidence = min(0.95, sorted_intents[0][1] / (sorted_intents[0][1] + sorted_intents[1][1]))
            else:
                # High confidence if only one intent detected
                confidence = min(0.95, sorted_intents[0][1] / 3.0)  # Max 3 patterns
        else:
            # No specific intent detected
            primary_intent = QueryIntent.GENERAL
            confidence = 0.5
        
        # Get secondary intents
        secondary_intents = []
        for intent, score in sorted_intents[1:4]:  # Top 3 secondary
            if score > 0:
                relative_confidence = score / max(sorted_intents[0][1], 1.0)
                secondary_intents.append((intent, relative_confidence))
        
        # Determine if specific source required
        requires_specific_source = primary_intent in [
            QueryIntent.DOSAGE,
            QueryIntent.CONTRAINDICATIONS,
            QueryIntent.PROCEDURE
        ]
        
        return IntentDetectionResult(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents,
            intent_keywords=matched_keywords.get(primary_intent, []),
            requires_specific_source=requires_specific_source
        )
    
    def get_retrieval_strategy(self, intent: QueryIntent) -> Dict[str, any]:
        """
        Get retrieval strategy parameters based on intent.
        
        Returns:
            Dictionary with retrieval parameters:
            - chunk_types: Preferred chunk types
            - boost_sections: Sections to boost in scoring
            - precision_required: Whether high precision is needed
            - min_confidence: Minimum confidence threshold
        """
        strategies = {
            QueryIntent.DEFINITION: {
                "chunk_types": ["heading", "text"],
                "boost_sections": ["definition", "introduction", "overview"],
                "precision_required": False,
                "min_confidence": 0.6,
            },
            QueryIntent.DIAGNOSIS: {
                "chunk_types": ["text", "list", "table"],
                "boost_sections": ["diagnosis", "clinical_features", "symptoms", "signs"],
                "precision_required": True,
                "min_confidence": 0.7,
            },
            QueryIntent.TREATMENT: {
                "chunk_types": ["text", "list", "medication"],
                "boost_sections": ["treatment", "management", "therapy", "guidelines"],
                "precision_required": True,
                "min_confidence": 0.75,
            },
            QueryIntent.DOSAGE: {
                "chunk_types": ["medication", "table", "text"],
                "boost_sections": ["dosage", "administration", "medications"],
                "precision_required": True,
                "min_confidence": 0.8,
            },
            QueryIntent.SIDE_EFFECTS: {
                "chunk_types": ["text", "list", "table"],
                "boost_sections": ["side_effects", "adverse_reactions", "warnings"],
                "precision_required": True,
                "min_confidence": 0.75,
            },
            QueryIntent.CONTRAINDICATIONS: {
                "chunk_types": ["text", "list", "warning"],
                "boost_sections": ["contraindications", "drug_interactions", "warnings"],
                "precision_required": True,
                "min_confidence": 0.8,
            },
            QueryIntent.COMPARISON: {
                "chunk_types": ["text", "table", "list"],
                "boost_sections": ["comparison", "versus", "differences"],
                "precision_required": False,
                "min_confidence": 0.65,
            },
            QueryIntent.PROCEDURE: {
                "chunk_types": ["text", "list", "table"],
                "boost_sections": ["procedure", "protocol", "technique", "method"],
                "precision_required": True,
                "min_confidence": 0.75,
            },
            QueryIntent.PREVENTION: {
                "chunk_types": ["text", "list"],
                "boost_sections": ["prevention", "prophylaxis", "risk_factors"],
                "precision_required": False,
                "min_confidence": 0.65,
            },
            QueryIntent.GENERAL: {
                "chunk_types": ["text", "heading", "table", "list"],
                "boost_sections": [],
                "precision_required": False,
                "min_confidence": 0.5,
            },
        }
        
        return strategies.get(intent, strategies[QueryIntent.GENERAL])
