import logging
from datetime import date
from typing import Dict, List, Optional
from openai import OpenAI

from ..core.settings import settings
from shared.query_analysis import EnhancedQueryProcessor, QueryIntent

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)
enhanced_processor = EnhancedQueryProcessor()


def enhance_query(query: str, use_llm_enhancement: bool = True) -> str:
    """
    Enhance the user's query using medical analysis and optionally LLM.
    
    Args:
        query: Original user query
        use_llm_enhancement: Whether to use LLM for additional enhancement
        
    Returns:
        Enhanced query string
    """
    # First, use our medical query processor
    try:
        enhanced_result = enhanced_processor.process_query(query)
        
        # Log intent detection
        logger.info(
            f"Query intent: {enhanced_result.primary_intent.value} "
            f"(confidence: {enhanced_result.intent_confidence:.2f})"
        )
        
        # Log medical entities found
        if enhanced_result.analysis.entities:
            entity_info = ", ".join([
                f"{e.text} ({e.entity_type.value})"
                for e in enhanced_result.analysis.entities
            ])
            logger.info(f"Medical entities found: {entity_info}")
        
        # Use the best enhanced query
        if enhanced_result.enhanced_queries:
            base_enhanced_query = enhanced_result.enhanced_queries[0]
        else:
            base_enhanced_query = enhanced_result.analysis.cleaned_query
        
        # If high confidence in our analysis, skip LLM
        if enhanced_result.intent_confidence > 0.8 and not use_llm_enhancement:
            logger.info(f"High confidence medical query, using: '{base_enhanced_query}'")
            return base_enhanced_query
        
        # For complex or low-confidence queries, use LLM for additional enhancement
        if use_llm_enhancement:
            return enhance_with_llm(
                base_enhanced_query,
                enhanced_result.primary_intent,
                enhanced_result.analysis.entities
            )
        else:
            return base_enhanced_query
            
    except Exception as e:
        logger.error(f"Error in medical query enhancement: {e}", exc_info=True)
        # Fall back to LLM enhancement
        if use_llm_enhancement:
            return enhance_with_llm(query, None, [])
        else:
            return query


def enhance_with_llm(
    query: str,
    intent: Optional[QueryIntent] = None,
    entities: Optional[List] = None
) -> str:
    """
    Use LLM to further enhance the query with context about intent and entities.
    """
    system_prompt = (
        "You are an AI assistant specialized in understanding and processing queries related to "
        "medical and healthcare documents, clinical guidelines, drug information, and medical research. "
        "Your task is to refine user queries to be highly effective for retrieving these specific types of documents."
    )
    
    # Add intent context if available
    intent_context = ""
    if intent:
        intent_context = f"\nThe query appears to be asking about: {intent.value}"
        if intent == QueryIntent.DOSAGE:
            intent_context += " - Focus on specific dosing information, administration routes, and schedules."
        elif intent == QueryIntent.CONTRAINDICATIONS:
            intent_context += " - Focus on warnings, drug interactions, and when not to use."
        elif intent == QueryIntent.SIDE_EFFECTS:
            intent_context += " - Focus on adverse reactions, complications, and safety information."
    
    # Add entity context if available
    entity_context = ""
    if entities:
        entity_types = set(e.entity_type.value for e in entities)
        entity_context = f"\nMedical entities identified: {', '.join(entity_types)}"
    
    today_date = str(date.today())
    user_prompt = f"""Please analyze the following medical query, considering today's date is {today_date}.{intent_context}{entity_context}
Your goal is to transform this query into an optimized search query suitable for retrieving relevant medical documents, clinical guidelines, drug information, or research papers.

Follow these steps carefully:
1.  **Identify Key Medical Entities**: Extract diseases, drugs, procedures, symptoms, or anatomical terms.
2.  **Determine Clinical Context**: Infer if the user needs diagnostic, therapeutic, pharmaceutical, or procedural information.
3.  **Use Medical Terminology**: Replace lay terms with professional medical terminology when appropriate.
4.  **Consider Current Guidelines**: For treatment or diagnostic queries, consider that guidelines may be updated annually.
5.  **Add Specificity**: Include relevant medical context (e.g., "hypertension" → "hypertension management guidelines").
6.  **Focus on Evidence**: For medical queries, prioritize evidence-based information and clinical guidelines.
7.  **Preserve Critical Details**: Keep specific drug names, dosages, or medical conditions exactly as mentioned.
8.  **Conciseness**: Create a focused search query that captures the medical information need.

User Query: "{query}"

Based on your analysis, provide ONLY the enhanced query string. Do not include any explanations, apologies, or introductory phrases. Just the query itself.

Enhanced Query:"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        enhanced = response.choices[0].message.content.strip()
        if enhanced.lower().startswith("enhanced query:"):
            enhanced = enhanced[len("enhanced query:"):].strip()
        if enhanced.startswith('"') and enhanced.endswith('"'):
            enhanced = enhanced[1:-1] # Remove surrounding quotes if present
        
        logger.info(f"Original query: '{query}', Enhanced query: '{enhanced}'")
        return enhanced
    except Exception as e:
        logger.error(f"Query enhancement failed for query '{query}', returning original query. Error: {e}", exc_info=True)
        return query
