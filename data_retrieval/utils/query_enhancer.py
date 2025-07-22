import logging
from datetime import date
from openai import OpenAI

from ..core.settings import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)


def enhance_query(query: str) -> str:
    """
    Rewrite the user's raw query into a concise, information-rich search query
    optimized for government documents, policies, FAQs, and guidelines.
    Returns ONLY the enhanced query. Falls back to the original query on any error.
    """
    system_prompt = (
        "You are an AI assistant specialized in understanding and processing queries related to "
        "government documents, public policies, official guidelines, and frequently asked questions (FAQs). "
        "Your task is to refine user queries to be highly effective for retrieving these specific types of documents."
    )
    today_date = str(date.today())
    user_prompt = f"""Please analyze the following user query, considering today's date is {today_date}.
Your goal is to transform this query into an optimized search query suitable for retrieving relevant government documents, policies, FAQs, or guidelines.

Follow these steps carefully:
1.  **Identify Key Entities and Concepts**: Extract the core subjects, organizations, programs, or topics mentioned in the query.
2.  **Determine Document Intent**: Infer if the user is looking for a specific policy, a set of guidelines, answers to common questions (FAQ), or general information that might be contained in official documents.
3.  **Incorporate Official Terminology**: If possible, replace colloquial terms with official or formal terminology commonly used in government and public sector documents (e.g., 'rules for new businesses' might become 'guidelines for new business registration' or 'policy on small business permits').
4.  **Consider Temporal Aspects**: If the query implies a time frame (e.g., 'latest update', 'rules from last year', 'current policy'), incorporate this into the enhanced query. Use today's date ({today_date}) as a reference for terms like 'current' or 'recent'.
5.  **Add Specificity**: If the query is broad, try to add specificity based on common structures of official documents. For example, if the query is 'environmental regulations', a possible enhancement could be 'environmental protection regulations and compliance guidelines'.
6.  **Focus on Keywords**: The enhanced query should be a string of keywords and phrases that are likely to appear in the target documents. Avoid natural language questions or conversational phrases.
7.  **Preserve Core Intent**: Ensure the enhanced query accurately reflects the user's original information need. Do not introduce new topics or significantly alter the scope.
8.  **Conciseness**: Keep the enhanced query concise and to the point, while maximizing its information content for retrieval.

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
