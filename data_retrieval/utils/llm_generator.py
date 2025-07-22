import logging
import re
from typing import Any
from openai import OpenAI

from data_retrieval.core.settings import settings
from data_retrieval.utils.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.openai_api_key)


def generate_llm_response(query: str, ranked_documents: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate an LLM response with sequential citations and return only referenced contexts.

    Args:
        query (str): The user's query.
        ranked_documents (list): A list of dictionaries, where each dictionary
                                represents a document and has 'content'
                                and 'metadata' keys.

    Returns:
        dict: A dictionary containing:
            - "llm_answer_with_sequential_citations" (str): The LLM's response with re-numbered citations like ^[1], ^[2].
            - "cited_source_documents" (list): A list of dictionaries, where each entry corresponds
                                            to a citation in the text (e.g., ^[1]) and contains
                                            the 'id' (sequential number), 'references_original_context' (e.g. "Context_5"),
                                            and 'source_data' (the actual content and metadata of "Context_5").
    """  # noqa: E501
    try:
        # 1. Prepare contexts that were initially provided to the LLM
        # This map holds all context data, keyed by "Context_N" (uppercase C)
        all_initial_contexts_map: dict[str, dict[str, Any]] = {}
        contexts_for_llm_input_string_list = []

        if not ranked_documents:
            logger.warning("No documents provided for LLM response generation.")
            return {
                "llm_answer_with_sequential_citations": "na",  # Consistent with prompt's "na"
                "cited_source_documents": [],
            }

        for i, doc in enumerate(ranked_documents):
            # Use uppercase 'C' for consistency with regex and LLM prompt examples
            original_context_key = f"Context_{i + 1}"
            text_content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            all_initial_contexts_map[original_context_key] = {
                "text": text_content,
                **metadata,
            }
            # Pass tags like <Context_1>, <Context_2> to the LLM
            contexts_for_llm_input_string_list.append(
                f"<{original_context_key}>\n{text_content}\n</{original_context_key}>"
            )

        contexts_string_for_prompt = "\n\n".join(contexts_for_llm_input_string_list)

        # 2. Prepare system and user messages for the LLM
        system_message = SYSTEM_PROMPT
        user_message = USER_PROMPT_TEMPLATE.format(contexts_string=contexts_string_for_prompt, query=query)

        # 3. Call the chat completion API
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        llm_raw_answer = response.choices[0].message.content.strip()

        # 4. Post-process LLM raw answer for sequential citations
        cited_source_documents_list: list[dict[str, Any]] = []
        sequential_citation_counter = 0

        # Define a replacer function for re.sub
        def citation_replacer(match_obj):
            nonlocal sequential_citation_counter
            nonlocal cited_source_documents_list

            sequential_citation_counter += 1
            # This will be "Context_N" (uppercase C) as captured by the regex
            original_context_key_from_match = match_obj.group(1)

            original_context_data = all_initial_contexts_map.get(original_context_key_from_match)

            if original_context_data is None:
                logger.warning(
                    f"LLM cited '{original_context_key_from_match}' which was not found in all_initial_contexts_map. This might indicate an LLM hallucination or an unexpected format. Keys in map: {list(all_initial_contexts_map.keys())}"  # noqa: E501
                )
                source_data_for_this_citation = {
                    "error": f"Original context {original_context_key_from_match} not found"
                }
            else:
                source_data_for_this_citation = original_context_data

            cited_source_documents_list.append(
                {
                    "id": sequential_citation_counter,
                    "references_original_context": original_context_key_from_match,
                    "source_data": source_data_for_this_citation,
                }
            )
            return f"^[{sequential_citation_counter}]"

        # Regex now expects and captures "Context_N" (uppercase C)
        llm_answer_with_sequential_citations = re.sub(r"\^\[(Context_\d+)\]", citation_replacer, llm_raw_answer)

        if llm_raw_answer.strip().lower() == "na" and not cited_source_documents_list:
            llm_answer_with_sequential_citations = "na"

        return {
            "llm_answer_with_sequential_citations": llm_answer_with_sequential_citations,
            "cited_source_documents": cited_source_documents_list,
        }

    except Exception as e:
        logger.error(f"Error generating LLM response: {e}", exc_info=True)
        return {
            "llm_answer_with_sequential_citations": "Error: Could not generate LLM response.",
            "cited_source_documents": [],
        }
