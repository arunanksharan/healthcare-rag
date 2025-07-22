import logging
import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from data_retrieval.app.api.models import SearchRequest
from data_retrieval.utils.llm_generator import generate_llm_response
from data_retrieval.utils.openai_embedding import (
    get_openai_embedding,
    get_embedding_for_query,
    get_embeddings_for_multiple_types,
)
from data_retrieval.utils.query_enhancer import enhance_query
from data_retrieval.utils.reranker import rerank_documents
from data_retrieval.utils.search import (
    search_with_metadata_and_embedding,
    search_multiple_collections,
    search_single_collection,
)
from shared.embeddings import EmbeddingType, get_collection_name
from shared.query_analysis import EnhancedQueryProcessor

logger = logging.getLogger(__name__)
router = APIRouter()
enhanced_processor = EnhancedQueryProcessor()


@router.post("/search", summary="Search & generate LLM response with sequential citations")
async def search(request: SearchRequest):
    try:
        query = request.query
        embedding_types = request.embedding_types
        
        # 1) Process query through enhanced analyzer
        query_analysis = await run_in_threadpool(enhanced_processor.process_query, query)
        
        # Log analysis results
        logger.info(
            f"Query intent: {query_analysis.primary_intent.value} "
            f"(confidence: {query_analysis.intent_confidence:.2f})"
        )
        
        # 2) Get enhanced query (may use LLM)
        enhanced_query = await run_in_threadpool(enhance_query, query)

        # 3) Determine which embedding types to search
        if embedding_types is None:
            # Use default or all types based on confidence
            if query_analysis.intent_confidence > 0.7:
                # High confidence - use default medical embedding
                embedding_types = [EmbeddingType.get_default()]
            else:
                # Low confidence - search multiple types
                embedding_types = [EmbeddingType.PUBMEDBERT, EmbeddingType.BIOBERT]
        
        # 4) Perform intent-aware search
        if len(embedding_types) == 1:
            # Single embedding type search
            embedding_type = embedding_types[0]
            query_vector = await run_in_threadpool(
                get_embedding_for_query,
                enhanced_query,
                embedding_type
            )
            
            # Get collection name
            from data_retrieval.core.settings import settings
            collection_name = get_collection_name(settings.qdrant_collection_name, embedding_type)
            
            # Search with intent filters
            candidate_documents = await run_in_threadpool(
                search_single_collection,
                embedding=query_vector,
                collection_name=collection_name,
                metadata=None,
                intent_filters=query_analysis.metadata_filters,
                top_k=50
            )
        else:
            # Multiple embedding type search
            query_embeddings = await run_in_threadpool(
                get_embeddings_for_multiple_types,
                enhanced_query,
                embedding_types
            )
            
            # For multiple collections, apply filters per collection
            # TODO: Enhance search_multiple_collections to support intent filters
            candidate_documents = await run_in_threadpool(
                search_multiple_collections,
                query_embeddings=query_embeddings,
                metadata=None,
            )

        llm_output_data: dict

        if not candidate_documents:
            llm_output_data = { 
                "llm_answer_with_sequential_citations": "No relevant documents found to answer the query based on initial search.",  # noqa: E501
                "cited_source_documents": [],
            }
        else:
            # 5) Rerank candidates - use enhanced queries for better reranking
            # Pass multiple queries for more comprehensive reranking
            rerank_queries = [enhanced_query]
            if len(query_analysis.enhanced_queries) > 1:
                rerank_queries.extend(query_analysis.enhanced_queries[1:3])  # Add 2 more variants
            
            ranked_documents = await run_in_threadpool(
                rerank_documents, 
                " ".join(rerank_queries),  # Combine for richer context
                candidate_documents, 
                50
            )

            # 6) Generate the final answer using LLM with intent context
            # Add intent information to improve response generation
            intent_context = (
                f"Query Intent: {query_analysis.primary_intent.value}. "
                f"Medical entities found: {', '.join([e.text for e in query_analysis.analysis.entities])}. "
            )
            
            llm_output_data = await run_in_threadpool(
                generate_llm_response, 
                enhanced_query,  # Keep original enhanced query for LLM
                ranked_documents
            )

        # Include information about which embedding types were searched
        searched_types = embedding_types if embedding_types else list(EmbeddingType)
        
        return {
            "message": "Search logic executed successfully.",
            "enhanced_query": enhanced_query,
            "searched_embedding_types": [t.value for t in searched_types],
            "llm_response": llm_output_data.get("llm_answer_with_sequential_citations"),
            "referenced_contexts": llm_output_data.get(
                "cited_source_documents"
            ),
            "query_analysis": {
                "intent": query_analysis.primary_intent.value,
                "confidence": query_analysis.intent_confidence,
                "entities": [
                    {"text": e.text, "type": e.entity_type.value}
                    for e in query_analysis.analysis.entities
                ],
                "abbreviations_expanded": query_analysis.analysis.expanded_abbreviations,
                "corrections_made": query_analysis.analysis.corrected_terms,
            }
        }
    except Exception as e:
        logger.error(f"Search endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e!s}") from None


@router.get("/view_document", response_class=FileResponse)
async def view_document(
    file_path: str = Query(..., description="The absolute path to the document to view.")
):
    """
    Serves a document file from the specified path, enabling users to view the source document.

    - **Security**: Ensures that only files from within the project's designated 'media' directory can be accessed.
    - **file_path**: The absolute path to the file, as provided in the `saved_file_path` field of the `/search` endpoint's response.
    """
    try:
        # Security: Define the absolute path to the allowed 'media' directory.
        # This assumes the script is run from the project root or a predictable location.
        # A more robust solution might use environment variables or a settings file.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        media_root = os.path.join(project_root, "media")

        # Security: Normalize the user-provided path and the media root path.
        safe_media_root = os.path.realpath(media_root)
        requested_path = os.path.realpath(file_path)

        # Security: Check if the requested path is within the safe media directory.
        if not requested_path.startswith(safe_media_root):
            logger.warning(
                f"Forbidden access attempt for path: {file_path}. Resolved to: {requested_path}"
            )
            raise HTTPException(
                status_code=403,
                detail="Forbidden: Access is restricted to the designated media directory.",
            )

        if not os.path.exists(requested_path) or not os.path.isfile(requested_path):
            raise HTTPException(status_code=404, detail="File not found.")

        # Forcing PDF media type as we are primarily handling PDFs.
        return FileResponse(
            path=requested_path,
            media_type="application/pdf",
            filename=os.path.basename(requested_path),
        )

    except HTTPException as http_exc:
        # Re-raise HTTPException to be handled by FastAPI.
        raise http_exc
    except Exception as e:
        logger.error(f"Error serving file at path {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
