import logging
import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from data_retrieval.app.api.models import SearchRequest
from data_retrieval.utils.llm_generator import generate_llm_response
from data_retrieval.utils.openai_embedding import (
    get_embedding_for_query,
    get_embeddings_for_multiple_types,
)
from data_retrieval.utils.reranker import rerank_documents
from data_retrieval.utils.search import (
    search_multiple_collections,
    search_single_collection,
)
from shared.embeddings import EmbeddingType, get_collection_name
from shared.query_analysis import QueryEnhancer

logger = logging.getLogger(__name__)
router = APIRouter()
query_enhancer = QueryEnhancer()


@router.post("/search", summary="Search & generate LLM response with sequential citations")
async def search(request: SearchRequest):
    try:
        query = request.query
        embedding_types = request.embedding_types
        
        # 1) Full query enhancement with NER and intent
        enhanced_query_obj = await run_in_threadpool(query_enhancer.enhance_query, query)
        
        # Log analysis results
        logger.info(
            f"Query intent: {enhanced_query_obj.intent.value} "
            f"(confidence: {enhanced_query_obj.intent_confidence:.2f})"
        )
        
        # Log entities found
        if enhanced_query_obj.entities:
            entity_info = ", ".join([
                f"{entity_type}: {entities}"
                for entity_type, entities in enhanced_query_obj.entities.items()
            ])
            logger.info(f"Medical entities: {entity_info}")
        
        # 2) Get search strategy
        search_strategy = query_enhancer.get_search_strategy(enhanced_query_obj)
        
        # 3) Use the primary enhanced query for embeddings
        primary_query = enhanced_query_obj.cleaned_query
        
        # 4) Determine which embedding types to search
        if embedding_types is None:
            # Use default or multiple types based on confidence
            if enhanced_query_obj.intent_confidence > 0.7:
                # High confidence - use default medical embedding
                embedding_types = [EmbeddingType.get_default()]
            else:
                # Low confidence - search multiple types
                embedding_types = [EmbeddingType.PUBMEDBERT, EmbeddingType.BIOBERT]
        
        # 5) Perform enhanced search with entity filtering
        if len(embedding_types) == 1:
            # Single embedding type search
            embedding_type = embedding_types[0]
            query_vector = await run_in_threadpool(
                get_embedding_for_query,
                primary_query,
                embedding_type
            )
            
            # Get collection name
            from data_retrieval.core.settings import settings
            collection_name = get_collection_name(settings.qdrant_collection_name, embedding_type)
            
            # Search with both intent and entity filters
            candidate_documents = await run_in_threadpool(
                search_single_collection,
                embedding=query_vector,
                collection_name=collection_name,
                metadata=None,
                intent_filters=search_strategy["boost_params"],
                entity_filters=search_strategy["filters"],
                top_k=50
            )
        else:
            # Multiple embedding type search
            query_embeddings = await run_in_threadpool(
                get_embeddings_for_multiple_types,
                primary_query,
                embedding_types
            )
            
            # Check if we got any embeddings
            if not query_embeddings:
                logger.error(f"Failed to generate embeddings for any of the requested types: {[t.value for t in embedding_types]}")
                candidate_documents = []
            else:
                # Search multiple collections with filters
                candidate_documents = await run_in_threadpool(
                    search_multiple_collections,
                    query_embeddings=query_embeddings,
                    metadata=None,
                    intent_filters=search_strategy["boost_params"],
                    entity_filters=search_strategy["filters"],
                    top_k_per_collection=50
                )

        llm_output_data: dict

        if not candidate_documents:
            llm_output_data = { 
                "llm_answer_with_sequential_citations": "No relevant documents found to answer the query based on initial search.",  # noqa: E501
                "cited_source_documents": [],
            }
        else:
            # 6) Rerank candidates using all query variants
            rerank_query = " ".join(enhanced_query_obj.get_all_query_texts()[:3])
            
            ranked_documents = await run_in_threadpool(
                rerank_documents, 
                rerank_query,
                candidate_documents, 
                50
            )

            # 7) Generate the final answer using LLM with context
            llm_output_data = await run_in_threadpool(
                generate_llm_response, 
                primary_query,
                ranked_documents
            )

        # Include comprehensive information in response
        searched_types = embedding_types if embedding_types else list(EmbeddingType)
        
        return {
            "message": "Search logic executed successfully.",
            "enhanced_query": primary_query,
            "query_variants": enhanced_query_obj.query_variants,
            "searched_embedding_types": [t.value for t in searched_types],
            "llm_response": llm_output_data.get("llm_answer_with_sequential_citations"),
            "referenced_contexts": llm_output_data.get(
                "cited_source_documents"
            ),
            "query_analysis": {
                "intent": enhanced_query_obj.intent.value,
                "confidence": enhanced_query_obj.intent_confidence,
                "entities": enhanced_query_obj.entities,
                "filters_applied": search_strategy["filters"],
                "boost_params": search_strategy["boost_params"],
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
