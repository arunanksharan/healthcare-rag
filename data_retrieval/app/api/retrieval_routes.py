import logging
import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from data_retrieval.app.api.models import SearchRequest
from data_retrieval.utils.llm_generator import generate_llm_response
from data_retrieval.utils.openai_embedding import get_openai_embedding
from data_retrieval.utils.query_enhancer import enhance_query
from data_retrieval.utils.reranker import rerank_documents
from data_retrieval.utils.search import search_with_metadata_and_embedding

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", summary="Search & generate LLM response with sequential citations")
async def search(request: SearchRequest):
    try:
        query = request.query
        # 1) Enhance the query off-loop
        enhanced_query = await run_in_threadpool(enhance_query, query)

        # 2) Vector search off-loop
        query_vector = await run_in_threadpool(get_openai_embedding, enhanced_query)
        candidate_documents = await run_in_threadpool(
            search_with_metadata_and_embedding,
            embedding=query_vector,
            metadata=None,
        )

        llm_output_data: dict

        if not candidate_documents:
            llm_output_data = { 
                "llm_answer_with_sequential_citations": "No relevant documents found to answer the query based on initial search.",  # noqa: E501
                "cited_source_documents": [],
            }
        else:
            # 3) Re-rank off-loop
            ranked_documents = await run_in_threadpool(rerank_documents, enhanced_query, candidate_documents, 50)

            # 4) LLM answer off-loop
            # generate_llm_response now returns a dictionary with sequentially numbered citations
            # and a list of only the cited source documents.
            llm_output_data = await run_in_threadpool(generate_llm_response, enhanced_query, ranked_documents)

        return {
            "message": "Search logic executed successfully.",
            "enhanced_query": enhanced_query,
            "llm_response": llm_output_data.get("llm_answer_with_sequential_citations"),
            "referenced_contexts": llm_output_data.get(
                "cited_source_documents"
            ),
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
