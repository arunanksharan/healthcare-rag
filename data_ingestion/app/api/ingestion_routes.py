import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, File, UploadFile, Form

from data_ingestion.app.api.models import NewDocumentMetadata, CustomDocumentType, ChunkerType
from data_ingestion.celery_worker.tasks import process_document_task
from shared.embeddings import EmbeddingType

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEDIA_DIR = PROJECT_ROOT / "media"

MEDIA_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"File uploads will be saved to: {MEDIA_DIR.resolve()}")

router = APIRouter()

@router.post("/ingest", summary="Ingest a document with metadata")
async def ingest_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    document_type: CustomDocumentType = Form(..., alias="type"),
    description: str = Form(...),
    date: str = Form(...),
    embedding_type: str = Form(default=None),
    chunker_type: str = Form(default=None),
    enable_ner: bool = Form(default=False)
):
    """
    API endpoint to receive a PDF file and its metadata, then queue for processing.
    Metadata fields (title, type, description, date) are passed as form fields.
    
    Args:
        file: The PDF file to ingest
        title: Document title
        document_type: Type of document (GUIDELINE, POLICY, FAQ, DOCUMENT)
        description: Document description
        date: Document date
        embedding_type: Optional embedding model to use
        chunker_type: Optional chunking strategy to use
        enable_ner: Enable NER-based entity extraction (default: False)
    """
    try:
        # Validate and set embedding type
        if embedding_type:
            try:
                embedding_type_enum = EmbeddingType(embedding_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid embedding_type: {embedding_type}. Valid options: {[e.value for e in EmbeddingType]}"
                )
        else:
            embedding_type_enum = EmbeddingType.get_default()
        
        # Validate and set chunker type
        if chunker_type:
            try:
                chunker_type_enum = ChunkerType(chunker_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid chunker_type: {chunker_type}. Valid options: {[e.value for e in ChunkerType]}"
                )
        else:
            chunker_type_enum = ChunkerType.get_default()
            
        metadata = NewDocumentMetadata(
            title=title,
            type=document_type,
            description=description,
            date=date,
            embedding_type=embedding_type_enum,
            chunker_type=chunker_type_enum,
            enable_ner=enable_ner
        )

        original_filename = file.filename
        if not original_filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty.")

        safe_filename = original_filename.replace("..", "")
        save_path = MEDIA_DIR / safe_filename
        logger.info(f"Saving file '{original_filename}' to '{save_path}'")

        try:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File '{original_filename}' saved to '{save_path}'")
        except Exception as e:
            logger.error(f"Error saving file '{original_filename}' to '{save_path}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

        task = process_document_task.delay(
            saved_file_path=str(save_path), 
            metadata=metadata.model_dump(),
            original_filename=original_filename
        )

        return {
            "message": "Document received and processing has started",
            "task_id": task.id,
            "filename": original_filename,
            "metadata": metadata.model_dump(),
        }
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to queue task: {str(e)}") from None
