import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, File, UploadFile, Form

from data_ingestion.app.api.models import NewDocumentMetadata, CustomDocumentType
from data_ingestion.celery_worker.tasks import process_document_task

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
    date: str = Form(...)
):
    """
    API endpoint to receive a PDF file and its metadata, then queue for processing.
    Metadata fields (title, type, description, date) are passed as form fields.
    """
    try:
        metadata = NewDocumentMetadata(
            title=title,
            type=document_type,
            description=description,
            date=date
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
