import logging
import ssl
from celery import Celery
from celery.signals import worker_process_init
from ..core.settings import settings

# ──────────────────────────────────────────────────────────────────────────────
# PRELOAD tokenizers & embedding model
from ..utils.custom_chunker import CustomJsonChunker
from ..utils.healthcare_chunker import HealthcareChunker
from ..utils.enhanced_healthcare_chunker import EnhancedHealthcareChunker
from ..utils.qdrant import (
    chunk_exists_by_metadata,
    init_qdrant_collection,
    store_embeddings_in_qdrant,
)
from shared.embeddings import EmbeddingType
from ..utils.task_utils import (
    chunk_parsed_document,
    generate_embeddings,
    parse_document,
)

# Initialize both chunkers
CustomJsonChunker.init_tokenizer_for_worker()
HealthcareChunker.init_tokenizer_for_worker()
EnhancedHealthcareChunker.init_tokenizer_for_worker()
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

logger.info("Starting Celery worker")


app = Celery(
    "tasks",
    broker=settings.redis_url,
    broker_transport_options={
        "fanout_patterns": True,
        "fanout_prefix": True,
        "visibility_timeout": 3600,
        "ssl_cert_reqs": ssl.CERT_NONE,
    },
)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
)


@worker_process_init.connect
def on_worker_init(**kwargs):
    """
    Called once in each worker process after fork.
    Initialize collections for all embedding types.
    """
    try:
        # Initialize collections for all embedding types
        for embedding_type in EmbeddingType:
            try:
                init_qdrant_collection(settings.qdrant_collection_name, embedding_type)
                logger.info(f"Worker init: Qdrant collection for {embedding_type.value} ready")
            except Exception as e:
                logger.warning(f"Could not init collection for {embedding_type.value}: {e}")
    except Exception as e:
        logger.error(f"Worker init failed: {e}", exc_info=True)


@app.task(bind=True, name="tasks.process_document_task")
def process_document_task(self, saved_file_path: str, metadata: dict, original_filename: str):
    """
    1) Receives saved_file_path and metadata
    2) Reads file from saved_file_path
    3) Parse via LlamaParse
    3) Chunk via CustomJsonChunker
    4) Generate embeddings
    5) Upsert to Qdrant
    """
    processing_metadata = metadata.copy()
    processing_metadata["original_filename"] = original_filename

    try:
        logger.info(f"Processing document: {original_filename}, metadata: {processing_metadata}")

        # STEP 0: Read file content from saved_file_path
        try:
            with open(saved_file_path, "rb") as f:
                file_bytes = f.read()
            logger.info(f"Read {len(file_bytes)} bytes from {saved_file_path} for {original_filename}")
        except Exception as e:
            logger.error(f"Failed to read file {saved_file_path} for {original_filename}: {e}", exc_info=True)
            raise

        # Get embedding type from metadata
        embedding_type = EmbeddingType(processing_metadata.get("embedding_type", EmbeddingType.get_default().value))
        
        # STEP 1: Check if document is already parsed using new metadata fields
        uniqueness_criteria = {
            "title": processing_metadata.get("title"),
            "type": processing_metadata.get("type"),
            "date": processing_metadata.get("date"),
            "original_filename": original_filename,
        }
        if chunk_exists_by_metadata(uniqueness_criteria, embedding_type):
            logger.info(f"Document already processed based on metadata: {uniqueness_criteria}")
            return {
                "status": "skipped",
                "message": f"Document matching {uniqueness_criteria} already exists in Qdrant.",
                "filename": original_filename,
            }

        # STEP 2: Parse
        logger.info(f"Parsing document {original_filename}")
        parsed_json = parse_document(file_bytes, processing_metadata)
        if not parsed_json:
            raise ValueError(f"No data produced from parsing {original_filename!r}")
        logger.info(f"Successfully parsed {original_filename}")

        # Step 3: Chunk
        logger.info(f"Chunking parsed document {original_filename}")
        chunks = chunk_parsed_document(parsed_json, processing_metadata)
        if not chunks:
            raise ValueError(f"No chunks produced for {original_filename!r}")
        logger.info(f"Successfully generated {len(chunks)} chunks for {original_filename}")

        # Step 4: Embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks from {original_filename} using {embedding_type.value}")
        embeddings = generate_embeddings(chunks, embedding_type)
        if not embeddings:
            raise ValueError(f"No embeddings generated for {original_filename!r}")
        logger.info(f"Successfully generated {len(embeddings)} embeddings for {original_filename} using {embedding_type.value}")

        # Step 5: Upsert
        logger.info(f"Storing {len(embeddings)} embeddings in Qdrant for {original_filename}")
        metadata_for_qdrant = processing_metadata.copy()
        metadata_for_qdrant["saved_file_path"] = saved_file_path # Add saved_file_path to Qdrant metadata
        
        store_embeddings_in_qdrant(chunks, embeddings, metadata_for_qdrant, embedding_type)
        logger.info(f"Successfully processed and stored {len(chunks)} chunks for {original_filename}")

        return {
            "status": "success",
            "message": f"Processed {original_filename!r}, generated {len(chunks)} chunks and {len(embeddings)} embeddings using {embedding_type.value}.",
            "filename": original_filename,
            "metadata_processed": processing_metadata # Return the metadata used for processing
        }

    except Exception as e:
        logger.error(f"Task failed for {original_filename}: {e}", exc_info=True)
        raise
