import logging
from fastapi import FastAPI

from ..app.api.ingestion_routes import router as ingestion_router
from ..utils.qdrant import init_qdrant_collection
from .settings import settings
from shared.embeddings import EmbeddingType, EmbeddingRegistry

# Configure logging at the application's entry point
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


app = FastAPI(title="Data Ingestion Service", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup: ensure Qdrant collections exist and preload default embedding model.
    """
    try:
        # Initialize collections for all embedding types
        for embedding_type in EmbeddingType:
            try:
                init_qdrant_collection(settings.qdrant_collection_name, embedding_type)
                logger.info(f"Startup: Qdrant collection for {embedding_type.value} ready")
            except Exception as e:
                logger.warning(f"Could not init collection for {embedding_type.value}: {e}")
        
        # Preload the default embedding model
        logger.info("Preloading default embedding model...")
        EmbeddingRegistry.get_instance().preload_models([EmbeddingType.get_default()])
        logger.info(f"Default embedding model {EmbeddingType.get_default().value} preloaded")
        
    except Exception as e:
        logger.error(f"Startup: Failed to initialize: {e}")


@app.get("/ingestion/health", summary="Health Check", tags=["Health"])
def health_check():
    return {"status": "ok"}


@app.get("/health", summary="Health Check", tags=["Health"])
def health_check_outer():
    return {"status": "ok"}


app.include_router(ingestion_router, prefix="/ingestion/api/v1")


@app.get("/ingestion")
def root():
    return {"message": "Data Ingestion Service Running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("data_ingestion.core.main:app", host="0.0.0.0", port=8000, reload=True)
