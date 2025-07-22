import logging
from fastapi import FastAPI

from data_retrieval.app.api.retrieval_routes import router as retrieval_router
from .settings import settings
from shared.embeddings import EmbeddingType, EmbeddingRegistry

# Configure logging at the application's entry point
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Data Retrieval Service", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup: preload default embedding model for faster first query.
    """
    try:
        # Preload the default embedding model
        logger.info("Preloading default embedding model...")
        EmbeddingRegistry.get_instance().preload_models([EmbeddingType.get_default()])
        logger.info(f"Default embedding model {EmbeddingType.get_default().value} preloaded")
    except Exception as e:
        logger.warning(f"Could not preload embedding model: {e}")


@app.get("/retrieval/health", summary="Health Check", tags=["Health"])
def health_check():
    return {"status": "ok"}


@app.get("/health", summary="Health Check", tags=["Health"])
def health_check_outer():
    return {"status": "ok"}


# Mount the retrieval router under /retrieval/api/v1
app.include_router(retrieval_router, prefix="/retrieval/api/v1")


@app.get("/retrieval")
def root():
    return {"message": "Data Retrieval Service Running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("data_retrieval.core.main:app", host="0.0.0.0", port=8001, reload=True)
