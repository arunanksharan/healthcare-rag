#!/usr/bin/env python3
"""
Script to download all medical embedding models for the healthcare RAG system.
This ensures models are cached locally before running the services.
"""
import os
import sys
from pathlib import Path

# Add project root to path so Python can find shared, data_ingestion, data_retrieval modules
project_root = Path(__file__).parent.parent  # healthcare-rag/
project_root_str = str(project_root.resolve())

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
    print(f"Added to Python path: {project_root_str}")

# Debug: Print the path and verify shared directory exists
shared_dir = project_root / "shared"
print(f"Looking for shared module at: {shared_dir}")
print(f"Shared directory exists: {shared_dir.exists()}")
print(f"Current Python path includes: {[p for p in sys.path if 'healthcare-rag' in p]}")

# Try to import the required modules with error handling
try:
    from shared.embeddings import EmbeddingType, get_embedding_model
    print("✅ Successfully imported from shared.embeddings")
except ImportError as e:
    print(f"❌ Failed to import from shared.embeddings: {e}")
    print("\nTrying alternative import approaches...")
    
    # Alternative 1: Try adding current directory
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Alternative 2: Try adding parent directory differently
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
        print(f"Added alternative path: {parent_dir}")
    
    # Alternative 3: Try relative import approach
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
        from embeddings import EmbeddingType, get_embedding_model
        print("✅ Successfully imported using alternative approach")
    except ImportError as e2:
        print(f"❌ All import attempts failed. Last error: {e2}")
        print("\nDebugging information:")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {Path(__file__).parent}")
        print(f"Project root: {Path(__file__).parent.parent}")
        print(f"Python path: {sys.path[:5]}...")  # First 5 entries
        raise ImportError(f"Cannot import required modules. Original error: {e}")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_models():
    """Download and cache all embedding models."""
    print("=" * 60)
    print("Healthcare RAG - Embedding Model Downloader")
    print("=" * 60)
    print()
    
    # Skip OpenAI as it doesn't need downloading
    models_to_download = [
        EmbeddingType.PUBMEDBERT,
        EmbeddingType.BIOBERT,
        EmbeddingType.SCIBERT,
        EmbeddingType.CLINICALBERT,
        EmbeddingType.BIOLINKBERT,
    ]
    
    for i, embedding_type in enumerate(models_to_download, 1):
        print(f"[{i}/{len(models_to_download)}] Downloading {embedding_type.value}...")
        try:
            # Get the model (this will trigger download if not cached)
            model = get_embedding_model(embedding_type)
            
            # Test the model with a sample text
            test_text = "This is a test for medical embedding model."
            embedding = model.embed_text(test_text)
            
            print(f"✓ Successfully downloaded and tested {embedding_type.value}")
            print(f"  Embedding dimension: {len(embedding)}")
            print()
            
        except Exception as e:
            print(f"✗ Failed to download {embedding_type.value}: {e}")
            print()
    
    print("=" * 60)
    print("Model download complete!")
    print(f"Models are cached in: {os.path.join(project_root, 'model_cache')}")
    print("=" * 60)


if __name__ == "__main__":
    download_models()
