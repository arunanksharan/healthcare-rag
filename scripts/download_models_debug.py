#!/usr/bin/env python3
"""
Script to download all medical embedding models for the healthcare RAG system.
This ensures models are cached locally before running the services.
"""
import os
import sys
from pathlib import Path
import logging

# Add project root to path - handle different execution contexts
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Debug path information
print(f"Current file: {current_file}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
print(f"Working directory: {os.getcwd()}")

try:
    from shared.embeddings import EmbeddingType, get_embedding_model
except ImportError as e:
    print(f"\nError importing shared.embeddings: {e}")
    print("\nTrying alternative import methods...")
    
    # Try to import from different paths
    possible_paths = [
        project_root,
        Path.cwd(),
        Path.cwd().parent,
        Path("/root/healthcare-rag"),
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "shared" / "embeddings.py").exists():
            sys.path.insert(0, str(path))
            try:
                from shared.embeddings import EmbeddingType, get_embedding_model
                print(f"Successfully imported from: {path}")
                break
            except ImportError:
                continue
    else:
        print("\nCould not find shared.embeddings module!")
        print("Please ensure you're running from the project root directory")
        print("and that the shared/embeddings.py file exists.")
        sys.exit(1)

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
