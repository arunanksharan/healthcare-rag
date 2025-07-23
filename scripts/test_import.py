#!/usr/bin/env python3
"""
Test script to verify that the shared module can be imported correctly.
"""
import os
import sys
from pathlib import Path

# Add project root to path so Python can find shared, data_ingestion, data_retrieval modules
project_root = Path(__file__).parent.parent  # healthcare-rag/
project_root_str = str(project_root.resolve())

print(f"Script location: {Path(__file__).resolve()}")
print(f"Project root: {project_root_str}")
print(f"Project root exists: {project_root.exists()}")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
    print(f"Added to Python path: {project_root_str}")

# Debug: Print the path and verify shared directory exists
shared_dir = project_root / "shared"
print(f"Looking for shared module at: {shared_dir}")
print(f"Shared directory exists: {shared_dir.exists()}")
print(f"Shared __init__.py exists: {(shared_dir / '__init__.py').exists()}")
print(f"Shared/embeddings directory exists: {(shared_dir / 'embeddings').exists()}")
print(f"Shared/embeddings __init__.py exists: {(shared_dir / 'embeddings' / '__init__.py').exists()}")

print(f"\nCurrent Python path:")
for i, path in enumerate(sys.path):
    if 'healthcare-rag' in path or i < 3:
        print(f"  {i}: {path}")

print("\nTrying to import shared module...")
try:
    import shared
    print("✅ Successfully imported 'shared'")
    print(f"Shared module location: {shared.__file__}")
except ImportError as e:
    print(f"❌ Failed to import 'shared': {e}")

print("\nTrying to import shared.embeddings...")
try:
    import shared.embeddings
    print("✅ Successfully imported 'shared.embeddings'")
    print(f"Embeddings module location: {shared.embeddings.__file__}")
except ImportError as e:
    print(f"❌ Failed to import 'shared.embeddings': {e}")

print("\nTrying to import specific items...")
try:
    from shared.embeddings import EmbeddingType, get_embedding_model
    print("✅ Successfully imported EmbeddingType and get_embedding_model")
except ImportError as e:
    print(f"❌ Failed to import EmbeddingType and get_embedding_model: {e}")
