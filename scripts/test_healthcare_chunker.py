#!/usr/bin/env python3
"""
Test script to verify the healthcare chunker integration.
"""
import json
from data_ingestion.app.api.models import ChunkerType
from data_ingestion.utils.healthcare_chunker import HealthcareChunker
from data_ingestion.utils.custom_chunker import CustomJsonChunker

# Sample parsed JSON structure (minimal example)
sample_parsed_json = {
    "job_id": "test_123",
    "pages": [
        {
            "page": 1,
            "width": 612,
            "height": 792,
            "items": [
                {
                    "type": "heading",
                    "bBox": {"x": 50, "y": 50, "w": 500, "h": 30},
                    "md": "Medications",
                    "level": 1
                },
                {
                    "type": "text",
                    "bBox": {"x": 50, "y": 100, "w": 500, "h": 100},
                    "md": "1. Metformin 500mg PO BID for diabetes management\n2. Lisinopril 10mg PO daily for hypertension"
                },
                {
                    "type": "heading",
                    "bBox": {"x": 50, "y": 220, "w": 500, "h": 30},
                    "md": "Lab Results",
                    "level": 1
                },
                {
                    "type": "table",
                    "bBox": {"x": 50, "y": 260, "w": 500, "h": 150},
                    "md": "Test | Result | Range\nGlucose | 126 | 70-100\nHbA1c | 7.2 | <5.7",
                    "rows": [
                        ["Test", "Result", "Range"],
                        ["Glucose", "126", "70-100"],
                        ["HbA1c", "7.2", "<5.7"]
                    ]
                }
            ],
            "images": [
                {
                    "x": 50,
                    "y": 450,
                    "width": 200,
                    "height": 100,
                    "ocr": [
                        {"text": "Patient ID: 12345"},
                        {"text": "Date: 2025-01-15"}
                    ]
                }
            ]
        }
    ]
}

sample_metadata = {
    "original_filename": "patient_record.pdf",
    "title": "Patient Medical Record",
    "type": "document",
    "description": "Clinical documentation",
    "date": "2025-01-15",
    "embedding_type": "pubmedbert",
    "chunker_type": ChunkerType.HEALTHCARE.value,
    "parse_type": "pdf"
}

def test_healthcare_chunker():
    """Test the healthcare chunker implementation."""
    print("Testing Healthcare Chunker Implementation")
    print("=" * 50)
    
    # Initialize the healthcare chunker
    chunker = HealthcareChunker.get_instance()
    if not hasattr(HealthcareChunker, '_tokenizer_instance') or HealthcareChunker._tokenizer_instance is None:
        print("Initializing PubMedBERT tokenizer...")
        HealthcareChunker.init_tokenizer_for_worker()
    
    # Generate chunks
    print("\nGenerating chunks...")
    chunks = chunker.chunk_json(sample_parsed_json, sample_metadata)
    
    print(f"\nGenerated {len(chunks)} chunks")
    print("-" * 50)
    
    # Display chunks
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Type: {chunk.get('chunk_type', 'unknown')}")
        print(f"  Page: {chunk.get('page', 'NA')}")
        print(f"  Page Width: {chunk.get('page_width', 'NA')}")
        print(f"  Page Height: {chunk.get('page_height', 'NA')}")
        print(f"  Content Preview: {chunk.get('chunk', '')[:100]}...")
        if 'medical_type' in chunk:
            print(f"  Medical Type: {chunk['medical_type']}")
        if 'bbox' in chunk:
            print(f"  BBox: {chunk['bbox']}")
        print()
    
    # Verify critical fields
    print("\nVerification:")
    print("-" * 50)
    
    all_have_page_dims = all(
        chunk.get('page_width') is not None and 
        chunk.get('page_height') is not None 
        for chunk in chunks
    )
    print(f"✓ All chunks have page dimensions: {all_have_page_dims}")
    
    has_section_context = any('[Section:' in chunk.get('chunk', '') for chunk in chunks)
    print(f"✓ Chunks have section context: {has_section_context}")
    
    chunk_types = set(chunk.get('chunk_type') for chunk in chunks)
    print(f"✓ Chunk types found: {chunk_types}")
    
    # Test comparison with generic chunker
    print("\n\nComparing with Generic Chunker:")
    print("=" * 50)
    
    generic_chunker = CustomJsonChunker.get_instance()
    generic_metadata = sample_metadata.copy()
    generic_metadata['chunker_type'] = ChunkerType.GENERIC.value
    
    generic_chunks = generic_chunker.chunk_json(sample_parsed_json, generic_metadata)
    print(f"Generic chunker produced {len(generic_chunks)} chunks")
    print(f"Healthcare chunker produced {len(chunks)} chunks")
    
    print("\nDifferences:")
    print(f"- Healthcare chunker has medical-specific types: {any('medical_type' in c for c in chunks)}")
    print(f"- Healthcare chunker preserves sections: {has_section_context}")
    print(f"- Healthcare chunker uses smaller chunks: {len(chunks) > len(generic_chunks)}")

if __name__ == "__main__":
    test_healthcare_chunker()
