#!/usr/bin/env python3
"""
Integration test for the complete query processing pipeline.
Tests chunking with metadata and retrieval with intent-based filtering.
"""
import json
import tempfile
from pathlib import Path

from data_ingestion.utils.healthcare_chunker import HealthcareChunker
from shared.query_analysis import EnhancedQueryProcessor, QueryIntent

# Test document that covers multiple intents
TEST_DOCUMENT = {
    "job_id": "test_integration_123",
    "pages": [
        {
            "page": 1,
            "width": 612,
            "height": 792,
            "items": [
                {
                    "type": "heading",
                    "level": 1,
                    "md": "Metformin (Glucophage)",
                    "bBox": {"x": 50, "y": 50, "w": 500, "h": 30}
                },
                {
                    "type": "text",
                    "md": "Metformin is an oral antidiabetic medication used to treat type 2 diabetes mellitus. It belongs to the biguanide class of medications.",
                    "bBox": {"x": 50, "y": 100, "w": 500, "h": 50}
                },
                {
                    "type": "heading",
                    "level": 2,
                    "md": "Dosage and Administration",
                    "bBox": {"x": 50, "y": 170, "w": 500, "h": 25}
                },
                {
                    "type": "text",
                    "md": "Initial dose: 500 mg orally twice a day or 850 mg once daily with meals. Maximum dose: 2550 mg per day in divided doses.",
                    "bBox": {"x": 50, "y": 200, "w": 500, "h": 50}
                },
                {
                    "type": "heading",
                    "level": 2,
                    "md": "Contraindications",
                    "bBox": {"x": 50, "y": 270, "w": 500, "h": 25}
                },
                {
                    "type": "text",
                    "md": "Do not use in patients with severe renal impairment (eGFR below 30), acute or chronic metabolic acidosis, or history of lactic acidosis.",
                    "bBox": {"x": 50, "y": 300, "w": 500, "h": 50}
                },
                {
                    "type": "heading",
                    "level": 2,
                    "md": "Adverse Reactions",
                    "bBox": {"x": 50, "y": 370, "w": 500, "h": 25}
                },
                {
                    "type": "text",
                    "md": "Common side effects include gastrointestinal upset, diarrhea, nausea, and abdominal discomfort. Rare but serious: lactic acidosis.",
                    "bBox": {"x": 50, "y": 400, "w": 500, "h": 50}
                }
            ]
        }
    ]
}

def test_chunking_with_metadata():
    """Test that the healthcare chunker adds proper metadata."""
    print("\n=== Testing Healthcare Chunker Metadata ===\n")
    
    # Initialize chunker
    chunker = HealthcareChunker.get_instance()
    if not hasattr(HealthcareChunker, '_tokenizer_instance') or HealthcareChunker._tokenizer_instance is None:
        HealthcareChunker.init_tokenizer_for_worker()
    
    # Chunk the document
    metadata = {
        "original_filename": "metformin_info.pdf",
        "parse_type": "pdf"
    }
    
    chunks = chunker.chunk_json(TEST_DOCUMENT, metadata)
    
    print(f"Generated {len(chunks)} chunks\n")
    
    # Analyze each chunk
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  Text: {chunk['chunk'][:80]}...")
        print(f"  Type: {chunk['chunk_type']}")
        print(f"  Answer Types: {chunk.get('answer_types', [])}")
        print(f"  Medical Entities: {chunk.get('medical_entities', [])}")
        print(f"  Boost Section: {chunk.get('boost_section', 'None')}")
        print(f"  Section: {chunk.get('section_title', 'None')}")
        print()
    
    return chunks


def test_query_intent_matching(chunks):
    """Test that query intents match the chunk metadata."""
    print("\n=== Testing Query Intent Matching ===\n")
    
    processor = EnhancedQueryProcessor()
    
    test_queries = [
        ("what is metformin", QueryIntent.DEFINITION),
        ("metformin dosage", QueryIntent.DOSAGE),
        ("metformin side effects", QueryIntent.SIDE_EFFECTS),
        ("when not to use metformin", QueryIntent.CONTRAINDICATIONS),
    ]
    
    for query, expected_intent in test_queries:
        print(f"\nQuery: '{query}'")
        result = processor.process_query(query)
        
        print(f"  Detected Intent: {result.primary_intent.value} (expected: {expected_intent.value})")
        print(f"  Confidence: {result.intent_confidence:.2f}")
        print(f"  Entities: {[e.text for e in result.analysis.entities]}")
        
        # Check which chunks would match
        metadata_filters = result.metadata_filters
        boost_sections = metadata_filters.get("boost_sections", [])
        
        matching_chunks = []
        for i, chunk in enumerate(chunks):
            # Check if chunk matches the query intent
            chunk_answer_types = chunk.get("answer_types", [])
            chunk_boost_section = chunk.get("boost_section")
            
            # Check if this chunk would be boosted
            is_boosted = False
            for section in boost_sections:
                if section in chunk_answer_types or chunk_boost_section == section:
                    is_boosted = True
                    break
            
            if is_boosted:
                matching_chunks.append(i+1)
        
        print(f"  Matching chunks: {matching_chunks}")


def test_end_to_end_flow():
    """Test the complete flow from document to retrieval."""
    print("\n=== End-to-End Integration Test ===\n")
    
    # Step 1: Chunk document
    chunks = test_chunking_with_metadata()
    
    # Step 2: Test query matching
    test_query_intent_matching(chunks)
    
    # Step 3: Verify metadata structure
    print("\n=== Metadata Structure Verification ===\n")
    
    required_fields = [
        "chunk", "chunk_type", "page", "answer_types", 
        "medical_entities", "entity_types", "has_medical_content"
    ]
    
    all_valid = True
    for i, chunk in enumerate(chunks):
        missing_fields = [field for field in required_fields if field not in chunk]
        if missing_fields:
            print(f"Chunk {i+1} missing fields: {missing_fields}")
            all_valid = False
    
    if all_valid:
        print("✓ All chunks have required metadata fields")
    else:
        print("✗ Some chunks are missing required fields")
    
    # Step 4: Check retrieval compatibility
    print("\n=== Retrieval Compatibility Check ===\n")
    
    # Simulate what retrieval expects
    sample_filters = {
        "chunk_types": ["text", "medication"],
        "boost_sections": ["dosage", "contraindications"],
    }
    
    print("Retrieval filters:")
    print(f"  Chunk types: {sample_filters['chunk_types']}")
    print(f"  Boost sections: {sample_filters['boost_sections']}")
    
    # Check how many chunks would match
    type_matches = sum(1 for c in chunks if c.get('chunk_type') in sample_filters['chunk_types'])
    section_matches = sum(1 for c in chunks 
                         if any(s in c.get('answer_types', []) or c.get('boost_section') == s 
                               for s in sample_filters['boost_sections']))
    
    print(f"\nMatching chunks:")
    print(f"  By type: {type_matches}/{len(chunks)}")
    print(f"  By section: {section_matches}/{len(chunks)}")
    
    print("\n✓ Pipeline integration test complete!")


if __name__ == "__main__":
    test_end_to_end_flow()
