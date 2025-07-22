#!/usr/bin/env python3
"""
Comprehensive verification of the end-to-end pipeline.
"""
import json
import sys
from typing import Dict, List, Any

# Add project root to path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.utils.healthcare_chunker import HealthcareChunker
from shared.query_analysis import EnhancedQueryProcessor, QueryIntent


def verify_chunk_metadata(chunk: Dict[str, Any], chunk_num: int) -> List[str]:
    """Verify a chunk has all required metadata."""
    issues = []
    
    # Required fields
    required_fields = [
        "chunk", "chunk_type", "page", "answer_types", 
        "medical_entities", "entity_types", "has_medical_content"
    ]
    
    for field in required_fields:
        if field not in chunk:
            issues.append(f"Chunk {chunk_num}: Missing required field '{field}'")
    
    # Check answer_types is a list
    if "answer_types" in chunk and not isinstance(chunk["answer_types"], list):
        issues.append(f"Chunk {chunk_num}: answer_types should be a list")
    
    # Check if boost_section matches answer_types
    boost_section = chunk.get("boost_section")
    answer_types = chunk.get("answer_types", [])
    
    # For chunks with boost_section, it should relate to answer_types
    if boost_section and boost_section not in answer_types:
        # This is OK - boost_section is from section, answer_types is from content
        pass
    
    return issues


def verify_query_retrieval_match(query_result: Dict, chunks: List[Dict]) -> List[str]:
    """Verify query processing results match chunk metadata."""
    issues = []
    
    # Get expected filters
    metadata_filters = query_result.get("metadata_filters", {})
    chunk_types = metadata_filters.get("chunk_types", [])
    boost_sections = metadata_filters.get("boost_sections", [])
    
    # Check if any chunks would match
    matching_by_type = 0
    matching_by_section = 0
    
    for i, chunk in enumerate(chunks):
        # Check type match
        if chunk.get("chunk_type") in chunk_types:
            matching_by_type += 1
        
        # Check section match (both boost_section and answer_types)
        chunk_boost = chunk.get("boost_section")
        chunk_answers = chunk.get("answer_types", [])
        
        for section in boost_sections:
            if chunk_boost == section or section in chunk_answers:
                matching_by_section += 1
                break
    
    if matching_by_type == 0:
        issues.append(f"No chunks match query chunk types: {chunk_types}")
    
    if matching_by_section == 0:
        issues.append(f"No chunks match query boost sections: {boost_sections}")
    
    return issues


def main():
    """Run comprehensive verification."""
    print("=== END-TO-END PIPELINE VERIFICATION ===\n")
    
    all_issues = []
    
    # Test document
    test_doc = {
        "job_id": "verify_123",
        "pages": [{
            "page": 1,
            "width": 612,
            "height": 792,
            "items": [
                {
                    "type": "heading",
                    "level": 1,
                    "md": "Medications",
                    "bBox": {"x": 50, "y": 50, "w": 500, "h": 30}
                },
                {
                    "type": "text",
                    "md": "Metformin 500mg: Take twice daily with meals for diabetes management. Maximum dose is 2000mg per day.",
                    "bBox": {"x": 50, "y": 100, "w": 500, "h": 50}
                },
                {
                    "type": "heading",
                    "level": 1,
                    "md": "Contraindications",
                    "bBox": {"x": 50, "y": 170, "w": 500, "h": 30}
                },
                {
                    "type": "text",
                    "md": "Do not use metformin in patients with severe kidney disease or metabolic acidosis.",
                    "bBox": {"x": 50, "y": 200, "w": 500, "h": 50}
                }
            ]
        }]
    }
    
    # Step 1: Test chunking
    print("1. Testing Healthcare Chunker...")
    chunker = HealthcareChunker.get_instance()
    if not hasattr(HealthcareChunker, '_tokenizer_instance') or HealthcareChunker._tokenizer_instance is None:
        HealthcareChunker.init_tokenizer_for_worker()
    
    chunks = chunker.chunk_json(test_doc, {"original_filename": "test.pdf", "parse_type": "pdf"})
    print(f"   Generated {len(chunks)} chunks")
    
    # Verify chunk metadata
    for i, chunk in enumerate(chunks):
        issues = verify_chunk_metadata(chunk, i+1)
        all_issues.extend(issues)
    
    # Step 2: Test query processing
    print("\n2. Testing Query Processing...")
    processor = EnhancedQueryProcessor()
    
    test_queries = [
        ("metformin dosage", QueryIntent.DOSAGE),
        ("contraindications for metformin", QueryIntent.CONTRAINDICATIONS),
    ]
    
    for query, expected_intent in test_queries:
        result = processor.process_query(query)
        print(f"\n   Query: '{query}'")
        print(f"   Intent: {result.primary_intent.value} (expected: {expected_intent.value})")
        
        if result.primary_intent != expected_intent:
            all_issues.append(f"Query '{query}' got intent {result.primary_intent.value}, expected {expected_intent.value}")
        
        # Verify retrieval would work
        issues = verify_query_retrieval_match(result.__dict__, chunks)
        all_issues.extend(issues)
        
        # Show what would match
        metadata_filters = result.metadata_filters
        boost_sections = metadata_filters.get("boost_sections", [])
        
        print(f"   Boost sections: {boost_sections}")
        
        matching_chunks = []
        for i, chunk in enumerate(chunks):
            # Check both boost_section and answer_types
            if chunk.get("boost_section") in boost_sections:
                matching_chunks.append(f"{i+1} (via boost_section)")
            elif any(s in chunk.get("answer_types", []) for s in boost_sections):
                matching_chunks.append(f"{i+1} (via answer_types)")
        
        print(f"   Matching chunks: {matching_chunks if matching_chunks else 'NONE'}")
    
    # Step 3: Display chunk details
    print("\n3. Chunk Details:")
    for i, chunk in enumerate(chunks):
        print(f"\n   Chunk {i+1}:")
        print(f"   - Text: {chunk['chunk'][:60]}...")
        print(f"   - Type: {chunk['chunk_type']}")
        print(f"   - Answer Types: {chunk.get('answer_types', [])}")
        print(f"   - Boost Section: {chunk.get('boost_section', 'None')}")
        print(f"   - Medical Entities: {chunk.get('medical_entities', [])}")
        print(f"   - Section: {chunk.get('section_title', 'None')}")
    
    # Summary
    print("\n=== VERIFICATION SUMMARY ===")
    if all_issues:
        print(f"\n❌ Found {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"   - {issue}")
    else:
        print("\n✅ All verifications passed!")
    
    return len(all_issues) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
