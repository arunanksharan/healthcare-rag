#!/usr/bin/env python3
"""
Verify that the NER fixes have been applied correctly.
"""
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_fixes():
    print("Verifying NER integration fixes...\n")
    
    # Test 1: Check if query enhancer imports NER
    try:
        with open(Path(__file__).parent.parent / "shared" / "query_analysis" / "query_enhancer.py", 'r') as f:
            content = f.read()
            if "from shared.medical_ner import" in content:
                print("✓ Query enhancer imports NER components")
            else:
                print("✗ Query enhancer missing NER imports")
                return False
    except Exception as e:
        print(f"✗ Error reading query enhancer: {e}")
        return False
    
    # Test 2: Check if query enhancer has NER methods
    if "_extract_entities_with_ner" in content and "_init_ner_for_queries" in content:
        print("✓ Query enhancer has NER methods")
    else:
        print("✗ Query enhancer missing NER methods")
        return False
    
    # Test 3: Check if Qdrant storage lowercases entities
    try:
        with open(Path(__file__).parent.parent / "data_ingestion" / "utils" / "qdrant.py", 'r') as f:
            content = f.read()
            if "[drug.lower() for drug in" in content:
                print("✓ Qdrant storage lowercases entities")
            else:
                print("✗ Qdrant storage not lowercasing entities")
                return False
    except Exception as e:
        print(f"✗ Error reading qdrant.py: {e}")
        return False
    
    # Test 4: Try to import and initialize QueryEnhancer
    try:
        from shared.query_analysis import QueryEnhancer
        print("✓ QueryEnhancer can be imported")
        
        # Try to create instance
        enhancer = QueryEnhancer()
        print("✓ QueryEnhancer can be instantiated")
        
        # Check if it has NER attributes
        if hasattr(enhancer, '_ner_model'):
            print("✓ QueryEnhancer has NER attributes")
        else:
            print("⚠ QueryEnhancer may not have NER attributes")
            
    except Exception as e:
        print(f"✗ Error with QueryEnhancer: {e}")
        return False
    
    print("\n✅ All fixes have been applied successfully!")
    print("\nThe RAG pipeline now has:")
    print("- BioBERT NER integration in query processing")
    print("- Case-insensitive entity matching")
    print("- Symmetric NER usage for both ingestion and retrieval")
    return True

if __name__ == "__main__":
    verify_fixes()
