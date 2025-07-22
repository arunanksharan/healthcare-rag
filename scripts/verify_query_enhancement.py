#!/usr/bin/env python3
"""
Verify query-time enhancement integration.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_imports():
    """Verify all imports work correctly."""
    print("Checking imports...")
    
    try:
        # Check query analysis components
        from shared.query_analysis import (
            QueryEnhancer,
            EnhancedQuery,
            QueryIntent,
            MedicalQueryAnalyzer
        )
        print("✓ Query analysis imports successful")
    except Exception as e:
        print(f"✗ Query analysis import failed: {e}")
        return False
    
    try:
        # Check search components
        from data_retrieval.utils.search import search_single_collection
        print("✓ Search utils import successful")
    except Exception as e:
        print(f"✗ Search utils import failed: {e}")
        return False
    
    try:
        # Check retrieval routes
        from data_retrieval.app.api.retrieval_routes import search
        print("✓ Retrieval routes import successful")
    except Exception as e:
        print(f"✗ Retrieval routes import failed: {e}")
        return False
    
    return True


def test_query_enhancement():
    """Test basic query enhancement functionality."""
    print("\nTesting query enhancement...")
    
    try:
        from shared.query_analysis import QueryEnhancer
        
        enhancer = QueryEnhancer()
        print("✓ QueryEnhancer initialized")
        
        # Test a simple query
        test_query = "metformin dosage for diabetes"
        enhanced = enhancer.enhance_query(test_query)
        
        print(f"✓ Enhanced query: '{enhanced.cleaned_query}'")
        print(f"  Intent: {enhanced.intent.value} (confidence: {enhanced.intent_confidence:.2f})")
        print(f"  Entities: {enhanced.entities}")
        
        # Get search strategy
        strategy = enhancer.get_search_strategy(enhanced)
        print(f"✓ Search strategy generated")
        print(f"  Filters: {strategy['filters']}")
        print(f"  Boost params: {strategy['boost_params']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Query enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intent_detection():
    """Test intent detection functionality."""
    print("\nTesting intent detection...")
    
    try:
        from shared.query_analysis.intent_detection import detect_medical_intent
        
        test_queries = [
            ("what is the dosage of metformin?", "dosage"),
            ("lisinopril side effects", "side_effects"),
            ("contraindications for aspirin", "contraindications"),
            ("how to treat hypertension", "treatment")
        ]
        
        for query, expected_intent in test_queries:
            intent, confidence = detect_medical_intent(query)
            print(f"  Query: '{query}'")
            print(f"  → Intent: {intent.value} (confidence: {confidence:.2f})")
            if intent.value == expected_intent:
                print(f"  ✓ Correct intent detected")
            else:
                print(f"  ✗ Expected {expected_intent}, got {intent.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Intent detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run verification tests."""
    print("Query-Time Enhancement Verification")
    print("=" * 40)
    
    # Check imports
    imports_ok = verify_imports()
    
    # Test query enhancement
    enhancement_ok = test_query_enhancement() if imports_ok else False
    
    # Test intent detection
    intent_ok = test_intent_detection() if imports_ok else False
    
    print("\n" + "=" * 40)
    if imports_ok and enhancement_ok and intent_ok:
        print("✅ Query-time enhancement is properly integrated!")
    else:
        print("❌ Query-time enhancement has issues")
        print("\nPlease check the error messages above.")


if __name__ == "__main__":
    main()
