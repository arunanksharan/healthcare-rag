#!/usr/bin/env python3
"""
Test script to verify multi-collection search with filters works correctly.
"""
import asyncio
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_multi_collection_search():
    """Test the multi-collection search with filters."""
    try:
        # Import required modules
        from data_retrieval.utils.search import search_multiple_collections
        from data_retrieval.utils.openai_embedding import get_embeddings_for_multiple_types
        from shared.embeddings import EmbeddingType
        from shared.query_analysis import QueryEnhancer
        
        # Initialize components
        query_enhancer = QueryEnhancer()
        
        # Test queries with different intents
        test_queries = [
            "What is the dosage of aspirin for heart attack prevention?",
            "What are the side effects of metformin?",
            "Contraindications for warfarin therapy"
        ]
        
        for query in test_queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing query: {query}")
            logger.info(f"{'='*60}")
            
            # Enhance query
            enhanced_query = query_enhancer.enhance_query(query)
            logger.info(f"Intent: {enhanced_query.intent.value} (confidence: {enhanced_query.intent_confidence:.2f})")
            logger.info(f"Entities: {enhanced_query.entities}")
            
            # Get search strategy
            search_strategy = query_enhancer.get_search_strategy(enhanced_query)
            logger.info(f"Filters: {search_strategy['filters']}")
            logger.info(f"Boost params: {search_strategy['boost_params']}")
            
            # Generate embeddings for multiple types
            embedding_types = [EmbeddingType.PUBMEDBERT, EmbeddingType.BIOBERT]
            logger.info(f"Generating embeddings for types: {[e.value for e in embedding_types]}")
            
            query_embeddings = get_embeddings_for_multiple_types(
                enhanced_query.cleaned_query,
                embedding_types
            )
            
            # Search with filters
            logger.info("Searching multiple collections with filters...")
            results = search_multiple_collections(
                query_embeddings=query_embeddings,
                metadata=None,
                intent_filters=search_strategy["boost_params"],
                entity_filters=search_strategy["filters"],
                top_k_per_collection=10
            )
            
            logger.info(f"Found {len(results)} results")
            
            # Show top 3 results
            for i, result in enumerate(results[:3]):
                logger.info(f"\nResult {i+1}:")
                logger.info(f"Score: {result['embedding_score']:.4f}")
                logger.info(f"Collection: {result['collection_name']}")
                logger.info(f"Content preview: {result['content'][:200]}...")
                if result['metadata'].get('boost_applied'):
                    logger.info(f"Boosts applied: {result['metadata']['boost_applied']}")
                
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    return True


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_multi_collection_search())
    
    if success:
        logger.info("\n✅ Test completed successfully!")
    else:
        logger.error("\n❌ Test failed!")
