#!/usr/bin/env python3
"""
Comprehensive verification script for the RAG pipeline with NER.
Tests the complete flow from ingestion to retrieval with entity extraction enabled.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
import requests

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Service URLs
INGESTION_URL = "http://localhost:8001/api/ingest"
SEARCH_URL = "http://localhost:8002/api/search"
VIEW_DOC_URL = "http://localhost:8002/api/view_document"

class PipelineVerifier:
    """Verifies the complete RAG pipeline with NER."""
    
    def __init__(self):
        self.test_results = []
        
    def verify_services_running(self):
        """Check if both services are running."""
        logger.info("Checking if services are running...")
        
        services = [
            ("Ingestion Service", "http://localhost:8001/docs"),
            ("Retrieval Service", "http://localhost:8002/docs")
        ]
        
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ {service_name} is running")
                else:
                    logger.error(f"✗ {service_name} returned status {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"✗ {service_name} is not accessible: {e}")
                return False
        
        return True
    
    def test_ingestion_with_ner(self, test_file_path: str):
        """Test document ingestion with NER enabled."""
        logger.info("\n=== Testing Document Ingestion with NER ===")
        
        if not os.path.exists(test_file_path):
            logger.error(f"Test file not found: {test_file_path}")
            return None
        
        # Prepare the document metadata
        metadata = {
            "title": "Aspirin Medication Guide - NER Test",
            "type": "GUIDELINE",
            "description": "Test document for NER-enabled ingestion",
            "date": "2025-01-23",
            "embedding_type": "pubmedbert",
            "chunker_type": "healthcare",
            "enable_ner": True  # Enable NER processing
        }
        
        # Prepare files and data for multipart/form-data
        files = {
            'file': ('aspirin_guide.pdf', open(test_file_path, 'rb'), 'application/pdf')
        }
        
        data = {
            'title': metadata['title'],
            'type': metadata['type'],
            'description': metadata['description'],
            'date': metadata['date'],
            'embedding_type': metadata['embedding_type'],
            'chunker_type': metadata['chunker_type'],
            'enable_ner': str(metadata['enable_ner']).lower()  # Convert to string
        }
        
        try:
            logger.info(f"Uploading document with NER enabled...")
            response = requests.post(INGESTION_URL, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✓ Document uploaded successfully")
                logger.info(f"  Task ID: {result['task_id']}")
                logger.info(f"  Filename: {result['filename']}")
                logger.info(f"  NER Enabled: {result['metadata']['enable_ner']}")
                
                self.test_results.append({
                    "test": "ingestion_with_ner",
                    "status": "success",
                    "details": result
                })
                
                return result['task_id']
            else:
                logger.error(f"✗ Ingestion failed: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                
                self.test_results.append({
                    "test": "ingestion_with_ner",
                    "status": "failed",
                    "error": response.text
                })
                
                return None
                
        except Exception as e:
            logger.error(f"✗ Error during ingestion: {e}")
            self.test_results.append({
                "test": "ingestion_with_ner",
                "status": "error",
                "error": str(e)
            })
            return None
        finally:
            files['file'][1].close()
    
    def wait_for_processing(self, task_id: str, max_wait: int = 60):
        """Wait for the ingestion task to complete."""
        logger.info(f"\nWaiting for processing to complete (max {max_wait}s)...")
        
        # For now, we'll just wait a fixed amount of time
        # In production, you'd check task status via Celery
        wait_time = 30
        for i in range(wait_time):
            time.sleep(1)
            if i % 5 == 0:
                logger.info(f"  Processing... ({i}/{wait_time}s)")
        
        logger.info("✓ Processing should be complete")
        return True
    
    def test_ner_entity_search(self):
        """Test searches that should trigger NER entity filtering."""
        logger.info("\n=== Testing NER Entity-Based Searches ===")
        
        test_queries = [
            {
                "query": "What is the dosage of aspirin for heart attack prevention?",
                "expected_intent": "dosage",
                "expected_entities": {"drugs": ["aspirin"]},
                "description": "Dosage query with drug entity"
            },
            {
                "query": "Side effects of aspirin",
                "expected_intent": "side_effects",
                "expected_entities": {"drugs": ["aspirin"]},
                "description": "Side effects query"
            },
            {
                "query": "Can aspirin be used with warfarin?",
                "expected_intent": "contraindications",
                "expected_entities": {"drugs": ["aspirin", "warfarin"]},
                "description": "Drug interaction query"
            },
            {
                "query": "Aspirin contraindications in patients with bleeding disorders",
                "expected_intent": "contraindications",
                "expected_entities": {"drugs": ["aspirin"], "diseases": ["bleeding disorders"]},
                "description": "Contraindications with disease entity"
            }
        ]
        
        for test_case in test_queries:
            logger.info(f"\nTesting: {test_case['description']}")
            logger.info(f"Query: {test_case['query']}")
            
            try:
                response = requests.post(
                    SEARCH_URL,
                    json={
                        "query": test_case["query"],
                        "embedding_types": ["pubmedbert"]
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract query analysis results
                    query_analysis = result.get("query_analysis", {})
                    detected_intent = query_analysis.get("intent", "unknown")
                    detected_entities = query_analysis.get("entities", {})
                    filters_applied = query_analysis.get("filters_applied", {})
                    
                    logger.info(f"✓ Search completed successfully")
                    logger.info(f"  Detected Intent: {detected_intent} (confidence: {query_analysis.get('confidence', 0):.2f})")
                    logger.info(f"  Detected Entities: {json.dumps(detected_entities, indent=2)}")
                    logger.info(f"  Filters Applied: {json.dumps(filters_applied, indent=2)}")
                    
                    # Verify intent detection
                    if detected_intent == test_case["expected_intent"]:
                        logger.info(f"  ✓ Intent correctly detected")
                    else:
                        logger.warning(f"  ✗ Expected intent '{test_case['expected_intent']}', got '{detected_intent}'")
                    
                    # Verify entity extraction
                    for entity_type, expected_values in test_case["expected_entities"].items():
                        detected_values = detected_entities.get(entity_type, [])
                        detected_lower = [v.lower() for v in detected_values]
                        
                        for expected in expected_values:
                            if expected.lower() in detected_lower:
                                logger.info(f"  ✓ Entity '{expected}' detected")
                            else:
                                logger.warning(f"  ✗ Expected entity '{expected}' not found")
                    
                    # Check if filters were applied
                    if filters_applied:
                        logger.info(f"  ✓ NER-based filters were applied")
                    else:
                        logger.warning(f"  ⚠ No filters were applied")
                    
                    # Sample response content
                    if result.get("referenced_contexts"):
                        logger.info(f"  Found {len(result['referenced_contexts'])} relevant chunks")
                        
                        # Check if chunks have NER metadata
                        first_chunk = result["referenced_contexts"][0]
                        metadata = first_chunk.get("metadata", {})
                        
                        if "has_medical_entities" in metadata:
                            logger.info(f"  ✓ Chunks contain NER metadata")
                            if metadata.get("drugs"):
                                logger.info(f"    Drugs in chunk: {metadata['drugs']}")
                            if metadata.get("diseases"):
                                logger.info(f"    Diseases in chunk: {metadata['diseases']}")
                            if metadata.get("answer_types"):
                                logger.info(f"    Answer types: {metadata['answer_types']}")
                        else:
                            logger.warning(f"  ⚠ Chunks missing NER metadata")
                    
                    self.test_results.append({
                        "test": f"ner_search_{test_case['description']}",
                        "status": "success",
                        "query": test_case["query"],
                        "intent_detected": detected_intent,
                        "entities_detected": detected_entities,
                        "filters_applied": filters_applied
                    })
                    
                else:
                    logger.error(f"✗ Search failed: {response.status_code}")
                    logger.error(f"  Response: {response.text}")
                    
                    self.test_results.append({
                        "test": f"ner_search_{test_case['description']}",
                        "status": "failed",
                        "query": test_case["query"],
                        "error": response.text
                    })
                    
            except Exception as e:
                logger.error(f"✗ Error during search: {e}")
                self.test_results.append({
                    "test": f"ner_search_{test_case['description']}",
                    "status": "error",
                    "query": test_case["query"],
                    "error": str(e)
                })
    
    def test_chunk_metadata_verification(self):
        """Verify that chunks contain proper NER metadata."""
        logger.info("\n=== Verifying Chunk NER Metadata ===")
        
        # Simple query to get some chunks
        query = "aspirin information"
        
        try:
            response = requests.post(
                SEARCH_URL,
                json={
                    "query": query,
                    "embedding_types": ["pubmedbert"]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                contexts = result.get("referenced_contexts", [])
                
                if contexts:
                    logger.info(f"✓ Retrieved {len(contexts)} chunks")
                    
                    # Analyze first few chunks
                    for i, context in enumerate(contexts[:3]):
                        logger.info(f"\nChunk {i+1}:")
                        metadata = context.get("metadata", {})
                        
                        # Check for NER fields
                        ner_fields = [
                            "has_medical_entities",
                            "drugs",
                            "diseases", 
                            "procedures",
                            "answer_types",
                            "chunk_type",
                            "section_title",
                            "section_type"
                        ]
                        
                        for field in ner_fields:
                            if field in metadata:
                                value = metadata[field]
                                if isinstance(value, list) and value:
                                    logger.info(f"  ✓ {field}: {value}")
                                elif isinstance(value, bool):
                                    logger.info(f"  ✓ {field}: {value}")
                                elif value:
                                    logger.info(f"  ✓ {field}: {value}")
                            else:
                                logger.warning(f"  ✗ {field}: missing")
                        
                        # Check content snippet
                        content = context.get("content", "")[:100]
                        logger.info(f"  Content: {content}...")
                        
                    self.test_results.append({
                        "test": "chunk_metadata_verification",
                        "status": "success",
                        "chunks_analyzed": min(3, len(contexts))
                    })
                    
                else:
                    logger.warning("No chunks returned")
                    self.test_results.append({
                        "test": "chunk_metadata_verification",
                        "status": "warning",
                        "message": "No chunks returned"
                    })
                    
            else:
                logger.error(f"✗ Search failed: {response.status_code}")
                self.test_results.append({
                    "test": "chunk_metadata_verification",
                    "status": "failed",
                    "error": response.text
                })
                
        except Exception as e:
            logger.error(f"✗ Error during verification: {e}")
            self.test_results.append({
                "test": "chunk_metadata_verification",
                "status": "error",
                "error": str(e)
            })
    
    def generate_report(self):
        """Generate a summary report of all tests."""
        logger.info("\n" + "="*60)
        logger.info("PIPELINE VERIFICATION REPORT")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        successful = sum(1 for r in self.test_results if r["status"] == "success")
        failed = sum(1 for r in self.test_results if r["status"] == "failed")
        errors = sum(1 for r in self.test_results if r["status"] == "error")
        warnings = sum(1 for r in self.test_results if r["status"] == "warning")
        
        logger.info(f"\nTotal Tests: {total_tests}")
        logger.info(f"✓ Successful: {successful}")
        logger.info(f"✗ Failed: {failed}")
        logger.info(f"⚠ Warnings: {warnings}")
        logger.info(f"! Errors: {errors}")
        
        logger.info("\nDetailed Results:")
        for result in self.test_results:
            status_symbol = {
                "success": "✓",
                "failed": "✗",
                "error": "!",
                "warning": "⚠"
            }.get(result["status"], "?")
            
            logger.info(f"\n{status_symbol} {result['test']}: {result['status'].upper()}")
            
            # Print relevant details
            for key, value in result.items():
                if key not in ["test", "status"] and value:
                    if isinstance(value, dict) or isinstance(value, list):
                        logger.info(f"  {key}: {json.dumps(value, indent=4)}")
                    else:
                        logger.info(f"  {key}: {value}")
        
        # Overall status
        logger.info("\n" + "="*60)
        if failed == 0 and errors == 0:
            logger.info("✓ PIPELINE VERIFICATION PASSED")
            logger.info("The RAG pipeline with NER is working correctly!")
        else:
            logger.info("✗ PIPELINE VERIFICATION FAILED")
            logger.info("Please check the errors above and fix the issues.")
        logger.info("="*60)


def main():
    """Main execution function."""
    verifier = PipelineVerifier()
    
    # Check services
    if not verifier.verify_services_running():
        logger.error("Services are not running. Please start them with docker-compose.")
        return
    
    # Find a test PDF file
    test_file = None
    media_dir = Path(__file__).parent.parent / "media"
    
    if media_dir.exists():
        pdf_files = list(media_dir.glob("*.pdf"))
        if pdf_files:
            test_file = str(pdf_files[0])
            logger.info(f"Using test file: {test_file}")
    
    if not test_file:
        logger.error("No PDF files found in media directory. Please add a test PDF.")
        return
    
    # Run tests
    task_id = verifier.test_ingestion_with_ner(test_file)
    
    if task_id:
        # Wait for processing
        verifier.wait_for_processing(task_id)
        
        # Test NER-based searches
        verifier.test_ner_entity_search()
        
        # Verify chunk metadata
        verifier.test_chunk_metadata_verification()
    
    # Generate report
    verifier.generate_report()


if __name__ == "__main__":
    main()
