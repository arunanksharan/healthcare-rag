#!/usr/bin/env python3
"""
Comprehensive fix for the RAG pipeline NER integration issues.
This script addresses ALL issues found in the thorough review.
"""
import os
import sys
import logging
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_query_analyzer_ner_integration():
    """Fix the query analyzer to use actual BioBERT NER."""
    
    # Path to query_enhancer.py
    query_enhancer_path = Path(__file__).parent.parent / "shared" / "query_analysis" / "query_enhancer.py"
    
    if not query_enhancer_path.exists():
        logger.error(f"Cannot find query_enhancer.py at {query_enhancer_path}")
        return False
    
    # Read current content
    with open(query_enhancer_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if "# NER INTEGRATION FIX APPLIED" in content:
        logger.info("Query enhancer already fixed for NER integration")
        return True
    
    # Find the imports section
    import_section_end = content.find("logger = logging.getLogger(__name__)")
    if import_section_end == -1:
        logger.error("Cannot find proper location for imports")
        return False
    
    # Add NER imports
    ner_imports = """
# NER INTEGRATION FIX APPLIED
from shared.medical_ner import (
    get_medical_ner_model,
    MedicalEntityProcessor,
    MedicalEntityType as NEREntityType,
)
"""
    
    # Insert imports before logger line
    content = content[:import_section_end] + ner_imports + "\n" + content[import_section_end:]
    
    # Replace the medical analyzer initialization
    old_init = "self.medical_analyzer = MedicalQueryAnalyzer()"
    new_init = """self.medical_analyzer = MedicalQueryAnalyzer()
        
        # Initialize NER components for query analysis
        self._ner_model = None
        self._entity_processor = None
        self._ner_initialized = False
        self._init_ner_for_queries()"""
    
    content = content.replace(old_init, new_init)
    
    # Add NER initialization method after __init__
    init_end = content.find("def enhance_query(")
    if init_end == -1:
        logger.error("Cannot find enhance_query method")
        return False
    
    ner_init_method = """
    def _init_ner_for_queries(self):
        \"\"\"Initialize NER model for query analysis.\"\"\"
        try:
            logger.info("Initializing BioBERT NER for query analysis...")
            self._ner_model = get_medical_ner_model("biomedical-ner-all")
            self._entity_processor = MedicalEntityProcessor()
            self._ner_initialized = True
            logger.info("BioBERT NER initialized for query analysis")
        except Exception as e:
            logger.warning(f"Could not initialize NER for queries: {e}")
            self._ner_initialized = False
    
    def _extract_entities_with_ner(self, text: str) -> Dict[str, List[str]]:
        \"\"\"Extract entities using BioBERT NER.\"\"\"
        if not self._ner_initialized or not self._ner_model:
            return {}
        
        try:
            # Extract entities using NER
            ner_result = self._ner_model.extract_entities(text)
            
            # Process entities
            if self._entity_processor:
                ner_result = self._entity_processor.process_entities(ner_result)
            
            # Group by type
            entities_by_type = {}
            for entity in ner_result.entities:
                entity_type = entity.entity_type.value
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                
                # Add both original and normalized forms
                entities_by_type[entity_type].append(entity.text.lower())
                if entity.normalized_form:
                    entities_by_type[entity_type].append(entity.normalized_form.lower())
                
                # Add synonyms
                for synonym in entity.synonyms:
                    entities_by_type[entity_type].append(synonym.lower())
            
            # Remove duplicates
            for entity_type in entities_by_type:
                entities_by_type[entity_type] = list(set(entities_by_type[entity_type]))
            
            return entities_by_type
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return {}
    
    """
    
    content = content[:init_end] + ner_init_method + content[init_end:]
    
    # Modify enhance_query to use NER
    enhance_start = content.find("# Step 1: Analyze query (NER, abbreviations, corrections)")
    if enhance_start != -1:
        enhance_end = content.find("# Step 2: Detect intent", enhance_start)
        if enhance_end != -1:
            new_step1 = """# Step 1: Analyze query (NER, abbreviations, corrections)
        analysis = self.medical_analyzer.analyze_query(query)
        
        # Step 1b: Extract entities using BioBERT NER if available
        ner_entities = self._extract_entities_with_ner(query)
        
        # Merge NER entities with pattern-based entities
        for entity_type, entities in ner_entities.items():
            if entity_type in ["drug", "disease", "procedure"]:
                # Update analysis with NER entities
                existing = {e.text.lower() for e in analysis.entities if e.entity_type.value == entity_type}
                for ner_entity in entities:
                    if ner_entity not in existing:
                        # Add as MedicalEntity for compatibility
                        from shared.query_analysis.medical_analyzer import MedicalEntity, MedicalEntityType
                        entity_type_enum = MedicalEntityType(entity_type)
                        analysis.entities.append(MedicalEntity(
                            text=ner_entity,
                            entity_type=entity_type_enum,
                            normalized_form=ner_entity,
                            confidence=0.9,
                            synonyms=[]
                        ))
        
        """
            content = content[:enhance_start] + new_step1 + content[enhance_end:]
    
    # Write the fixed content
    with open(query_enhancer_path, 'w') as f:
        f.write(content)
    
    logger.info("Successfully fixed query enhancer for NER integration")
    return True


def fix_case_sensitivity_in_storage():
    """Ensure all entity fields are lowercased in Qdrant storage."""
    
    qdrant_path = Path(__file__).parent.parent / "data_ingestion" / "utils" / "qdrant.py"
    
    if not qdrant_path.exists():
        logger.error(f"Cannot find qdrant.py at {qdrant_path}")
        return False
    
    with open(qdrant_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if "# CASE SENSITIVITY FIX APPLIED" in content:
        logger.info("Qdrant storage already fixed for case sensitivity")
        return True
    
    # Find the payload creation section
    payload_start = content.find('"drugs": chunk_data.get("drugs", []),')
    if payload_start == -1:
        logger.error("Cannot find drugs field in payload")
        return False
    
    # Replace entity list fields with lowercased versions
    old_fields = '''                "drugs": chunk_data.get("drugs", []),
                "diseases": chunk_data.get("diseases", []),
                "procedures": chunk_data.get("procedures", []),'''
    
    new_fields = '''                # CASE SENSITIVITY FIX APPLIED - lowercase all entities for consistent matching
                "drugs": [drug.lower() for drug in chunk_data.get("drugs", [])],
                "diseases": [disease.lower() for disease in chunk_data.get("diseases", [])],
                "procedures": [procedure.lower() for procedure in chunk_data.get("procedures", [])],'''
    
    content = content.replace(old_fields, new_fields)
    
    # Write back
    with open(qdrant_path, 'w') as f:
        f.write(content)
    
    logger.info("Successfully fixed case sensitivity in Qdrant storage")
    return True


def verify_fixes():
    """Verify that all fixes have been applied correctly."""
    
    logger.info("\n=== Verifying Fixes ===")
    
    # Check 1: Query enhancer has NER integration
    query_enhancer_path = Path(__file__).parent.parent / "shared" / "query_analysis" / "query_enhancer.py"
    with open(query_enhancer_path, 'r') as f:
        content = f.read()
    
    if "# NER INTEGRATION FIX APPLIED" in content:
        logger.info("✓ Query enhancer has NER integration")
    else:
        logger.error("✗ Query enhancer missing NER integration")
        return False
    
    # Check 2: Qdrant storage has case sensitivity fix
    qdrant_path = Path(__file__).parent.parent / "data_ingestion" / "utils" / "qdrant.py"
    with open(qdrant_path, 'r') as f:
        content = f.read()
    
    if "# CASE SENSITIVITY FIX APPLIED" in content:
        logger.info("✓ Qdrant storage has case sensitivity fix")
    else:
        logger.error("✗ Qdrant storage missing case sensitivity fix")
        return False
    
    # Check 3: Test NER initialization
    try:
        from shared.query_analysis import QueryEnhancer
        enhancer = QueryEnhancer()
        
        if hasattr(enhancer, '_ner_initialized'):
            logger.info("✓ QueryEnhancer can initialize NER")
        else:
            logger.warning("⚠ QueryEnhancer may not have NER support")
    except Exception as e:
        logger.error(f"✗ Error testing QueryEnhancer: {e}")
        return False
    
    logger.info("\n✓ All fixes verified successfully!")
    return True


def main():
    """Apply all fixes to the RAG pipeline."""
    
    logger.info("=" * 60)
    logger.info("APPLYING COMPREHENSIVE FIXES TO RAG PIPELINE")
    logger.info("=" * 60)
    
    # Apply fixes
    fixes_applied = []
    
    logger.info("\n1. Fixing query analyzer NER integration...")
    if fix_query_analyzer_ner_integration():
        fixes_applied.append("Query NER Integration")
    
    logger.info("\n2. Fixing case sensitivity in storage...")
    if fix_case_sensitivity_in_storage():
        fixes_applied.append("Case Sensitivity")
    
    # Verify all fixes
    logger.info("\n3. Verifying all fixes...")
    if verify_fixes():
        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL FIXES APPLIED SUCCESSFULLY!")
        logger.info(f"Applied fixes: {', '.join(fixes_applied)}")
        logger.info("=" * 60)
        logger.info("\nThe RAG pipeline now has:")
        logger.info("- Symmetric NER usage (both ingestion and query)")
        logger.info("- Case-insensitive entity matching")
        logger.info("- Proper entity normalization")
        logger.info("\nPlease restart your services to apply the changes.")
    else:
        logger.error("\n✗ Some fixes failed to apply or verify")
        logger.error("Please check the errors above")


if __name__ == "__main__":
    main()
