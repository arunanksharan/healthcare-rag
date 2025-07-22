"""
Patch to integrate BioBERT NER into query analysis.

This patch modifies the QueryEnhancer to use the actual BioBERT NER model
instead of just pattern matching for entity extraction during query processing.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def apply_ner_integration_patch():
    """Apply the NER integration patch to query_enhancer.py"""
    
    # Path to the file to patch
    query_enhancer_path = Path(__file__).parent.parent / "shared" / "query_analysis" / "query_enhancer.py"
    
    if not query_enhancer_path.exists():
        logger.error(f"Cannot find query_enhancer.py at {query_enhancer_path}")
        return False
    
    # Read the current content
    with open(query_enhancer_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "MedicalQueryAnalyzerWithNER" in content:
        logger.info("Query enhancer already patched for NER integration")
        return True
    
    # Create the patch
    original_import = "from .medical_analyzer import MedicalQueryAnalyzer, QueryAnalysisResult"
    patched_import = """from .medical_analyzer import MedicalQueryAnalyzer, QueryAnalysisResult
try:
    from .medical_analyzer_with_ner import MedicalQueryAnalyzerWithNER
    USE_NER = True
except ImportError:
    logger.warning("NER-enabled analyzer not available, using pattern matching")
    USE_NER = False"""
    
    original_init = "self.medical_analyzer = MedicalQueryAnalyzer()"
    patched_init = """if USE_NER:
            self.medical_analyzer = MedicalQueryAnalyzerWithNER()
            logger.info("Query enhancer using BioBERT NER for entity extraction")
        else:
            self.medical_analyzer = MedicalQueryAnalyzer()
            logger.info("Query enhancer using pattern matching for entity extraction")"""
    
    # Apply the patches
    content = content.replace(original_import, patched_import)
    content = content.replace(original_init, patched_init)
    
    # Write back
    with open(query_enhancer_path, 'w') as f:
        f.write(content)
    
    logger.info("Successfully patched query enhancer for NER integration")
    return True


def verify_ner_integration():
    """Verify that NER is properly integrated."""
    try:
        from shared.query_analysis import QueryEnhancer
        from shared.query_analysis.medical_analyzer_with_ner import MedicalQueryAnalyzerWithNER
        
        enhancer = QueryEnhancer()
        
        # Check if the medical analyzer is the NER version
        if isinstance(enhancer.medical_analyzer, MedicalQueryAnalyzerWithNER):
            logger.info("✓ NER integration verified - QueryEnhancer is using BioBERT NER")
            return True
        else:
            logger.warning("✗ NER integration failed - QueryEnhancer is using pattern matching")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying NER integration: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Apply the patch
    if apply_ner_integration_patch():
        # Verify it worked
        verify_ner_integration()
