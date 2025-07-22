"""
BioBERT-based NER implementation.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline
)

from .base import MedicalEntity, MedicalEntityType, NERResult, NERConfig

logger = logging.getLogger(__name__)


class BioBERTNER:
    """
    BioBERT-based Named Entity Recognition for medical text.
    
    Uses HuggingFace transformers for efficient inference.
    """
    
    # Recommended models for medical NER
    RECOMMENDED_MODELS = {
        "emilyalsentzer/Bio_ClinicalBERT": {
            "description": "Clinical notes focused",
            "entity_types": ["PROBLEM", "TEST", "TREATMENT"],
        },
        "alvaroalon2/biobert_diseases_ner": {
            "description": "Disease-focused NER",
            "entity_types": ["DISEASE"],
        },
        "samrawal/bert-large-uncased_med-ner": {
            "description": "Comprehensive medical NER",
            "entity_types": ["DRUG", "DISEASE", "SYMPTOM"],
        },
        "d4data/biomedical-ner-all": {
            "description": "All medical entity types",
            "entity_types": ["DRUG", "DISEASE", "GENE", "CHEMICAL", "ANATOMY"],
        },
    }
    
    def __init__(self, config: NERConfig):
        """Initialize BioBERT NER model."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Load model and tokenizer."""
        if self._initialized:
            return
            
        try:
            logger.info(f"Loading BioBERT NER model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path or self.config.model_name,
                cache_dir=self.config.cache_dir,
            )
            
            # Load model
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_path or self.config.model_name,
                cache_dir=self.config.cache_dir,
            )
            
            # Move to device if CUDA available
            if self.config.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("BioBERT NER model moved to CUDA")
            
            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.config.device == "cuda" and torch.cuda.is_available() else -1,
                aggregation_strategy="simple" if self.config.aggregate_subwords else "none",
            )
            
            self._initialized = True
            logger.info(f"BioBERT NER model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BioBERT NER: {e}")
            raise
    
    def extract_entities(self, text: str) -> NERResult:
        """
        Extract medical entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            NERResult containing extracted entities
        """
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Handle long texts by chunking
            if len(text) > self.config.max_length * 4:  # Rough estimate
                entities = self._extract_from_long_text(text)
            else:
                entities = self._extract_from_text(text)
            
            # Filter by confidence
            entities = [
                e for e in entities 
                if e.confidence >= self.config.confidence_threshold
            ]
            
            # Sort by position
            entities.sort(key=lambda x: x.start_char)
            
            processing_time = (time.time() - start_time) * 1000
            
            return NERResult(
                text=text,
                entities=entities,
                processing_time_ms=processing_time,
                model_name=self.config.model_name,
                model_version=self.config.metadata.get("version"),
            )
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            # Return empty result on error
            return NERResult(
                text=text,
                entities=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                model_name=self.config.model_name,
            )
    
    def _extract_from_text(self, text: str) -> List[MedicalEntity]:
        """Extract entities from a single text segment."""
        # Get predictions from pipeline
        predictions = self.pipeline(text)
        
        entities = []
        for pred in predictions:
            # Handle both possible keys from HuggingFace pipeline
            entity_label = pred.get("entity_group") or pred.get("entity", "UNKNOWN")
            
            entity = MedicalEntity(
                text=pred["word"],
                entity_type=self._map_label_to_type(entity_label),
                start_char=pred["start"],
                end_char=pred["end"],
                confidence=pred["score"],
                metadata={
                    "original_label": entity_label,
                }
            )
            entities.append(entity)
        
        return entities
    
    def _extract_from_long_text(self, text: str) -> List[MedicalEntity]:
        """Extract entities from long text by chunking."""
        entities = []
        
        # Simple sentence-based chunking
        # In production, use better sentence segmentation
        chunks = self._chunk_text(text)
        
        offset = 0
        for chunk in chunks:
            chunk_entities = self._extract_from_text(chunk)
            
            # Adjust positions
            for entity in chunk_entities:
                entity.start_char += offset
                entity.end_char += offset
            
            entities.extend(chunk_entities)
            offset += len(chunk) + 1  # +1 for space/newline
        
        return entities
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into processable segments."""
        # Simple chunking by sentences
        # In production, use spaCy or NLTK for better sentence segmentation
        sentences = text.split('. ')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.config.max_length * 3:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _map_label_to_type(self, label: str) -> MedicalEntityType:
        """Map model's label to our entity type system."""
        return MedicalEntityType.from_bio_tag(label)
    
    def batch_extract_entities(self, texts: List[str]) -> List[NERResult]:
        """
        Extract entities from multiple texts efficiently.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of NERResult objects
        """
        if not self._initialized:
            self.initialize()
        
        results = []
        
        # Process in batches for efficiency
        batch_size = self.config.batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process each text in batch
            # Note: HuggingFace pipeline handles batching internally
            for text in batch:
                result = self.extract_entities(text)
                results.append(result)
        
        return results
    
    @classmethod
    def get_recommended_config(
        cls, 
        use_case: str = "general",
        device: str = "cpu"
    ) -> NERConfig:
        """Get recommended configuration for common use cases."""
        configs = {
            "general": NERConfig(
                model_name="d4data/biomedical-ner-all",
                model_path="d4data/biomedical-ner-all",
                confidence_threshold=0.7,
                device=device,
            ),
            "clinical": NERConfig(
                model_name="emilyalsentzer/Bio_ClinicalBERT",
                model_path="emilyalsentzer/Bio_ClinicalBERT",
                confidence_threshold=0.75,
                device=device,
            ),
            "diseases": NERConfig(
                model_name="alvaroalon2/biobert_diseases_ner",
                model_path="alvaroalon2/biobert_diseases_ner",
                confidence_threshold=0.8,
                device=device,
            ),
        }
        
        return configs.get(use_case, configs["general"])
