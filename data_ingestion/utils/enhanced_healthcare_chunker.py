"""
Enhanced Healthcare chunker with NER-based enrichment.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from transformers import AutoTokenizer

# Import our new NER module
from shared.medical_ner import (
    get_medical_ner_model,
    MedicalEntityProcessor,
    MedicalEntityType as NEREntityType,
)

logger = logging.getLogger(__name__)

# Healthcare-specific chunk sizes
TEXT_CHUNK_SIZE = 384
TEXT_CHUNK_OVERLAP = 64
MIN_TEXT_CHUNK_SIZE = 50
HEADING_MIN_WORDS = 1
MAX_TABLE_CHUNK_SIZE = 768


class ChunkType(Enum):
    """Types of chunks in healthcare documents."""
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    MEDICATION = "medication"
    LAB_RESULT = "lab_result"
    VITAL_SIGNS = "vital_signs"
    SECTION = "section"


@dataclass
class HealthcareSection:
    """Represents a section in a healthcare document."""
    title: str
    level: int
    content: List[Dict[str, Any]]
    section_type: Optional[str] = None


# Common healthcare section patterns (simplified from original)
HEALTHCARE_SECTIONS = {
    r"chief\s+complaint|cc": "chief_complaint",
    r"medications?|meds?": "medications",
    r"allergies?": "allergies",
    r"physical\s+exam(?:ination)?|pe": "physical_exam",
    r"assessment\s+(?:and|&)?\s+plan|a&p": "assessment_plan",
    r"dosage?\s+(?:and\s+)?administration": "dosage",
    r"contraindications?": "contraindications",
    r"adverse\s+reactions?|side\s+effects?": "adverse_reactions",
}


class ChunkingError(Exception):
    """Raised when chunking fails."""
    def __init__(self, message: str, job_id: str = "unknown"):
        super().__init__(f"[job_id={job_id}] {message}")
        self.job_id = job_id


class EnhancedHealthcareChunker:
    """Healthcare chunker with NER-based enrichment."""
    
    _instance: Optional["EnhancedHealthcareChunker"] = None
    _tokenizer_instance: Optional[AutoTokenizer] = None
    _ner_model = None
    _entity_processor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize with NER components."""
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self._ner_enabled = True  # Can be toggled
            self._ner_model_name = "biomedical-ner-all"
            
    @classmethod
    def init_tokenizer_for_worker(cls, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        """Initialize tokenizer for worker process."""
        if cls._tokenizer_instance is None:
            try:
                logger.info(f"Loading medical tokenizer '{model_name}'...")
                cls._tokenizer_instance = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                logger.info("Medical tokenizer loaded.")
            except Exception as e:
                logger.error(f"Tokenizer init error: {e}", exc_info=True)
                raise
    
    def initialize_ner(self):
        """Initialize NER components lazily."""
        if self._ner_enabled and EnhancedHealthcareChunker._ner_model is None:
            try:
                logger.info("Initializing medical NER model...")
                EnhancedHealthcareChunker._ner_model = get_medical_ner_model(self._ner_model_name)
                EnhancedHealthcareChunker._entity_processor = MedicalEntityProcessor()
                logger.info("Medical NER initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NER: {e}")
                self._ner_enabled = False  # Disable on failure
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        if EnhancedHealthcareChunker._tokenizer_instance is None:
            self.init_tokenizer_for_worker()
        return EnhancedHealthcareChunker._tokenizer_instance
    
    def chunk_json(
        self,
        parsed_json: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Main entry point for chunking healthcare documents."""
        job_id = parsed_json.get("job_id", "unknown")
        original_filename = metadata.get("original_filename", "unknown")
        
        pages = parsed_json.get("pages", [])
        if not isinstance(pages, list):
            raise ChunkingError("'pages' missing or not a list", job_id)
        
        # Check if NER is requested in metadata
        self._ner_enabled = metadata.get("enable_ner", True)  # Default to True for enhanced chunker
        
        # Initialize NER if enabled
        if self._ner_enabled:
            self.initialize_ner()
        
        # Extract content and create chunks
        all_content = self._extract_structured_content(pages, job_id)
        sections = self._identify_sections(all_content)
        chunks = self._create_healthcare_chunks(sections, metadata)
        
        logger.info(f"Generated {len(chunks)} healthcare chunks for '{original_filename}'")
        return chunks
    
    def _extract_structured_content(
        self, 
        pages: List[Dict[str, Any]], 
        job_id: str
    ) -> List[Dict[str, Any]]:
        """Extract all content preserving structure."""
        content = []
        
        for page in pages:
            if not isinstance(page, dict):
                continue
                
            page_num = page.get("page", 0)
            page_width = page.get("width")
            page_height = page.get("height")
            items = page.get("items", [])
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                item_type = item.get("type")
                item_data = {
                    "type": item_type,
                    "page": page_num,
                    "page_width": page_width,
                    "page_height": page_height,
                    "bbox": item.get("bBox", {}),
                    "content": item.get("md", "") or item.get("text", ""),
                    "level": item.get("level", 0),
                    "raw": item,
                }
                
                content.append(item_data)
        
        return content
    
    def _identify_sections(self, content: List[Dict[str, Any]]) -> List[HealthcareSection]:
        """Identify healthcare document sections."""
        sections = []
        current_section = None
        section_content = []
        
        for item in content:
            if item["type"] == "heading":
                if current_section:
                    current_section.content = section_content
                    sections.append(current_section)
                
                section_type = self._classify_section(item["content"])
                current_section = HealthcareSection(
                    title=item["content"],
                    level=item.get("level", 1),
                    content=[],
                    section_type=section_type
                )
                section_content = []
            else:
                section_content.append(item)
        
        if current_section:
            current_section.content = section_content
            sections.append(current_section)
        elif section_content:
            sections.append(HealthcareSection(
                title="Document Content",
                level=0,
                content=section_content
            ))
        
        return sections
    
    def _classify_section(self, heading: str) -> Optional[str]:
        """Classify a heading into a healthcare section type."""
        heading_lower = heading.lower().strip()
        
        for pattern, section_type in HEALTHCARE_SECTIONS.items():
            if re.search(pattern, heading_lower):
                return section_type
        
        return None
    
    def _create_healthcare_chunks(
        self,
        sections: List[HealthcareSection],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create chunks respecting healthcare document structure."""
        chunks = []
        
        for section in sections:
            section_context = f"[Section: {section.title}]\n"
            section_chunks = self._chunk_section_content(
                section,
                section_context,
                metadata
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_section_content(
        self,
        section: HealthcareSection,
        section_context: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk content within a section."""
        chunks = []
        
        for item in section.content:
            if item["type"] == "text" and item["content"].strip():
                # Create text chunks
                text_chunks = self._chunk_text_with_context(
                    item["content"],
                    section_context,
                    item.get("page", 0),
                    metadata,
                    item.get("bbox"),
                    item.get("page_width"),
                    item.get("page_height"),
                    section
                )
                chunks.extend(text_chunks)
        
        return chunks
    
    def _chunk_text_with_context(
        self,
        text: str,
        section_context: str,
        page_num: int,
        metadata: Dict[str, Any],
        bbox: Optional[Dict[str, Any]] = None,
        page_width: Optional[Any] = None,
        page_height: Optional[Any] = None,
        section: Optional[HealthcareSection] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text while preserving section context."""
        if not text.strip():
            return []
        
        full_text = section_context + text
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        if len(tokens) <= TEXT_CHUNK_SIZE:
            return [self._create_enhanced_chunk(
                full_text,
                ChunkType.TEXT,
                page_num,
                metadata,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                section=section
            )]
        
        # Split into smaller chunks
        chunks = []
        context_tokens = self.tokenizer.encode(section_context, add_special_tokens=False)
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        effective_chunk_size = TEXT_CHUNK_SIZE - len(context_tokens)
        effective_overlap = TEXT_CHUNK_OVERLAP
        
        start = 0
        while start < len(text_tokens):
            end = min(start + effective_chunk_size, len(text_tokens))
            chunk_tokens = context_tokens + text_tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunks.append(self._create_enhanced_chunk(
                chunk_text,
                ChunkType.TEXT,
                page_num,
                metadata,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                section=section
            ))
            
            start = end - effective_overlap
            if start >= len(text_tokens):
                break
        
        return chunks
    
    def _create_enhanced_chunk(
        self,
        text: str,
        chunk_type: ChunkType,
        page_num: int,
        metadata: Dict[str, Any],
        bbox: Optional[Dict[str, Any]] = None,
        page_width: Optional[Any] = None,
        page_height: Optional[Any] = None,
        section: Optional[HealthcareSection] = None
    ) -> Dict[str, Any]:
        """Create a chunk enriched with NER entities."""
        # Get page dimensions
        if page_width is None:
            page_width = metadata.get("page_width", "NA")
        if page_height is None:
            page_height = metadata.get("page_height", "NA")
            
        page_width_final = page_width if isinstance(page_width, (int, float)) else "NA"
        page_height_final = page_height if isinstance(page_height, (int, float)) else "NA"
        
        # Clean text
        clean_text = text.strip()
        
        # Base chunk structure
        chunk = {
            "chunk": clean_text,
            "chunk_type": chunk_type.value,
            "page": page_num,
            "parse_type": metadata.get("parse_type", "pdf"),
            "page_width": page_width_final,
            "page_height": page_height_final,
        }
        
        if bbox:
            chunk["bbox"] = bbox
        
        if section:
            chunk["section_title"] = section.title
            chunk["section_type"] = section.section_type
        
        # Add NER entities if enabled
        if self._ner_enabled and EnhancedHealthcareChunker._ner_model is not None:
            try:
                # Extract entities using NER
                ner_result = EnhancedHealthcareChunker._ner_model.extract_entities(clean_text)
                
                # Process entities
                if EnhancedHealthcareChunker._entity_processor:
                    ner_result = EnhancedHealthcareChunker._entity_processor.process_entities(ner_result)
                
                # Extract entity summary
                entity_summary = EnhancedHealthcareChunker._entity_processor.extract_entity_summary(ner_result.entities)
                
                # Add NER metadata
                chunk["ner_entities"] = [e.to_dict() for e in ner_result.entities]
                chunk["entity_summary"] = entity_summary
                chunk["has_medical_entities"] = len(ner_result.entities) > 0
                
                # Add specific entity lists for filtering
                chunk["drugs"] = entity_summary.get(NEREntityType.DRUG.value, [])
                chunk["diseases"] = entity_summary.get(NEREntityType.DISEASE.value, [])
                chunk["procedures"] = entity_summary.get(NEREntityType.PROCEDURE.value, [])
                
                # Determine answer types based on entities
                chunk["answer_types"] = self._determine_answer_types(ner_result.entities, section)
                
            except Exception as e:
                logger.warning(f"NER extraction failed for chunk: {e}")
                # Fallback to basic metadata
                chunk["has_medical_entities"] = False
                chunk["answer_types"] = ["general"]
        else:
            # NER disabled - use basic classification
            chunk["has_medical_entities"] = False
            chunk["answer_types"] = self._classify_content_purpose(clean_text)
        
        return chunk
    
    def _determine_answer_types(self, entities, section):
        """Determine answer types based on NER entities and section."""
        answer_types = set()
        
        # Entity-based classification
        entity_types = {e.entity_type for e in entities}
        
        if NEREntityType.DRUG in entity_types:
            answer_types.add("medication_info")
        if NEREntityType.DISEASE in entity_types:
            answer_types.add("disease_info")
        if NEREntityType.PROCEDURE in entity_types:
            answer_types.add("procedure_info")
        if NEREntityType.DOSAGE in entity_types or NEREntityType.FREQUENCY in entity_types:
            answer_types.add("dosage_info")
        
        # Section-based classification
        if section and section.section_type:
            section_to_answer = {
                "contraindications": "contraindications",
                "adverse_reactions": "side_effects",
                "dosage": "dosage_info",
                "diagnosis": "diagnosis",
            }
            if section.section_type in section_to_answer:
                answer_types.add(section_to_answer[section.section_type])
        
        return list(answer_types) if answer_types else ["general"]
    
    def _classify_content_purpose(self, text: str) -> List[str]:
        """Basic content classification without NER."""
        answer_types = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["dose", "dosage", "mg", "ml"]):
            answer_types.append("dosage_info")
        if any(word in text_lower for word in ["side effect", "adverse", "reaction"]):
            answer_types.append("side_effects")
        if any(word in text_lower for word in ["contraindication", "do not use"]):
            answer_types.append("contraindications")
        
        return answer_types if answer_types else ["general"]
    
    @classmethod
    def get_instance(cls) -> "EnhancedHealthcareChunker":
        """Get singleton instance."""
        return cls()
