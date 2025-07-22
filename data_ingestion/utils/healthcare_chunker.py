"""
Healthcare-specific document chunking strategy.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Healthcare-specific chunk sizes
TEXT_CHUNK_SIZE = 384  # Smaller for precision in medical context
TEXT_CHUNK_OVERLAP = 64  # Larger overlap for medical continuity
MIN_TEXT_CHUNK_SIZE = 50  # Smaller minimum for short medical notes
HEADING_MIN_WORDS = 1  # Medical headings can be single words
MAX_TABLE_CHUNK_SIZE = 768  # Larger limit for complex medical tables


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


# Common healthcare section patterns
HEALTHCARE_SECTIONS = {
    # Clinical sections
    r"chief\s+complaint|cc": "chief_complaint",
    r"history\s+of\s+present\s+illness|hpi": "hpi",
    r"past\s+medical\s+history|pmh": "pmh",
    r"medications?|meds?": "medications",
    r"allergies?": "allergies",
    r"review\s+of\s+systems|ros": "ros",
    r"physical\s+exam(?:ination)?|pe": "physical_exam",
    r"vital\s+signs?|vitals?": "vital_signs",
    r"lab(?:oratory)?\s+(?:results?|findings?|data)": "lab_results",
    r"imaging|radiology|x-?ray|ct|mri": "imaging",
    r"assessment\s+(?:and|&)?\s+plan|a&p|a/p": "assessment_plan",
    r"diagnosis|diagnoses|dx": "diagnosis",
    r"impression": "impression",
    r"plan|treatment\s+plan": "treatment_plan",
    r"discharge\s+(?:summary|instructions?)": "discharge",
    r"follow[\s-]?up": "followup",
    
    # Drug information sections
    r"indication": "indication",
    r"dosage?\s+(?:and\s+)?administration": "dosage",
    r"contraindications?": "contraindications",
    r"warnings?\s+(?:and\s+)?precautions?": "warnings",
    r"adverse\s+reactions?|side\s+effects?": "adverse_reactions",
    r"drug\s+interactions?": "drug_interactions",
    
    # Guidelines and protocols
    r"guidelines?": "guideline",
    r"protocols?": "protocol",
    r"recommendations?": "recommendations",
    r"inclusion\s+criteria": "inclusion_criteria",
    r"exclusion\s+criteria": "exclusion_criteria",
}

# Patterns that indicate content should stay together
KEEP_TOGETHER_PATTERNS = [
    # Medication entries
    r"^\s*(?:•|\-|\d+\.)\s*\w+.*?(?:mg|mcg|g|ml|unit).*?(?:daily|bid|tid|qid|prn)",
    # Lab results with values
    r"^\s*\w+.*?:\s*\d+\.?\d*\s*(?:\w+/?)+(?:\s*\[.*?\])?",
    # Vital signs
    r"^\s*(?:BP|HR|RR|Temp|O2\s*Sat|SpO2).*?:\s*\d+",
]


class ChunkingError(Exception):
    """Raised when chunking fails."""
    def __init__(self, message: str, job_id: str = "unknown"):
        super().__init__(f"[job_id={job_id}] {message}")
        self.job_id = job_id


class HealthcareChunker:
    """Healthcare-specific document chunker."""
    
    _instance: Optional["HealthcareChunker"] = None
    _tokenizer_instance: Optional[AutoTokenizer] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def init_tokenizer_for_worker(cls, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        """Initialize with medical tokenizer."""
        if cls._tokenizer_instance is None:
            try:
                logger.info(f"Loading medical tokenizer '{model_name}'...")
                cls._tokenizer_instance = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                logger.info("Medical tokenizer loaded.")
            except Exception as e:
                logger.error(f"Tokenizer init error: {e}", exc_info=True)
                raise
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        if HealthcareChunker._tokenizer_instance is None:
            self.init_tokenizer_for_worker()
        return HealthcareChunker._tokenizer_instance
    
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
        
        # First pass: Extract all content with structure
        all_content = self._extract_structured_content(pages, job_id)
        
        # Second pass: Identify sections and hierarchy
        sections = self._identify_sections(all_content)
        
        # Third pass: Create chunks respecting sections
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
            
            # Handle images at page level
            for image in page.get("images", []):
                if isinstance(image, dict):
                    image_data = {
                        "type": "image",
                        "page": page_num,
                        "page_width": page_width,
                        "page_height": page_height,
                        "bbox": {
                            "x": image.get("x"),
                            "y": image.get("y"),
                            "w": image.get("width"),
                            "h": image.get("height"),
                        },
                        "content": self._extract_image_text(image),
                        "raw": image,
                    }
                    if image_data["content"]:
                        content.append(image_data)
            
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
                    "level": item.get("level", 0),  # For headings
                    "raw": item,
                }
                
                # Special handling for tables
                if item_type == "table":
                    item_data["table_data"] = self._extract_table_structure(item)
                
                content.append(item_data)
        
        return content
    
    def _extract_image_text(self, image: Dict[str, Any]) -> str:
        """Extract text from image OCR data."""
        ocr = image.get("ocr", [])
        if not isinstance(ocr, list):
            return ""
        
        texts = []
        for ocr_item in ocr:
            if isinstance(ocr_item, dict) and "text" in ocr_item:
                text = ocr_item["text"].strip()
                if text:
                    texts.append(text)
        
        return " ".join(texts)
    
    def _extract_table_structure(self, table_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from tables."""
        rows = table_item.get("rows", [])
        if not rows:
            return {"type": "text", "content": table_item.get("md", "")}
        
        # Try to identify table type (lab results, medications, etc.)
        table_type = self._identify_table_type(rows)
        
        return {
            "type": table_type,
            "rows": rows,
            "headers": rows[0] if rows else [],
            "data": rows[1:] if len(rows) > 1 else [],
        }
    
    def _identify_table_type(self, rows: List[List[str]]) -> str:
        """Identify the type of medical table."""
        if not rows:
            return "generic"
        
        # Check headers (if present)
        headers = " ".join(str(cell).lower() for cell in rows[0]) if rows else ""
        
        if any(term in headers for term in ["test", "result", "value", "range", "unit"]):
            return "lab_results"
        elif any(term in headers for term in ["medication", "drug", "dose", "frequency", "route"]):
            return "medications"
        elif any(term in headers for term in ["vital", "bp", "hr", "temp", "spo2"]):
            return "vital_signs"
        
        return "generic"
    
    def _identify_sections(self, content: List[Dict[str, Any]]) -> List[HealthcareSection]:
        """Identify healthcare document sections."""
        sections = []
        current_section = None
        section_content = []
        
        for item in content:
            # Check if this is a section heading
            if item["type"] == "heading":
                # Save previous section
                if current_section:
                    current_section.content = section_content
                    sections.append(current_section)
                
                # Start new section
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
        
        # Don't forget the last section
        if current_section:
            current_section.content = section_content
            sections.append(current_section)
        elif section_content:
            # No sections found, treat as single section
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
            # Add section heading as context
            section_context = f"[Section: {section.title}]\n"
            
            # Process section content
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
        
        # Group related content
        content_groups = self._group_related_content(section.content)
        
        for group in content_groups:
            # Extract page dimensions from first item in group
            page_width = None
            page_height = None
            if "items" in group and group["items"]:
                page_width = group["items"][0].get("page_width")
                page_height = group["items"][0].get("page_height")
            elif "content" in group and isinstance(group["content"], dict):
                page_width = group["content"].get("page_width")
                page_height = group["content"].get("page_height")
            elif "page_width" in group:
                page_width = group["page_width"]
                page_height = group.get("page_height")
                
            if group["type"] == "table":
                # Handle tables specially
                table_chunks = self._chunk_table_content(
                    group,
                    section_context,
                    metadata,
                    page_width,
                    page_height,
                    section
                )
                chunks.extend(table_chunks)
            elif group["type"] == "list":
                # Keep lists together when possible
                list_chunk = self._create_list_chunk(
                    group,
                    section_context,
                    metadata,
                    page_width,
                    page_height,
                    section
                )
                if list_chunk:
                    chunks.append(list_chunk)
            else:
                # Regular text chunking with context
                text_chunks = self._chunk_text_with_context(
                    group["content"],
                    section_context,
                    group.get("page", 0),
                    metadata,
                    group.get("bbox"),
                    page_width,
                    page_height,
                    section
                )
                chunks.extend(text_chunks)
        
        return chunks
    
    def _group_related_content(
        self,
        content_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Group related content that should stay together."""
        groups = []
        current_group = None
        
        for item in content_items:
            content = item.get("content", "")
            
            # Check if this looks like a list item
            if re.match(r"^\s*(?:•|\-|\d+\.|\w\))", content):
                if current_group and current_group["type"] == "list":
                    current_group["items"].append(item)
                else:
                    if current_group:
                        groups.append(current_group)
                    current_group = {
                        "type": "list",
                        "items": [item],
                        "page": item.get("page", 0),
                        "page_width": item.get("page_width"),
                        "page_height": item.get("page_height")
                    }
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = None
                
                if item["type"] == "table":
                    groups.append({
                        "type": "table",
                        "content": item,
                        "page": item.get("page", 0),
                        "page_width": item.get("page_width"),
                        "page_height": item.get("page_height")
                    })
                else:
                    groups.append({
                        "type": "text",
                        "content": content,
                        "page": item.get("page", 0),
                        "bbox": item.get("bbox", {}),
                        "page_width": item.get("page_width"),
                        "page_height": item.get("page_height")
                    })
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _chunk_table_content(
        self,
        table_group: Dict[str, Any],
        section_context: str,
        metadata: Dict[str, Any],
        page_width: Optional[Any] = None,
        page_height: Optional[Any] = None,
        section: Optional[HealthcareSection] = None
    ) -> List[Dict[str, Any]]:
        """Create chunks from table content."""
        table_data = table_group["content"].get("table_data", {})
        table_type = table_data.get("type", "generic")
        
        # For medical tables, try to keep related rows together
        if table_type == "lab_results":
            return self._chunk_lab_results_table(table_data, section_context, table_group, metadata, page_width, page_height, section)
        elif table_type == "medications":
            return self._chunk_medications_table(table_data, section_context, table_group, metadata, page_width, page_height, section)
        else:
            # Generic table handling
            return self._chunk_generic_table(table_group["content"], section_context, table_group, metadata, page_width, page_height, section)
    
    def _chunk_lab_results_table(
        self,
        table_data: Dict[str, Any],
        section_context: str,
        table_group: Dict[str, Any],
        metadata: Dict[str, Any],
        page_width: Optional[Any] = None,
        page_height: Optional[Any] = None,
        section: Optional[HealthcareSection] = None
    ) -> List[Dict[str, Any]]:
        """Chunk lab results keeping test groups together."""
        chunks = []
        headers = table_data.get("headers", [])
        data_rows = table_data.get("data", [])
        
        # Group related lab tests
        current_chunk_rows = [headers] if headers else []
        current_tokens = self._count_tokens(section_context)
        
        for row in data_rows:
            row_text = " | ".join(str(cell) for cell in row)
            row_tokens = self._count_tokens(row_text)
            
            if current_tokens + row_tokens > MAX_TABLE_CHUNK_SIZE and current_chunk_rows:
                # Create chunk
                chunk_text = self._format_table_chunk(current_chunk_rows, section_context)
                chunks.append(self._create_chunk(
                    chunk_text,
                    ChunkType.LAB_RESULT,
                    table_group.get("page", 0),
                    metadata,
                    page_width=page_width,
                    page_height=page_height,
                    section=section
                ))
                current_chunk_rows = [headers] if headers else []
                current_tokens = self._count_tokens(section_context)
            
            current_chunk_rows.append(row)
            current_tokens += row_tokens
        
        # Don't forget the last chunk
        if current_chunk_rows:
            chunk_text = self._format_table_chunk(current_chunk_rows, section_context)
            chunks.append(self._create_chunk(
                chunk_text,
                ChunkType.LAB_RESULT,
                table_group.get("page", 0),
                metadata,
                page_width=page_width,
                page_height=page_height,
                section=section
            ))
        
        return chunks
    
    def _chunk_medications_table(
        self,
        table_data: Dict[str, Any],
        section_context: str,
        table_group: Dict[str, Any],
        metadata: Dict[str, Any],
        page_width: Optional[Any] = None,
        page_height: Optional[Any] = None,
        section: Optional[HealthcareSection] = None
    ) -> List[Dict[str, Any]]:
        """Chunk medications keeping each medication entry complete."""
        chunks = []
        headers = table_data.get("headers", [])
        data_rows = table_data.get("data", [])
        
        # Each medication should be its own chunk with headers
        for row in data_rows:
            med_info = []
            if headers:
                med_info.append(headers)
            med_info.append(row)
            
            chunk_text = self._format_table_chunk(med_info, section_context)
            chunks.append(self._create_chunk(
                chunk_text,
                ChunkType.MEDICATION,
                table_group.get("page", 0),
                metadata,
                page_width=page_width,
                page_height=page_height,
                section=section
            ))
        
        return chunks
    
    def _chunk_generic_table(
        self,
        table_content: Dict[str, Any],
        section_context: str,
        table_group: Dict[str, Any],
        metadata: Dict[str, Any],
        page_width: Optional[Any] = None,
        page_height: Optional[Any] = None,
        section: Optional[HealthcareSection] = None
    ) -> List[Dict[str, Any]]:
        """Chunk generic tables."""
        # Fallback to original table handling
        table_text = table_content.get("content", "")
        if not table_text:
            return []
        
        full_text = section_context + "\n" + table_text
        tokens = self._count_tokens(full_text)
        
        if tokens <= MAX_TABLE_CHUNK_SIZE:
            return [self._create_chunk(
                full_text,
                ChunkType.TABLE,
                table_group.get("page", 0),
                metadata,
                page_width=page_width,
                page_height=page_height,
                section=section
            )]
        else:
            # Split large tables
            return self._chunk_text_with_context(
                table_text,
                section_context,
                table_group.get("page", 0),
                metadata,
                None,
                page_width,
                page_height,
                section
            )
    
    def _create_list_chunk(
        self,
        list_group: Dict[str, Any],
        section_context: str,
        metadata: Dict[str, Any],
        page_width: Optional[Any] = None,
        page_height: Optional[Any] = None,
        section: Optional[HealthcareSection] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a chunk from a list, keeping items together."""
        items = list_group.get("items", [])
        if not items:
            return None
        
        list_text = "\n".join(item.get("content", "") for item in items)
        full_text = section_context + "\n" + list_text
        
        tokens = self._count_tokens(full_text)
        if tokens <= TEXT_CHUNK_SIZE:
            return self._create_chunk(
                full_text,
                ChunkType.LIST,
                list_group.get("page", 0),
                metadata,
                page_width=page_width,
                page_height=page_height,
                section=section
            )
        else:
            # List too long, will be chunked as regular text
            return None
    
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
        
        # Add context to beginning
        full_text = section_context + text
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        if len(tokens) <= TEXT_CHUNK_SIZE:
            return [self._create_chunk(
                full_text,
                ChunkType.TEXT,
                page_num,
                metadata,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                section=section
            )]
        
        # Need to split - preserve context in each chunk
        chunks = []
        context_tokens = self.tokenizer.encode(section_context, add_special_tokens=False)
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Calculate effective chunk size after context
        effective_chunk_size = TEXT_CHUNK_SIZE - len(context_tokens)
        effective_overlap = TEXT_CHUNK_OVERLAP
        
        start = 0
        while start < len(text_tokens):
            end = min(start + effective_chunk_size, len(text_tokens))
            
            # Create chunk with context
            chunk_tokens = context_tokens + text_tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunks.append(self._create_chunk(
                chunk_text,
                ChunkType.TEXT,
                page_num,
                metadata,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                section=section
            ))
            
            # Move forward with overlap
            start = end - effective_overlap
            if start >= len(text_tokens):
                break
        
        return chunks
    
    def _format_table_chunk(self, rows: List[List[str]], context: str) -> str:
        """Format table rows into readable text."""
        lines = [context.strip()]
        
        for row in rows:
            line = " | ".join(str(cell) for cell in row)
            lines.append(line)
        
        return "\n".join(lines)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _create_chunk(
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
        """Create a standardized chunk with comprehensive metadata."""
        # Get page dimensions from method params or metadata
        if page_width is None:
            page_width = metadata.get("page_width", "NA")
        if page_height is None:
            page_height = metadata.get("page_height", "NA")
            
        # Ensure proper format for page dimensions
        page_width_final = page_width if isinstance(page_width, (int, float)) else "NA"
        page_height_final = page_height if isinstance(page_height, (int, float)) else "NA"
        
        # Clean text for analysis
        clean_text = text.strip()
        
        # Extract medical entities
        medical_entities, entity_types = self._extract_medical_entities(clean_text)
        
        # Classify content purpose
        answer_types = self._classify_content_purpose(clean_text)
        
        # Determine boost section from current section
        boost_section = None
        if section and section.section_type:
            boost_section = self._determine_section_boost_type(section.section_type)
        
        chunk = {
            "chunk": clean_text,
            "chunk_type": chunk_type.value,
            "page": page_num,
            "parse_type": metadata.get("parse_type", "pdf"),
            "page_width": page_width_final,
            "page_height": page_height_final,
            # New metadata for query processing
            "answer_types": answer_types,
            "medical_entities": medical_entities,
            "entity_types": list(set(entity_types)),  # Unique entity types
            "has_medical_content": len(medical_entities) > 0,
        }
        
        if bbox:
            chunk["bbox"] = bbox
        
        # Add section information
        if section:
            chunk["section_title"] = section.title
            chunk["section_type"] = section.section_type
        
        # Add boost section for retrieval
        if boost_section:
            chunk["boost_section"] = boost_section
        
        # Legacy medical type for backward compatibility
        if chunk_type == ChunkType.MEDICATION:
            chunk["medical_type"] = "medication"
        elif chunk_type == ChunkType.LAB_RESULT:
            chunk["medical_type"] = "lab_result"
        elif chunk_type == ChunkType.VITAL_SIGNS:
            chunk["medical_type"] = "vital_signs"
        
        return chunk
    
    def _classify_content_purpose(self, text: str) -> List[str]:
        """Classify what types of questions this chunk can answer."""
        answer_types = []
        text_lower = text.lower()
        
        # Definition patterns
        if any(pattern in text_lower for pattern in [
            "is a", "refers to", "defined as", "means", "definition",
            "also known as", "abbreviated as"
        ]):
            answer_types.append("definition")
        
        # Dosage patterns
        if any(pattern in text_lower for pattern in [
            "dose", "dosage", "dosing", "mg", "mcg", "ml", "units",
            "administration", "frequency", "daily", "twice", "three times",
            "every", "hours", "bid", "tid", "qid", "prn"
        ]):
            answer_types.append("dosage")
        
        # Side effects patterns
        if any(pattern in text_lower for pattern in [
            "side effect", "adverse", "reaction", "complication",
            "toxicity", "warning", "caution", "risk"
        ]):
            answer_types.append("side_effects")
        
        # Contraindications patterns
        if any(pattern in text_lower for pattern in [
            "contraindication", "do not use", "avoid", "interaction",
            "should not", "must not", "incompatible", "allergy",
            "hypersensitivity", "precaution"
        ]):
            answer_types.append("contraindications")
        
        # Treatment patterns
        if any(pattern in text_lower for pattern in [
            "treatment", "therapy", "management", "protocol",
            "guideline", "recommendation", "approach", "intervention"
        ]):
            answer_types.append("treatment")
        
        # Diagnosis patterns
        if any(pattern in text_lower for pattern in [
            "diagnosis", "diagnostic", "symptom", "sign", "test",
            "screening", "examination", "finding", "presentation",
            "criteria", "evaluation"
        ]):
            answer_types.append("diagnosis")
        
        # Procedure patterns
        if any(pattern in text_lower for pattern in [
            "procedure", "technique", "method", "step", "perform",
            "operation", "surgical", "protocol for"
        ]):
            answer_types.append("procedure")
        
        # Prevention patterns
        if any(pattern in text_lower for pattern in [
            "prevent", "prevention", "prophylaxis", "reduce risk",
            "avoid", "protective", "screening for prevention"
        ]):
            answer_types.append("prevention")
        
        # Comparison patterns
        if any(pattern in text_lower for pattern in [
            "versus", "vs", "compared to", "difference between",
            "better than", "worse than", "alternative"
        ]):
            answer_types.append("comparison")
        
        return answer_types if answer_types else ["general"]
    
    def _extract_medical_entities(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract medical entities from text."""
        medical_entities = []
        entity_types = []
        
        # Drug patterns
        drug_patterns = [
            r'\b(\w+)(cillin|cycline|mycin|statin|pril|sartan|olol|azole|prazole|pine|done|pam|zepam)\b',
            r'\b(aspirin|insulin|metformin|lisinopril|atorvastatin|levothyroxine|amlodipine|ibuprofen)\b',
        ]
        
        for pattern in drug_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = match.group(0).lower()
                if entity not in medical_entities:
                    medical_entities.append(entity)
                    entity_types.append("drug")
        
        # Disease patterns
        disease_patterns = [
            r'\b(\w+)(itis|osis|emia|oma|pathy|syndrome|disease|disorder)\b',
            r'\b(diabetes|hypertension|cancer|asthma|arthritis|pneumonia|covid|influenza)\b',
        ]
        
        for pattern in disease_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = match.group(0).lower()
                if entity not in medical_entities:
                    medical_entities.append(entity)
                    entity_types.append("disease")
        
        # Procedure patterns
        procedure_patterns = [
            r'\b(\w+)(scopy|ectomy|otomy|plasty|graphy|gram)\b',
            r'\b(surgery|biopsy|examination|screening|test|scan)\b',
        ]
        
        for pattern in procedure_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = match.group(0).lower()
                if entity not in medical_entities:
                    medical_entities.append(entity)
                    entity_types.append("procedure")
        
        return medical_entities, entity_types
    
    def _determine_section_boost_type(self, section_type: Optional[str]) -> Optional[str]:
        """Map section types to boost categories used by query processor."""
        # Map healthcare sections to query boost sections
        section_mapping = {
            "medications": "dosage",
            "dosage": "dosage",
            "administration": "dosage",
            "contraindications": "contraindications",
            "drug_interactions": "contraindications",
            "warnings": "contraindications",
            "adverse_reactions": "side_effects",
            "side_effects": "side_effects",
            "diagnosis": "diagnosis",
            "clinical_features": "diagnosis",
            "symptoms": "diagnosis",
            "signs": "diagnosis",
            "treatment": "treatment",
            "management": "treatment",
            "therapy": "treatment",
            "guidelines": "treatment",
            "procedure": "procedure",
            "protocol": "procedure",
            "technique": "procedure",
            "prevention": "prevention",
            "prophylaxis": "prevention",
        }
        
        return section_mapping.get(section_type)
    
    @classmethod
    def get_instance(cls) -> "HealthcareChunker":
        """Get singleton instance."""
        return cls()
