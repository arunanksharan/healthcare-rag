import logging

from ..parsers.base_parser import BaseParser
from ..parsers.document_parser import DocumentParser
from .custom_chunker import CustomJsonChunker
from .healthcare_chunker import HealthcareChunker
from shared.embeddings import EmbeddingType, get_embedding_model
from data_ingestion.app.api.models import ChunkerType

logger = logging.getLogger(__name__)

PARSER_MAP: dict[str, type[BaseParser]] = {
    "guideline": DocumentParser,
    "policy": DocumentParser,
    "faq": DocumentParser,
    "document": DocumentParser,
}

DOC_TYPE_TO_PARSE_TYPE_MAP: dict[str, str] = {
    "guideline": "pdf",
    "policy": "pdf",
    "faq": "pdf",
    "document": "pdf",
}


def get_parser(doc_type: str) -> type[BaseParser]:
    # 'doc_type' parameter now refers to metadata['type']
    key = doc_type.lower().strip()
    parser_cls = PARSER_MAP.get(key)
    if not parser_cls:
        logger.error(f"No parser for type '{doc_type}'")
        raise ValueError(f"No parser for type '{doc_type}'")
    return parser_cls


def parse_document(content: bytes, metadata: dict[str, any]) -> dict[str, any]:
    """
    Parses the document content using the appropriate parser based on metadata.

    Args:
        content: The raw bytes of the document to parse.
        metadata: A dictionary containing metadata about the document,
                including 'type' and 'original_filename'.

    Returns:
        A dictionary containing the parsed document structure from LlamaParse.
    """
    parser_cls = get_parser(metadata["type"])
    original_filename = metadata.get("original_filename")

    try:
        # Step 1: Parse the main document
        logger.info(f"Parsing document ({parser_cls.__name__}) for filename='{original_filename}'...")
        parsed_json = parser_cls.parse(raw_bytes=content, original_filename=original_filename)
        logger.info(f"Successfully parsed document for filename='{original_filename}'.")

        return parsed_json
    except Exception as e:
        logger.error(
            f"Overall processing error in parse_document for filename='{original_filename}': {e}",
            exc_info=True,
        )
        raise


def chunk_parsed_document(parsed_json: dict[str, any], metadata: dict[str, any]) -> list[dict[str, any]]:
    original_filename = metadata.get("original_filename")
    doc_type_original = metadata.get("type", "").lower().strip()
    parse_type = DOC_TYPE_TO_PARSE_TYPE_MAP.get(doc_type_original, "unknown")
    
    # Get chunker type from metadata
    chunker_type_str = metadata.get("chunker_type", ChunkerType.get_default().value)
    try:
        chunker_type = ChunkerType(chunker_type_str)
    except ValueError:
        logger.warning(f"Invalid chunker_type '{chunker_type_str}', using default")
        chunker_type = ChunkerType.get_default()

    # Creating a copy of metadata to add parse_type without altering the original dict upstream
    metadata_with_parse_type = metadata.copy()
    metadata_with_parse_type["parse_type"] = parse_type

    try:
        logger.info(f"Chunking parsed document for filename='{original_filename}' (parse_type='{parse_type}', chunker_type='{chunker_type.value}')...")
        
        # Check if NER enrichment is requested
        use_ner = metadata.get("enable_ner", False)
        
        # Select appropriate chunker
        if chunker_type == ChunkerType.HEALTHCARE:
            if use_ner:
                # Use enhanced chunker with NER
                logger.info(f"Using enhanced healthcare chunker with NER for '{original_filename}'")
                from ..utils.enhanced_healthcare_chunker import EnhancedHealthcareChunker
                chunker = EnhancedHealthcareChunker.get_instance()
                # Initialize tokenizer if needed
                if not hasattr(EnhancedHealthcareChunker, '_tokenizer_instance') or EnhancedHealthcareChunker._tokenizer_instance is None:
                    EnhancedHealthcareChunker.init_tokenizer_for_worker()
            else:
                # Use regular healthcare chunker
                chunker = HealthcareChunker.get_instance()
                # Initialize healthcare chunker with PubMedBERT tokenizer if not already done
                if not hasattr(HealthcareChunker, '_tokenizer_instance') or HealthcareChunker._tokenizer_instance is None:
                    HealthcareChunker.init_tokenizer_for_worker()
        else:
            chunker = CustomJsonChunker.get_instance()
            
        chunks = chunker.chunk_json(parsed_json, metadata=metadata_with_parse_type)
        logger.info(f"Generated {len(chunks)} chunks using {chunker_type.value} chunker for filename='{original_filename}'")
        return chunks
    except Exception as e:
        job_id = parsed_json.get("job_id", "unknown_job_id")  # Get job_id for logging
        logger.error(
            f"Chunking error for filename='{original_filename}', job_id='{job_id}': {e}",
            exc_info=True,
        )
        raise


def generate_embeddings(chunks: list[dict[str, any]], embedding_type: EmbeddingType) -> list:
    """
    Generate embeddings for chunks using the specified embedding model.
    
    Args:
        chunks: List of chunk dictionaries containing 'chunk' text
        embedding_type: The type of embedding model to use
        
    Returns:
        List of embedding vectors
    """
    # Get the embedding model from registry
    embedding_model = get_embedding_model(embedding_type)
    
    # Extract texts from chunks
    texts = []
    valid_indices = []
    
    for i, chunk in enumerate(chunks):
        text = chunk.get("chunk", "").strip()
        if text:
            texts.append(text)
            valid_indices.append(i)
        else:
            logger.warning(f"Skipping empty chunk in embedding generation: {chunk}")
    
    if not texts:
        logger.warning("No valid texts to embed; possibly empty input chunks.")
        return []
    
    # Generate embeddings in batch for efficiency
    try:
        embeddings_batch = embedding_model.embed_batch(texts)
        
        # Create full embeddings list with None for skipped chunks
        embeddings = [None] * len(chunks)
        for i, embedding in zip(valid_indices, embeddings_batch):
            embeddings[i] = embedding
        
        # Filter out None values
        embeddings = [e for e in embeddings if e is not None]
        
        logger.info(f"Generated {len(embeddings)} embeddings using {embedding_type.value}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings with {embedding_type.value}: {e}")
        raise
