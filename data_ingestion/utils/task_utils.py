import logging

from ..parsers.base_parser import BaseParser
from ..parsers.document_parser import DocumentParser
from .custom_chunker import CustomJsonChunker

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

    # Creating a copy of metadata to add parse_type without altering the original dict upstream
    metadata_with_parse_type = metadata.copy()
    metadata_with_parse_type["parse_type"] = parse_type

    try:
        logger.info(f"Chunking parsed document for filename='{original_filename}' (parse_type='{parse_type}')...")
        chunker = CustomJsonChunker.get_instance()
        chunks = chunker.chunk_json(parsed_json, metadata=metadata_with_parse_type)
        logger.info(f"Generated {len(chunks)} chunks for filename='{original_filename}'")
        return chunks
    except Exception as e:
        job_id = parsed_json.get("job_id", "unknown_job_id")  # Get job_id for logging
        logger.error(
            f"Chunking error for filename='{original_filename}', job_id='{job_id}': {e}",
            exc_info=True,
        )
        raise


def generate_embeddings(chunks: list[dict[str, any]]) -> list:
    from .openai_embedding import get_openai_embedding

    embeddings = []
    for chunk in chunks:
        text = chunk.get("chunk", "").strip()
        if not text:
            logger.warning(f"Skipping empty chunk in embedding generation: {chunk}")
            continue
        embeddings.append(get_openai_embedding(text))

    if not embeddings:
        logger.warning("No embeddings generated; possibly empty input chunks.")
    return embeddings
