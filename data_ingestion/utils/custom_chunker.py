import logging
from typing import Any, Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# --- Token-based chunk sizes ---
TEXT_CHUNK_SIZE = 512
TEXT_CHUNK_OVERLAP = 50
MIN_TEXT_CHUNK_SIZE = 100
HEADING_MIN_WORDS = 15


class ChunkingError(Exception):
    """Raised when chunking fails, carrying job_id for DLQ tracking."""

    def __init__(self, message: str, job_id: str):
        super().__init__(f"[job_id={job_id}] {message}")
        self.job_id = job_id


class CustomJsonChunker:
    """Singleton chunker for LlamaParse JSON output."""

    _instance: Optional["CustomJsonChunker"] = None
    _tokenizer_instance: AutoTokenizer | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def init_tokenizer_for_worker(cls):
        if cls._tokenizer_instance is None:
            try:
                logger.info("Loading tokenizer 'yiyanghkust/finbert-tone'…")
                cls._tokenizer_instance = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone", use_fast=True)
                logger.info("Tokenizer loaded.")
            except Exception as e:
                logger.error(f"Tokenizer init error: {e}", exc_info=True)
                raise

    @property
    def tokenizer(self) -> AutoTokenizer:
        if CustomJsonChunker._tokenizer_instance is None:
            # lazy init on first chunk_json call
            self.init_tokenizer_for_worker()
        return CustomJsonChunker._tokenizer_instance

    @classmethod
    def get_instance(cls) -> "CustomJsonChunker":
        return cls()

    def chunk_json(
        self,
        parsed_json: dict[str, Any],
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        job_id = parsed_json.get("job_id")
        original_filename = metadata.get("original_filename") 
        parse_type = metadata.get("parse_type", "unknown")
        pages = parsed_json.get("pages")
        if not isinstance(pages, list):
            raise ChunkingError("'pages' missing or not a list", job_id)

        out: list[dict[str, Any]] = []
        for page in pages:
            self._process_page(page, metadata, out, job_id, original_filename, parse_type) 
        logger.info(f"Generated {len(out)} chunks for job_id={job_id}, original_filename='{original_filename}'") 
        return out

    def _process_page(  # noqa: PLR0913
        self,
        page: dict[str, Any],
        metadata: dict[str, Any],
        out: list[dict[str, Any]],
        job_id: str,
        original_filename: str, 
        parse_type: str,
    ):
        if not isinstance(page, dict):
            return
        page_num = page.get("page")
        if not isinstance(page_num, int):
            return

        # Get page dimensions, default to 'NA' if not valid numbers
        raw_page_width = page.get("width")
        raw_page_height = page.get("height")
        page_width_for_chunk = raw_page_width if isinstance(raw_page_width, int | float) else "NA"
        page_height_for_chunk = raw_page_height if isinstance(raw_page_height, int | float) else "NA"

        # Images
        for image in page.get("images") or []:
            try:
                image_chunk = self._chunk_image(
                    image,
                    page_num,
                    parse_type,
                    page_width_for_chunk,
                    page_height_for_chunk,
                )
                if image_chunk:
                    out.append(image_chunk)
            except Exception as e:
                logger.error(f"[Image] {e} (job_id={job_id}, original_filename='{original_filename}')") 

        # Items or fallback
        items = page.get("items")
        if isinstance(items, list) and items:
            for item in items:
                self._process_item(
                    item,
                    page_num,
                    out,
                    job_id,
                    original_filename, 
                    parse_type,
                    page_width_for_chunk,
                    page_height_for_chunk,
                )
        else:
            self._fallback_page_text(
                page,
                page_num,
                out,
                job_id,
                original_filename, 
                parse_type,
                page_width_for_chunk,
                page_height_for_chunk,
            )

    def _process_item(  # noqa: PLR0913
        self,
        item: dict[str, Any],
        page_num: int,
        out: list[dict[str, Any]],
        job_id: str,
        original_filename: str, 
        parse_type: str,
        page_width: Any,  # Can be int, float, or 'NA'
        page_height: Any,  # Can be int, float, or 'NA'
    ):
        itype = item.get("type")
        bbox = item.get("bBox")
        if not (isinstance(bbox, dict) and all(k in bbox for k in ("x", "y", "w", "h"))):
            return

        md = item.get("md") or ""
        if not isinstance(md, str):
            md = ""

        try:
            if itype == "heading":
                heading_chunk = self._chunk_heading(md, bbox, page_num, parse_type, page_width, page_height)
                if heading_chunk:
                    out.append(heading_chunk)
            elif itype == "table":
                table_chunk = self._chunk_table(item, bbox, page_num, parse_type, page_width, page_height)
                if table_chunk:
                    out.append(table_chunk)
            elif itype == "text":
                out.extend(self._chunk_text(md, bbox, page_num, original_filename, parse_type, page_width, page_height)) 
        except Exception as e:
            logger.error(f"[Item:{itype}] {e} (job_id={job_id}, original_filename='{original_filename}')") 

    def _fallback_page_text(  # noqa: PLR0913
        self,
        page: dict[str, Any],
        page_num: int,
        out: list[dict[str, Any]],
        job_id: str,
        original_filename: str,
        parse_type: str,
        page_width: Any,
        page_height: Any,
    ):
        content = page.get("md") or page.get("text") or ""
        if not isinstance(content, str) or not content.strip():
            return
        bbox = {"x": 0, "y": 0, "w": page.get("width"), "h": page.get("height")}
        if not all(isinstance(v, int | float) for v in bbox.values()):
            return
        out.extend(self._split_text_with_overlap(content, bbox, page_num, original_filename, parse_type, page_width, page_height))

    def _chunk_image(
        self,
        image: dict[str, Any],
        page_num: int,
        parse_type: str,
        page_width: Any,
        page_height: Any,
    ) -> dict[str, Any] | None:
        ocr = image.get("ocr")
        if not isinstance(ocr, list) or not ocr:
            return None
        texts = [o.get("text", "").strip() for o in ocr if isinstance(o, dict) and o.get("text")]
        full = " ".join(texts).strip()
        if not full:
            return None
        bbox = {
            "x": image.get("x"),
            "y": image.get("y"),
            "w": image.get("width"),
            "h": image.get("height"),
        }
        if not all(isinstance(v, int | float) for v in bbox.values()):
            return None
        return {
            "page": page_num,
            "bbox": bbox,
            "chunk": full,
            "parse_type": parse_type,
            "page_width": page_width,
            "page_height": page_height,
        }

    def _chunk_heading(  # noqa: PLR0913
        self,
        text: str,
        bbox: dict[str, Any],
        page_num: int,
        parse_type: str,
        page_width: Any,
        page_height: Any,
    ) -> dict[str, Any] | None:
        if len(text.split()) > HEADING_MIN_WORDS:
            return {
                "page": page_num,
                "bbox": bbox,
                "chunk": text.strip(),
                "parse_type": parse_type,
                "page_width": page_width,
                "page_height": page_height,
            }
        return None

    def _chunk_table(  # noqa: PLR0913
        self,
        item: dict[str, Any],
        bbox: dict[str, Any],
        page_num: int,
        parse_type: str,
        page_width: Any,
        page_height: Any,
    ) -> dict[str, Any] | None:
        rows = item.get("rows")
        if isinstance(rows, list):
            lines = ["\t".join(str(cell) for cell in row) for row in rows if isinstance(row, list)]
            text = "\n".join(lines).strip()
        else:
            text = item.get("md", "").strip()
        if text:
            return {
                "page": page_num,
                "bbox": bbox,
                "chunk": text,
                "parse_type": parse_type,
                "page_width": page_width,
                "page_height": page_height,
            }
        return None

    def _chunk_text(  # noqa: PLR0913
        self,
        text: str,
        bbox: dict[str, Any],
        page_num: int,
        original_filename: str,
        parse_type: str,
        page_width: Any,
        page_height: Any,
    ) -> list[dict[str, Any]]:
        return self._split_text_with_overlap(text, bbox, page_num, original_filename, parse_type, page_width, page_height)

    def _split_text_with_overlap(  # noqa: PLR0912, PLR0913
        self,
        text: str,
        bbox: dict[str, Any],
        page_num: int,
        original_filename: str,
        parse_type: str,
        page_width: Any,
        page_height: Any,
    ) -> list[dict[str, Any]]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)

        # Handle empty input
        if total_tokens == 0:
            return []

        # Handle text that is already short enough (does not need splitting loop)
        # If total_tokens <= TEXT_CHUNK_SIZE, it means the text might not need to be split.
        # We keep it if it decodes to any non-empty string.
        # The MIN_TEXT_CHUNK_SIZE is not applied here to discard it, because this text
        # was short to begin with, not a small remnant of a split.
        if total_tokens <= TEXT_CHUNK_SIZE:
            decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if decoded_text:  # If there's actual content after decoding
                return [
                    {
                        "page": page_num,
                        "bbox": bbox,
                        "chunk": decoded_text,
                        "parse_type": parse_type,
                        "page_width": page_width,
                        "page_height": page_height,
                    }
                ]
            else:
                # Decoded to empty string (e.g., all whitespace or only special tokens tokenizer might strip)
                return []

        # Text is longer than TEXT_CHUNK_SIZE and needs the splitting process
        split_token_parts: list[list[int]] = []
        # Ensure overlap is less than chunk size to guarantee progress
        current_overlap = min(TEXT_CHUNK_OVERLAP, TEXT_CHUNK_SIZE - 1) if TEXT_CHUNK_SIZE > 0 else 0

        start_index = 0
        while start_index < total_tokens:
            end_index = min(start_index + TEXT_CHUNK_SIZE, total_tokens)
            current_part_tokens = tokens[start_index:end_index]

            if current_part_tokens:  # Make sure the slice is not empty
                split_token_parts.append(current_part_tokens)

            if end_index == total_tokens:  # Reached the end of all tokens
                break

            next_start_index = end_index - current_overlap

            # Safeguard: if next_start_index doesn't advance, break to prevent infinite loop.
            # This should be rare with current_overlap calculation ensuring overlap < chunk_size.
            if next_start_index <= start_index:
                logger.warning(
                    f"Chunk splitting logic stalled for original_filename='{original_filename}'. Overlap issue: "
                    f"next_start_index ({next_start_index}) <= start_index ({start_index}). "
                    f"Review TEXT_CHUNK_SIZE ({TEXT_CHUNK_SIZE}) and TEXT_CHUNK_OVERLAP ({TEXT_CHUNK_OVERLAP}). "
                    f"Breaking loop. Text processed so far for this item: '{text[: start_index + 50]}...'"
                )
                break

            start_index = next_start_index

        if not split_token_parts:  # Should not happen if total_tokens > TEXT_CHUNK_SIZE, but as a fallback
            return []

        # Handle merging of a potentially too-small last chunk from the splitting process
        # The `len(split_token_parts) > 1` check correctly guards `split_token_parts[-2]`
        if len(split_token_parts) > 1 and len(split_token_parts[-1]) < MIN_TEXT_CHUNK_SIZE:
            split_token_parts[-2].extend(split_token_parts[-1])
            split_token_parts.pop()

        # Decode token parts into text chunks and apply final filtering
        final_chunks: list[dict[str, Any]] = []
        for part_tokens in split_token_parts:
            if not part_tokens:  # Should not happen
                continue

            decoded_text = self.tokenizer.decode(part_tokens, skip_special_tokens=True).strip()

            # Filter based on MIN_TEXT_CHUNK_SIZE (token count of this specific part)
            # and ensure the decoded text is not empty.
            if len(part_tokens) >= MIN_TEXT_CHUNK_SIZE and decoded_text:
                final_chunks.append(
                    {
                        "page": page_num,
                        "bbox": bbox,
                        "chunk": decoded_text,
                        "parse_type": parse_type,
                        "page_width": page_width,
                        "page_height": page_height,
                    }
                )

        return final_chunks
