import logging
from pathlib import Path

from llama_parse import LlamaParse

from ..core.settings import settings
from .base_parser import BaseParser

logger = logging.getLogger(__name__)

parser = LlamaParse(
    api_key=settings.llama_cloud_api_key,
    auto_mode=True,
    result_type="json",
)

parser2 = LlamaParse(api_key=settings.llama_cloud_api_key, premium_mode=True, result_type="json")


class DocumentParser(BaseParser):
    @classmethod
    def parse(cls, raw_bytes: bytes, original_filename: str) -> dict:
        file_name = Path(original_filename).name
        try:
            result = parser.parse(raw_bytes, extra_info={"file_name": file_name})
            return result.model_dump()
        except Exception as e:
            logger.error(f"[DocumentParser] parse error for {original_filename}: {e}")
            try:
                result = parser2.parse(raw_bytes, extra_info={"file_name": file_name})
                return result.model_dump()
            except Exception as e:
                logger.error(f"[DocumentParser] fallback parse error for {original_filename}: {e}")

            return {}
