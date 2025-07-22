from abc import ABC, abstractmethod


class BaseParser(ABC):
    @classmethod
    @abstractmethod
    def parse(cls, raw_bytes: bytes, original_filename: str) -> dict:
        """
        Parse a document (PDF/HTML/etc) into llama parse json result.
        raw_bytes: file contents
        original_filename:   original filename (so you can extract file_name/extension)
        Returns llama parse json result.
        """
        pass
