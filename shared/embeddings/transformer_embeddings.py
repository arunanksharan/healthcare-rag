"""
Transformer-based medical embedding implementations.
"""
import logging
import os
from typing import List, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from .base import BaseEmbedding, EmbeddingConfig

logger = logging.getLogger(__name__)

# Set cache directory for models
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class TransformerEmbedding(BaseEmbedding):
    """Base class for transformer-based embeddings."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def initialize(self) -> None:
        """Download and initialize the transformer model."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading {self.config.model_name} from {self.config.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                cache_dir=CACHE_DIR
            )
            
            self.model = AutoModel.from_pretrained(
                self.config.model_path,
                cache_dir=CACHE_DIR
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            logger.info(f"Successfully loaded {self.config.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.config.model_name}: {e}")
            raise
    
    def _pool_embeddings(self, model_output, attention_mask):
        """Apply pooling strategy to transformer outputs."""
        if self.config.pooling_strategy == "cls":
            # Use [CLS] token embedding
            return model_output.last_hidden_state[:, 0, :]
        
        elif self.config.pooling_strategy == "mean":
            # Mean pooling with attention mask
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.config.pooling_strategy == "max":
            # Max pooling
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(token_embeddings, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Pool embeddings
            embeddings = self._pool_embeddings(outputs, inputs["attention_mask"])
            
            # Normalize if configured
            if self.config.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to list and return
            return embeddings[0].cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error generating {self.config.model_name} embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Pool embeddings
            embeddings = self._pool_embeddings(outputs, inputs["attention_mask"])
            
            # Normalize if configured
            if self.config.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to list and return
            return embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error generating batch {self.config.model_name} embeddings: {e}")
            raise


class PubMedBERTEmbedding(TransformerEmbedding):
    """PubMedBERT embedding implementation."""
    pass


class BioBERTEmbedding(TransformerEmbedding):
    """BioBERT embedding implementation."""
    pass


class SciBERTEmbedding(TransformerEmbedding):
    """SciBERT embedding implementation."""
    pass


class ClinicalBERTEmbedding(TransformerEmbedding):
    """ClinicalBERT embedding implementation."""
    pass


class BioLinkBERTEmbedding(TransformerEmbedding):
    """BioLinkBERT embedding implementation."""
    pass
