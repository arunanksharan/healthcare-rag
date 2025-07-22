"""
Medical NER model registry for managing different NER implementations.
"""
import logging
from typing import Dict, Optional, Type, List
from threading import Lock

from .base import NERConfig, NERResult
from .biobert_ner import BioBERTNER

logger = logging.getLogger(__name__)


class MedicalNERRegistry:
    """
    Singleton registry for managing medical NER model instances.
    Similar pattern to the embedding registry.
    """
    
    _instance: Optional["MedicalNERRegistry"] = None
    _lock = Lock()
    
    def __new__(cls) -> "MedicalNERRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._ner_models: Dict[str, BioBERTNER] = {}
            self._configs: Dict[str, NERConfig] = {}
            self._default_model = "biomedical-ner-all"
            self._initialized = True
            logger.info("MedicalNERRegistry initialized")
    
    def register_model(self, name: str, config: NERConfig) -> None:
        """Register a NER model configuration."""
        self._configs[name] = config
        logger.info(f"Registered NER model: {name}")
    
    def get_model(self, name: Optional[str] = None) -> BioBERTNER:
        """
        Get or create a NER model instance.
        Models are lazily loaded and cached.
        """
        model_name = name or self._default_model
        
        if model_name not in self._ner_models:
            with self._lock:
                if model_name not in self._ner_models:
                    self._create_model_instance(model_name)
        
        return self._ner_models[model_name]
    
    def _create_model_instance(self, name: str) -> None:
        """Create and initialize a NER model instance."""
        # Check if we have a registered config
        if name in self._configs:
            config = self._configs[name]
        else:
            # Create default config for the model
            logger.info(f"Creating default config for NER model: {name}")
            config = NERConfig(
                model_name=name,
                model_path=name,
                confidence_threshold=0.7,
            )
        
        logger.info(f"Creating NER model instance for {name}")
        
        # Create instance
        model = BioBERTNER(config)
        
        # Initialize (download models, etc.)
        try:
            model.initialize()
            self._ner_models[name] = model
            logger.info(f"Successfully initialized {name} NER model")
        except Exception as e:
            logger.error(f"Failed to initialize {name} NER model: {e}")
            raise
    
    def preload_models(self, model_names: Optional[List[str]] = None) -> None:
        """
        Preload specified NER models into memory.
        If no names specified, preloads the default model.
        """
        if model_names is None:
            model_names = [self._default_model]
        
        for name in model_names:
            try:
                self.get_model(name)
                logger.info(f"Preloaded {name} NER model")
            except Exception as e:
                logger.error(f"Failed to preload {name}: {e}")
    
    def set_default_model(self, name: str) -> None:
        """Set the default NER model."""
        self._default_model = name
        logger.info(f"Set default NER model to: {name}")
    
    @classmethod
    def get_instance(cls) -> "MedicalNERRegistry":
        """Get the singleton instance of the NER registry."""
        return cls()


# Convenience function
def get_medical_ner_model(name: Optional[str] = None) -> BioBERTNER:
    """Get a medical NER model instance from the registry."""
    return MedicalNERRegistry.get_instance().get_model(name)


# Register default models on import
def _register_default_models():
    """Register commonly used medical NER models."""
    registry = MedicalNERRegistry.get_instance()
    
    # General purpose medical NER
    registry.register_model(
        "biomedical-ner-all",
        NERConfig(
            model_name="d4data/biomedical-ner-all",
            model_path="d4data/biomedical-ner-all",
            confidence_threshold=0.7,
            aggregate_subwords=True,
        )
    )
    
    # Clinical notes focused
    registry.register_model(
        "clinical-ner",
        NERConfig(
            model_name="emilyalsentzer/Bio_ClinicalBERT",
            model_path="emilyalsentzer/Bio_ClinicalBERT",
            confidence_threshold=0.75,
            aggregate_subwords=True,
        )
    )
    
    # Disease-specific NER
    registry.register_model(
        "disease-ner",
        NERConfig(
            model_name="alvaroalon2/biobert_diseases_ner",
            model_path="alvaroalon2/biobert_diseases_ner",
            confidence_threshold=0.8,
            aggregate_subwords=True,
        )
    )


# Register defaults when module is imported
_register_default_models()
