"""
Embeddings module using Hugging Face sentence-transformers.
Runs locally without paid APIs, generating 384-dimensional dense semantic vectors.
"""

from typing import Optional
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM
from utils.logging_utils import get_logger

logger = get_logger("embeddings")

_embed_model_instance: Optional[HuggingFaceEmbedding] = None

def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbedding:
    """
    Returns a cached HuggingFaceEmbedding model instance and sets it in global Settings.
    The model runs locally on CPU or GPU without external API calls.
    """
    global _embed_model_instance
    if _embed_model_instance is None:
        logger.info(f"Loading local embedding model: {model_name}...")
        _embed_model_instance = HuggingFaceEmbedding(
            model_name=model_name,
            # Normalize embeddings to enable fast cosine similarity
            normalize=True
        )
        # Configure LlamaIndex global settings
        Settings.embed_model = _embed_model_instance
        logger.info(f"Embedding model loaded successfully. Dimension: {EMBEDDING_DIM}")
    return _embed_model_instance

def get_embedding_dimension() -> int:
    """Returns the expected embedding dimension (384 for all-MiniLM-L6-v2)."""
    return EMBEDDING_DIM
