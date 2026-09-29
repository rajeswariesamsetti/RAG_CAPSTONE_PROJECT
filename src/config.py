"""
Configuration module for the RAG Knowledge Assistant.
Centralizes all paths, model identifiers, default hyperparameters, and environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory: root of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Directories
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
TESTS_DIR = BASE_DIR / "tests"

# Ensure essential runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = "capstone_knowledge_base"

# Embedding Model Configuration
# sentence-transformers/all-MiniLM-L6-v2 creates 384-dimensional dense vectors locally
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Document Chunking Defaults
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64

# Retrieval Defaults
DEFAULT_SIMILARITY_TOP_K = 3

# Groq LLM Configuration
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
FALLBACK_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]
AVAILABLE_GROQ_MODELS = FALLBACK_GROQ_MODELS.copy()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Ollama LLM Configuration (Alternative local option)
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Strict RAG fallback response when information is insufficient
OUT_OF_CONTEXT_RESPONSE = "I don't have enough information to answer that."

# Supported Document Extensions
SUPPORTED_EXTENSIONS = [".pdf", ".txt"]
