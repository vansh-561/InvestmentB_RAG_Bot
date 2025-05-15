"""
Investment Banking RAG Bot package.
A Retrieval Augmented Generation system for investment banking documents.
"""

__version__ = "0.1.0"

# Define public API
__all__ = ["config", "PDFProcessor", "GeminiEmbeddings", "GeminiLLM", "VectorStore", "RAGEngine"]

# Import main components for easier access
from .config import config

# Avoid circular imports by importing components individually
# rather than importing RAGEngine which depends on all other components
from .pdf_processor import PDFProcessor
from .embeddings import GeminiEmbeddings
from .llm import GeminiLLM
from .vector_store import VectorStore

# Import RAGEngine last to avoid circular dependency issues
from .rag_engine import RAGEngine

# Validate configuration on import
valid, message = config.validate_api_keys()
if not valid:
    import warnings
    warnings.warn(f"Configuration issue: {message}")
