"""
Sentinel Log AI - Python ML Engine

This package provides the AI/ML capabilities for log analysis:
- Embedding generation using sentence-transformers
- Vector storage and similarity search with FAISS
- Clustering with HDBSCAN
- Novelty detection
- LLM-powered explanations via Ollama
"""

__version__ = "0.1.0"
__all__ = [
    "config",
    "embedding",
    "exceptions",
    "logging",
    "models",
    "normalization",
    "parser",
    "preprocessing",
    "server",
    "vectorstore",
]
