"""
Sentinel Log AI - Python ML Engine

This package provides the AI/ML capabilities for log analysis:
- Embedding generation using sentence-transformers
- Vector storage and similarity search with FAISS
- Clustering with HDBSCAN
- Novelty detection with k-NN density estimation
- LLM-powered explanations via Ollama
- Rich CLI output with themes, progress, and reports
"""

__version__ = "0.6.0"
__all__ = [
    "cli",
    "clustering",
    "config",
    "embedding",
    "exceptions",
    "llm",
    "logging",
    "models",
    "normalization",
    "novelty",
    "parser",
    "preprocessing",
    "server",
    "vectorstore",
]
