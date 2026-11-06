"""
Configuration module for the Runeterra Lore RAG.
Handles environment variables, paths, model settings, and hardware detection.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

# API Keys & LLM settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Model settings
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Storage and Cache Directories
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
BOOKS_DIR = Path(os.getenv("BOOKS_DIR", DATA_DIR / "books"))
WEB_LORE_DIR = Path(os.getenv("WEB_LORE_DIR", DATA_DIR / "web_lore"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", DATA_DIR / "cache"))

# Ensure essential directories exist
for p in [DATA_DIR, BOOKS_DIR, WEB_LORE_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Chunking & Retrieval Hyperparameters
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", 320))
OVERLAP_WORDS = int(os.getenv("OVERLAP_WORDS", 50))
CANDIDATES = int(os.getenv("CANDIDATES", 50))
RERANK_POOL = int(os.getenv("RERANK_POOL", 20))
RRF_K = int(os.getenv("RRF_K", 60))
NEIGHBORS_FOR_TOP_N = int(os.getenv("NEIGHBORS_FOR_TOP_N", 2))
DEFAULT_RETRIEVAL_MODE = os.getenv("DEFAULT_RETRIEVAL_MODE", "hybrid+rerank")
LORE_LANG = os.getenv("LORE_LANG", "pt_BR")

# Device detection
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()

def get_book_paths():
    """
    Finds all PDF files in books directory, root directory, or Google Colab Drive.
    """
    found = []
    # 1. Search in data/books
    if BOOKS_DIR.exists():
        found.extend(list(BOOKS_DIR.glob("*.pdf")))
    # 2. Search in project root
    for pdf in BASE_DIR.glob("*.pdf"):
        if pdf not in found:
            found.append(pdf)
    # 3. Google Colab fallback
    colab_dir = Path("/content/drive/MyDrive/Lol")
    if colab_dir.exists():
        for pdf in colab_dir.glob("*.pdf"):
            if pdf not in found:
                found.append(pdf)

    return sorted(list(set(str(p.resolve()) for p in found)))
