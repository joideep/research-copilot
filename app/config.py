"""Central configuration, loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/processed/chroma")
    bm25_index_path: str = os.getenv("BM25_INDEX_PATH", "./data/processed/bm25_index.pkl")
    memory_db_path: str = os.getenv("MEMORY_DB_PATH", "./data/processed/memory.json")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    generation_model: str = "claude-sonnet-4-6"

    # Chunking parameters
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval parameters
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_final: int = 6
    # Weight given to the dense (semantic) score vs sparse (BM25) score
    # when combining results in hybrid search, 0-1.
    dense_weight: float = 0.6


settings = Settings()
