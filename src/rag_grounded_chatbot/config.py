"""Central configuration for the grounded RAG chatbot."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Directory layout
    # ------------------------------------------------------------------
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    documents_dir: Path = data_dir / "documents"
    storage_dir: Path = data_dir / "storage"
    chroma_dir: Path = data_dir / "chroma"

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "llama3.2:3b"
    max_new_tokens: int = 2048
    ollama_host: str = "http://localhost:11434"

    # ------------------------------------------------------------------
    # Ingestion / retrieval
    # ------------------------------------------------------------------
    chunk_size: int = 1024
    chunk_overlap: int = 128
    top_k: int = 20           # candidates retrieved before reranking
    rerank_top_k: int = 15    # final chunks passed to the LLM

    # ------------------------------------------------------------------
    # Retrieval strategy toggles (performance vs. accuracy)
    # ------------------------------------------------------------------
    use_bm25: bool = True      # enable BM25 for keyword-aware retrieval
    use_query_expansion: bool = False

    # ------------------------------------------------------------------
    # GPU / device acceleration
    # ------------------------------------------------------------------
    device: str = "cpu"  # "cpu" or "cuda" — set to "cuda" for GPU acceleration

    # ------------------------------------------------------------------
    # OCR for scanned / image-only PDF pages
    # ------------------------------------------------------------------
    ocr_enabled: bool = True  # attempt OCR when digital text is sparse
    ocr_min_text_chars: int = 50  # pages below this get OCR'd
    ocr_dpi: int = 200  # render resolution for OCR

    # ------------------------------------------------------------------
    # Citation format used in prompts and parsed from answers
    # ------------------------------------------------------------------
    citation_format: str = "[source: {file}#p{page}]"

    def __post_init__(self) -> None:
        for path in (self.data_dir, self.documents_dir,
                     self.storage_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)