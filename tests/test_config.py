"""Unit tests for config.py"""

from __future__ import annotations

from pathlib import Path

from rag_grounded_chatbot.config import Config


class TestConfigDefaults:
    """Test default Config values."""

    def test_default_paths(self) -> None:
        config = Config()
        assert config.project_root == Path(__file__).resolve().parents[1]
        assert config.data_dir == config.project_root / "data"
        assert config.documents_dir == config.data_dir / "documents"
        assert config.storage_dir == config.data_dir / "storage"
        assert config.chroma_dir == config.data_dir / "chroma"

    def test_default_models(self) -> None:
        config = Config()
        assert config.embedding_model == "BAAI/bge-small-en-v1.5"
        assert config.rerank_model == "BAAI/bge-reranker-small"
        assert config.llm_model == "llama3.1:8b-instruct-q4_K_M"
        assert config.ollama_host == "http://localhost:11434"

    def test_default_ingestion_settings(self) -> None:
        config = Config()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.top_k == 10
        assert config.rerank_top_k == 5

    def test_citation_format(self) -> None:
        config = Config()
        assert config.citation_format == "[source: {file}#p{page}]"


class TestConfigPostInit:
    """Test Config.__post_init__ directory creation."""

    def test_creates_directories(self, tmp_path: Path) -> None:
        config = Config(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            documents_dir=tmp_path / "data" / "documents",
            storage_dir=tmp_path / "data" / "storage",
            chroma_dir=tmp_path / "data" / "chroma",
        )
        assert config.data_dir.exists()
        assert config.documents_dir.exists()
        assert config.storage_dir.exists()
        assert config.chroma_dir.exists()

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "data"
        Config(
            project_root=tmp_path,
            data_dir=deep,
            documents_dir=deep / "documents",
            storage_dir=deep / "storage",
            chroma_dir=deep / "chroma",
        )
        assert deep.exists()
        assert (deep / "documents").exists()
        assert (deep / "storage").exists()
        assert (deep / "chroma").exists()
