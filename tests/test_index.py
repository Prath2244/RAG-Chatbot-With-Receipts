"""Unit tests for index.py"""

from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import Document

from rag_grounded_chatbot.config import Config
from rag_grounded_chatbot.index import LocalVectorStore


class TestLocalVectorStore:
    """Test LocalVectorStore build/load round-trip."""

    def test_build_and_load_index_round_trip(
        self, tmp_path: Path, vector_store: LocalVectorStore
    ) -> None:
        documents = [
            Document(
                text="First document content.",
                metadata={"file_name": "a.txt", "page_number": 1, "chunk_index": 0},
            ),
            Document(
                text="Second document content.",
                metadata={"file_name": "b.txt", "page_number": 2, "chunk_index": 0},
            ),
        ]

        index = vector_store.build_index(documents)
        assert index is not None

        loaded = vector_store.load_index()
        assert loaded is not None

    def test_load_index_no_index(self, vector_store: LocalVectorStore) -> None:
        loaded = vector_store.load_index()
        assert loaded is None

    def test_vector_store_uses_temp_chroma_dir(self, tmp_path: Path) -> None:
        config = Config(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            documents_dir=tmp_path / "data" / "documents",
            storage_dir=tmp_path / "data" / "storage",
            chroma_dir=tmp_path / "data" / "chroma",
        )
        store = LocalVectorStore(config)
        assert store is not None
