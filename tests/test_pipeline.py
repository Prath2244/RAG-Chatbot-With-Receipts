"""Unit tests for pipeline.py"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_grounded_chatbot.pipeline import GroundedRAGPipeline


class TestGroundedRAGPipelineNoIndex:
    """Test pipeline behavior when no index exists."""

    def test_query_returns_no_index_message(self, pipeline: GroundedRAGPipeline) -> None:
        answer, receipts, nodes = pipeline.query("What is the policy?")
        assert answer == "No documents have been indexed yet. Please upload documents first."
        assert receipts == []
        assert nodes == []

    def test_has_index_false(self, pipeline: GroundedRAGPipeline) -> None:
        assert pipeline.has_index() is False


class TestGroundedRAGPipelineHappyPath:
    """Test pipeline happy path with mocked dependencies."""

    def test_index_documents(self, pipeline: GroundedRAGPipeline, tmp_path: Path) -> None:
        # Create a small text file
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("Hello world content here.", encoding="utf-8")

        with patch.object(pipeline._vector_store, "build_index") as mock_build:
            count = pipeline.index_documents([str(txt_file)])
            assert count == 1
            mock_build.assert_called_once()
            assert pipeline._indexed is True

    def test_query_happy_path(self, pipeline: GroundedRAGPipeline, mock_ollama: MagicMock) -> None:
        # Mock the index and retriever
        fake_node = MagicMock()
        fake_node.node.text = "Context text."
        fake_node.node.metadata = {"file_name": "doc.txt", "page_number": 1, "chunk_index": 0}

        with (
            patch.object(pipeline._vector_store, "load_index", return_value=MagicMock()),
            patch.object(pipeline._retriever, "retrieve", return_value=[fake_node]),
            patch.object(
                pipeline._generator, "generate", return_value="Answer [source: doc.txt#p1]"
            ),
        ):
            answer, receipts, nodes = pipeline.query("What is the policy?")

            assert "Answer" in answer
            assert len(receipts) == 1
            assert receipts[0].file_name == "doc.txt"
            assert len(nodes) == 1
