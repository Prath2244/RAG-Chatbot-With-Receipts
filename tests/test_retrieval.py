"""Unit tests for retrieval.py"""

from __future__ import annotations

from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode

from rag_grounded_chatbot.retrieval import LocalRetriever


def _make_node(
    text: str, score: float = 0.9, file_name: str = "doc.txt", page: int | None = 1
) -> NodeWithScore:
    node = TextNode(
        text=text, metadata={"file_name": file_name, "page_number": page, "chunk_index": 0}
    )
    return NodeWithScore(node=node, score=score)


class TestLocalRetriever:
    """Test LocalRetriever retrieval behavior."""

    def test_retrieve_returns_top_k_nodes(self, retriever: LocalRetriever) -> None:
        fake_index = MagicMock()
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [
            _make_node("Node A", score=0.9),
            _make_node("Node B", score=0.8),
        ]
        fake_index.as_retriever.return_value = fake_retriever

        # Patch VectorIndexRetriever to use our fake
        from rag_grounded_chatbot import retrieval

        original = retrieval.VectorIndexRetriever
        retrieval.VectorIndexRetriever = MagicMock(return_value=fake_retriever)
        try:
            nodes = retriever.retrieve(fake_index, "test query")
        finally:
            retrieval.VectorIndexRetriever = original

        assert len(nodes) == 2
        assert nodes[0].node.text == "Node A"
        assert nodes[1].node.text == "Node B"

    def test_retrieve_empty_results(self, retriever: LocalRetriever) -> None:
        fake_index = MagicMock()
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = []
        fake_index.as_retriever.return_value = fake_retriever

        from rag_grounded_chatbot import retrieval

        original = retrieval.VectorIndexRetriever
        retrieval.VectorIndexRetriever = MagicMock(return_value=fake_retriever)
        try:
            nodes = retriever.retrieve(fake_index, "test query")
        finally:
            retrieval.VectorIndexRetriever = original

        assert nodes == []

    def test_retrieve_nodes_have_provenance(self, retriever: LocalRetriever) -> None:
        fake_index = MagicMock()
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [
            _make_node("Content with metadata", file_name="policy.pdf", page=3)
        ]
        fake_index.as_retriever.return_value = fake_retriever

        from rag_grounded_chatbot import retrieval

        original = retrieval.VectorIndexRetriever
        retrieval.VectorIndexRetriever = MagicMock(return_value=fake_retriever)
        try:
            nodes = retriever.retrieve(fake_index, "test query")
        finally:
            retrieval.VectorIndexRetriever = original

        assert len(nodes) == 1
        meta = nodes[0].node.metadata or {}
        assert meta["file_name"] == "policy.pdf"
        assert meta["page_number"] == 3
