"""Unit tests for generation.py"""

from __future__ import annotations

from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode

from rag_grounded_chatbot.generation import LocalGenerator


def _make_node(
    text: str, score: float = 0.9, file_name: str = "doc.txt", page: int | None = 1
) -> NodeWithScore:
    node = TextNode(
        text=text, metadata={"file_name": file_name, "page_number": page, "chunk_index": 0}
    )
    return NodeWithScore(node=node, score=score)


class TestLocalGeneratorBuildContext:
    """Test LocalGenerator.build_context."""

    def test_build_context_labels_sources(self, mocked_generator: LocalGenerator) -> None:
        nodes = [
            _make_node("Content from doc A.", file_name="a.txt", page=1),
            _make_node("Content from doc B.", file_name="b.pdf", page=2),
        ]
        context = mocked_generator.build_context(nodes)

        assert "Source: a.txt#p1" in context
        assert "Source: b.pdf#p2" in context
        assert "Content from doc A." in context
        assert "Content from doc B." in context

    def test_build_context_unknown_file(self, mocked_generator: LocalGenerator) -> None:
        node = _make_node("Orphan content.", file_name="unknown", page=None)
        context = mocked_generator.build_context([node])
        assert "Source: unknown#pNone" in context

    def test_build_context_empty(self, mocked_generator: LocalGenerator) -> None:
        context = mocked_generator.build_context([])
        assert context == ""


class TestLocalGeneratorGenerate:
    """Test LocalGenerator.generate with mocked Ollama."""

    def test_generate_formats_prompt(
        self, mocked_generator: LocalGenerator, mock_ollama: MagicMock
    ) -> None:
        nodes = [_make_node("Context text.", file_name="doc.txt", page=1)]
        answer = mocked_generator.generate("What is the policy?", nodes)

        assert mock_ollama.complete.called
        call_args = mock_ollama.complete.call_args
        prompt = call_args[0][0]
        assert "SYSTEM_PROMPT" not in prompt  # system prompt is inline
        assert "Context documents:" in prompt
        assert "User question: What is the policy?" in prompt
        assert "Answer (include citations):" in prompt
        assert answer == "This is a mocked answer. [source: policy.txt#p1]"

    def test_generate_includes_system_prompt(
        self, mocked_generator: LocalGenerator, mock_ollama: MagicMock
    ) -> None:
        mocked_generator.generate("Q", [])
        prompt = mock_ollama.complete.call_args[0][0]
        assert "careful research assistant" in prompt
        assert "[source: <file_name>#p<page_number>]" in prompt

    def test_generate_returns_string(self, mocked_generator: LocalGenerator) -> None:
        answer = mocked_generator.generate("Q", [])
        assert isinstance(answer, str)
