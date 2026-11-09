"""Unit tests for citations.py"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from rag_grounded_chatbot.citations import CitationGrounder, Receipt


def _make_node(text: str, file_name: str = "doc.txt", page: int | None = 1) -> NodeWithScore:
    node = TextNode(
        text=text, metadata={"file_name": file_name, "page_number": page, "chunk_index": 0}
    )
    return NodeWithScore(node=node, score=0.9)


class TestCitationGrounder:
    """Test CitationGrounder parsing and receipt building."""

    def test_ground_parses_citation(self, grounder: CitationGrounder) -> None:
        answer = "The refund window is 30 days [source: policy.pdf#p3]."
        nodes = [_make_node("Refund window is 30 days.", file_name="policy.pdf", page=3)]
        _result_answer, receipts = grounder.ground(answer, nodes)

        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt.file_name == "policy.pdf"
        assert receipt.page_number == 3
        assert receipt.citation == "[source: policy.pdf#p3]"
        assert receipt.quote == "Refund window is 30 days."

    def test_ground_deduplicates_identical_citations(self, grounder: CitationGrounder) -> None:
        answer = "Refunds allowed [source: policy.pdf#p3]. Also [source: policy.pdf#p3]."
        nodes = [_make_node("Refunds allowed.", file_name="policy.pdf", page=3)]
        _, receipts = grounder.ground(answer, nodes)
        assert len(receipts) == 1

    def test_ground_handles_missing_page_number(self, grounder: CitationGrounder) -> None:
        answer = "Info here [source: notes.txt]."
        nodes = [_make_node("Info here.", file_name="notes.txt", page=None)]
        _, receipts = grounder.ground(answer, nodes)
        assert len(receipts) == 1
        assert receipts[0].page_number is None
        assert receipts[0].file_name == "notes.txt"

    def test_ground_no_citations(self, grounder: CitationGrounder) -> None:
        answer = "No citations here."
        _, receipts = grounder.ground(answer, [])
        assert receipts == []

    def test_ground_returns_answer_unchanged(self, grounder: CitationGrounder) -> None:
        answer = "Some answer text."
        result_answer, _ = grounder.ground(answer, [])
        assert result_answer == answer

    def test_ground_multiple_citations(self, grounder: CitationGrounder) -> None:
        answer = "A [source: a.txt#p1]. B [source: b.txt#p2]."
        nodes = [
            _make_node("Text A.", file_name="a.txt", page=1),
            _make_node("Text B.", file_name="b.txt", page=2),
        ]
        _, receipts = grounder.ground(answer, nodes)
        assert len(receipts) == 2
        files = {r.file_name for r in receipts}
        assert files == {"a.txt", "b.txt"}

    def test_ground_citation_without_page_matches_node_without_page(
        self, grounder: CitationGrounder
    ) -> None:
        answer = "Text [source: notes.txt]."
        nodes = [_make_node("Text.", file_name="notes.txt", page=None)]
        _, receipts = grounder.ground(answer, nodes)
        assert len(receipts) == 1
        assert receipts[0].page_number is None

    def test_receipt_dataclass(self) -> None:
        receipt = Receipt(
            file_name="doc.pdf",
            page_number=5,
            quote="A quote.",
            citation="[source: doc.pdf#p.5]",
        )
        assert receipt.file_name == "doc.pdf"
        assert receipt.page_number == 5
        assert receipt.quote == "A quote."
        assert receipt.citation == "[source: doc.pdf#p.5]"
