"""Backend integration test: ingest -> index -> retrieve -> generate -> cite.

Does NOT launch or test the Gradio website.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from rag_grounded_chatbot.pipeline import GroundedRAGPipeline


def test_full_backend_flow(
    pipeline: GroundedRAGPipeline,
    tmp_path: Path,
    mock_ollama,
) -> None:
    """Full backend flow end-to-end with mocked LLM."""
    # Create sample text and PDF documents
    txt_file = tmp_path / "policy.txt"
    txt_file.write_text(
        "The refund window is 30 days from purchase date. Shipping is free for orders over $50.",
        encoding="utf-8",
    )

    pdf_file = tmp_path / "policy.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Returns must be in original condition.")
    doc.save(str(pdf_file))
    doc.close()

    # 1. Ingest + index
    count = pipeline.index_documents([str(txt_file), str(pdf_file)])
    assert count == 2

    # 2. Verify index exists
    assert pipeline.has_index() is True

    # 3. Query (retrieval + generation + citation grounding, all mocked LLM)
    answer, receipts, nodes = pipeline.query("What is the refund window?")

    # 4. Assertions
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert len(receipts) > 0, "Receipts list should be non-empty"
    assert len(nodes) > 0

    # Answer should contain a citation in the expected format
    import re

    citation_pattern = r"\[source:\s*[^\]#]+#p\d+\]"
    assert re.search(citation_pattern, answer), f"Answer missing citation format: {answer}"

    # Each receipt should have required fields
    for receipt in receipts:
        assert receipt.file_name
        assert receipt.citation
        assert receipt.citation.startswith("[source:")
