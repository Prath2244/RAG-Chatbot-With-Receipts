"""Unit tests for ingestion.py"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import fitz  # PyMuPDF
import pytest

from rag_grounded_chatbot.config import Config
from rag_grounded_chatbot.ingestion import IngestedChunk, LocalDataIngestion


class _FakeTable:
    def __init__(self, rows: list[list[str | None]]) -> None:
        self._rows = rows

    def extract(self) -> list[list[str | None]]:
        return self._rows


class _FakeTableFinder:
    def __init__(self, tables: list[_FakeTable]) -> None:
        self.tables = tables


class _FakePage:
    def __init__(self, text: str, tables: list[_FakeTable]) -> None:
        self._text = text
        self.tables = tables
        self.number = 0

    def get_text(self, *args, **kwargs) -> str:
        return self._text

    def find_tables(self) -> _FakeTableFinder:
        return _FakeTableFinder(self.tables)


class TestIngestedChunk:
    """Test IngestedChunk dataclass."""

    def test_chunk_creation(self) -> None:
        chunk = IngestedChunk(
            text="hello world",
            file_name="doc.pdf",
            page_number=3,
            chunk_index=0,
            char_offset=0,
        )
        assert chunk.text == "hello world"
        assert chunk.file_name == "doc.pdf"
        assert chunk.page_number == 3
        assert chunk.chunk_index == 0
        assert chunk.char_offset == 0

    def test_chunk_with_none_page(self) -> None:
        chunk = IngestedChunk(
            text="text",
            file_name="doc.txt",
            page_number=None,
            chunk_index=1,
            char_offset=5,
        )
        assert chunk.page_number is None


class TestLocalDataIngestionText:
    """Test text file ingestion."""

    def test_ingest_text_file(self, tmp_path: Path, ingestion: LocalDataIngestion) -> None:
        text = "First sentence. Second sentence. Third sentence."
        file_path = tmp_path / "sample.txt"
        file_path.write_text(text, encoding="utf-8")

        chunks = ingestion.ingest_file(file_path)

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, IngestedChunk)
            assert chunk.file_name == "sample.txt"
            assert chunk.page_number is None
            assert chunk.text.strip() == chunk.text

    def test_ingest_text_nonexistent_file(self, ingestion: LocalDataIngestion) -> None:
        with pytest.raises(FileNotFoundError):
            ingestion.ingest_file(Path("nonexistent.txt"))

    def test_ingest_text_metadata_passthrough(
        self, tmp_path: Path, ingestion: LocalDataIngestion
    ) -> None:
        text = "Alpha beta gamma delta epsilon zeta eta theta."
        file_path = tmp_path / "notes.txt"
        file_path.write_text(text, encoding="utf-8")

        chunks = ingestion.ingest_file(file_path)
        documents = ingestion.to_documents(chunks)

        assert len(documents) == len(chunks)
        for chunk, doc in zip(chunks, documents):
            assert doc.text == chunk.text
            assert doc.metadata["file_name"] == chunk.file_name
            assert doc.metadata["page_number"] == chunk.page_number
            assert doc.metadata["chunk_index"] == chunk.chunk_index


class TestLocalDataIngestionPDF:
    """Test PDF ingestion with page-level metadata."""

    def test_ingest_pdf_pages(self, tmp_path: Path, ingestion: LocalDataIngestion) -> None:
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((50, 50), "Page one content about refunds.")
        page2 = doc.new_page()
        page2.insert_text((50, 50), "Page two content about shipping.")
        pdf_path = tmp_path / "policy.pdf"
        doc.save(str(pdf_path))
        doc.close()

        chunks = ingestion.ingest_file(pdf_path)

        assert len(chunks) > 0
        pages = {chunk.page_number for chunk in chunks}
        assert 1 in pages
        assert 2 in pages
        for chunk in chunks:
            assert chunk.file_name == "policy.pdf"
            assert chunk.page_number is not None

    def test_ingest_pdf_1_based_page_number(
        self, tmp_path: Path, ingestion: LocalDataIngestion
    ) -> None:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Only page content.")
        pdf_path = tmp_path / "single.pdf"
        doc.save(str(pdf_path))
        doc.close()

        chunks = ingestion.ingest_file(pdf_path)
        assert all(chunk.page_number == 1 for chunk in chunks)

    def test_ingest_pdf_to_documents_metadata(
        self, tmp_path: Path, ingestion: LocalDataIngestion
    ) -> None:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "PDF page content here.")
        pdf_path = tmp_path / "doc.pdf"
        doc.save(str(pdf_path))
        doc.close()

        chunks = ingestion.ingest_file(pdf_path)
        documents = ingestion.to_documents(chunks)

        for chunk, doc in zip(chunks, documents):
            assert doc.metadata["file_name"] == "doc.pdf"
            assert doc.metadata["page_number"] == chunk.page_number
            assert doc.metadata["chunk_index"] == chunk.chunk_index

    def test_ingest_unsupported_file_type(
        self, tmp_path: Path, ingestion: LocalDataIngestion
    ) -> None:
        file_path = tmp_path / "data.xyz"
        file_path.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingestion.ingest_file(file_path)

    def test_ingest_files_multiple(self, tmp_path: Path, ingestion: LocalDataIngestion) -> None:
        f1 = tmp_path / "a.txt"
        f1.write_text("Content of file A.", encoding="utf-8")
        f2 = tmp_path / "b.txt"
        f2.write_text("Content of file B.", encoding="utf-8")

        chunks = ingestion.ingest_files([f1, f2])

        file_names = {chunk.file_name for chunk in chunks}
        assert file_names == {"a.txt", "b.txt"}


class TestPDFTableExtraction:
    """Table extraction from PDF pages."""

    def test_extract_tables_renders_markdown(self, ingestion: LocalDataIngestion) -> None:
        rows = [["Item", "Price"], ["Apple", "1.00"], ["Banana", "0.50"]]
        page = _FakePage("", [_FakeTable(rows)])

        text = ingestion._extract_tables(page)

        assert "| Item | Price |" in text
        assert "| Apple | 1.00 |" in text
        assert "| Banana | 0.50 |" in text

    def test_extract_tables_no_tables(self, ingestion: LocalDataIngestion) -> None:
        page = _FakePage("", [])
        assert ingestion._extract_tables(page) == ""

    @pytest.mark.parametrize(
        "rows, must_contain",
        [
            ([["A", "B"], ["1"]], ["| A | B |"]),  # uneven row padded
            ([["X", None], ["1", "2"]], ["| X |", "| 1 | 2 |"]),  # None cell
            ([["a|b", "c"], ["1", "2"]], ["&#124;"]),  # pipe escaped
        ],
    )
    def test_table_to_markdown_edge_cases(
        self,
        rows: list[list[str | None]],
        must_contain: list[str],
        ingestion: LocalDataIngestion,
    ) -> None:
        md = ingestion._table_to_markdown(rows)
        for fragment in must_contain:
            assert fragment in md, f"{fragment!r} not in {md!r}"


class TestPDFScannedPages:
    """OCR fallback for scanned / image-only PDF pages."""

    def test_scanned_page_ocr_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        temp_config: Config,
    ) -> None:
        fake_pytesseract = SimpleNamespace(
            image_to_string=lambda image: "SCANNED INVOICE TOTAL: 42.00"
        )
        monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/tesseract")

        ingestion = LocalDataIngestion(temp_config)
        assert ingestion._ocr_available is True

        doc = fitz.open()
        doc.new_page()  # blank page: no digital text layer
        pdf_path = tmp_path / "scan.pdf"
        doc.save(str(pdf_path))
        doc.close()

        chunks = ingestion.ingest_file(pdf_path)

        assert chunks, "scanned page should yield chunks via OCR"
        assert any("SCANNED INVOICE TOTAL" in chunk.text for chunk in chunks)

    def test_scanned_page_without_ocr_yields_no_chunks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        temp_config: Config,
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        ingestion = LocalDataIngestion(temp_config)
        assert ingestion._ocr_available is False

        doc = fitz.open()
        doc.new_page()  # blank page: no digital text layer
        pdf_path = tmp_path / "scan.pdf"
        doc.save(str(pdf_path))
        doc.close()

        assert ingestion.ingest_file(pdf_path) == []
