"""Shared pytest fixtures for the RAG + Citation Grounding chatbot."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import fitz  # PyMuPDF
import pytest
from llama_index.core import Settings
from llama_index.core.llms.mock import MockLLM

from rag_grounded_chatbot.citations import CitationGrounder
from rag_grounded_chatbot.config import Config
from rag_grounded_chatbot.generation import LocalGenerator
from rag_grounded_chatbot.index import LocalVectorStore
from rag_grounded_chatbot.ingestion import LocalDataIngestion
from rag_grounded_chatbot.pipeline import GroundedRAGPipeline
from rag_grounded_chatbot.retrieval import LocalRetriever


@pytest.fixture
def temp_config(tmp_path: Path) -> Config:
    """Return a Config whose directories live under tmp_path."""
    return Config(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        documents_dir=tmp_path / "data" / "documents",
        storage_dir=tmp_path / "data" / "storage",
        chroma_dir=tmp_path / "data" / "chroma",
        chunk_size=200,
        chunk_overlap=20,
        top_k=5,
        rerank_top_k=3,
    )


@pytest.fixture
def sample_text() -> str:
    """Small deterministic text document for chunking tests."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco. "
    )


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """Generate a minimal two-page PDF in memory using PyMuPDF."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page one content about refunds and returns.")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page two content about shipping policies.")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture(autouse=True)
def _testing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set IS_TESTING so llama-index uses MockEmbedding in tests."""
    monkeypatch.setenv("IS_TESTING", "1")
    # Patch resolve_llm to return MockLLM so Settings.llm assignment doesn't
    # fail validation on MagicMocks used in tests.
    from llama_index.core import settings as _settings_mod
    from llama_index.core.llms.mock import MockLLM as _MockLLM

    def _resolve_llm(llm=None, callback_manager=None) -> _MockLLM:
        return _MockLLM()

    monkeypatch.setattr(_settings_mod, "resolve_llm", _resolve_llm)


@pytest.fixture
def mock_ollama(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock Ollama used by generation module so no real server is called."""
    mock = MagicMock()
    response = MagicMock()
    response.text = "This is a mocked answer. [source: policy.txt#p1]"
    response.__str__ = lambda self: self.text
    mock.complete.return_value = response
    monkeypatch.setattr("rag_grounded_chatbot.generation.Ollama", lambda *args, **kwargs: mock)
    return mock


@pytest.fixture
def mock_reranker(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock SentenceTransformerRerank to avoid downloading models."""
    mock = MagicMock()
    # postprocess_nodes returns nodes unchanged (no reranking)
    mock.postprocess_nodes.side_effect = lambda nodes, **kwargs: nodes
    monkeypatch.setattr(
        "rag_grounded_chatbot.retrieval.SentenceTransformerRerank",
        lambda *args, **kwargs: mock,
    )
    return mock


@pytest.fixture
def retriever(temp_config: Config, mock_reranker: MagicMock) -> LocalRetriever:
    """Return a LocalRetriever instance with mocked reranker."""
    return LocalRetriever(temp_config)


@pytest.fixture
def mocked_generator(temp_config: Config, mock_ollama: MagicMock) -> LocalGenerator:
    """Return a LocalGenerator with Ollama mocked out."""
    generator = LocalGenerator(temp_config)
    Settings.llm = MockLLM()
    return generator


@pytest.fixture
def ingestion(temp_config: Config) -> LocalDataIngestion:
    """Return a LocalDataIngestion instance."""
    return LocalDataIngestion(temp_config)


@pytest.fixture
def vector_store(temp_config: Config) -> LocalVectorStore:
    """Return a LocalVectorStore using a temp ChromaDB directory."""
    return LocalVectorStore(temp_config)


@pytest.fixture
def grounder(temp_config: Config) -> CitationGrounder:
    """Return a CitationGrounder instance."""
    return CitationGrounder(temp_config)


@pytest.fixture
def pipeline(
    temp_config: Config, mock_ollama: MagicMock, mock_reranker: MagicMock
) -> GroundedRAGPipeline:
    """Return a GroundedRAGPipeline with Ollama and reranker mocked."""
    return GroundedRAGPipeline(temp_config)
