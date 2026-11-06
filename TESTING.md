# Testing

Run the backend test suite with pytest from the project root:

```bash
pytest
```

## What is tested

Unit tests cover every backend module under `src/rag_grounded_chatbot/`:

| Module | Test file | What is verified |
|--------|-----------|------------------|
| `config.py` | `tests/test_config.py` | Defaults, `__post_init__` directory creation |
| `ingestion.py` | `tests/test_ingestion.py` | TXT chunking, PDF page-level metadata (1-based), `to_documents` passthrough |
| `index.py` | `tests/test_index.py` | `build_index` + `load_index` round-trip using temp ChromaDB dir |
| `retrieval.py` | `tests/test_retrieval.py` | Top-k nodes with provenance metadata, empty results |
| `generation.py` | `tests/test_generation.py` | Context labeling, prompt formatting (mocked Ollama) |
| `citations.py` | `tests/test_citations.py` | Regex citation parsing, dedup, `Receipt`, missing page numbers |
| `pipeline.py` | `tests/test_pipeline.py` | "No documents indexed" message, happy-path orchestration |

An integration test (`tests/test_integration.py`) exercises the full backend flow:
ingest sample docs → build index → retrieve → generate (mocked LLM) → ground citations,
stopping before the Gradio UI.

## What is NOT tested

- Gradio UI interactions or rendering
- Browser/website launch
- Real Ollama server or any network endpoint
- Real PDFs on disk (all test PDFs are generated in-memory with PyMuPDF)

## Requirements

- `pytest` and `ruff` are dev dependencies (see `pyproject.toml`).
- Install the package in editable mode: `pip install -e .`
- Run from the project root: `pytest`

## Linting

```bash
ruff check tests/
ruff format --check tests/
```
