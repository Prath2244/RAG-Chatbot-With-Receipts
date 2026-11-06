# 🧠 RAG + Citation Grounding — Answers with Receipts

[![Python](https://img.shields.io/badge?label=Python&message=3.14&color=3776AB&style=flat-square)](https://www.python.org/)

[![Gradio](https://img.shields.io/badge?label=Gradio&message=4.x&color=1F6FEB&style=flat-square)](https://www.gradio.app/)

[![Ollama](https://img.shields.io/badge?label=Ollama&message=Local%20LLM&color=E85C3F&style=flat-square)](https://ollama.com/)

[![ChromaDB](https://img.shields.io/badge?label=ChromaDB&message=Vector%20Store&color=4A148C&style=flat-square)](https://www.trychroma.com/)

[![LlamaIndex](https://img.shields.io/badge?label=LlamaIndex&message=RAG%20Framework&color=005CA9&style=flat-square)](https://docs.llamaindex.ai/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> A from-scratch RAG chatbot where **every answer is verifiable**. Unlike a basic RAG bot that returns a bare answer string, this project ingests documents with page-level provenance, prompts the model to cite every claim inline, parses those citations, and renders answer + receipts side by side in Gradio so users can verify each answer against the original document.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Core Workflows](#-core-workflows)
- [How It Works](#-how-it-works)
- [Configuration Reference](#-configuration-reference)
- [Limitations](#-limitations)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)
- [License](#-license)
- [Author](#-author)

---

## 🧾 Overview

**RAG + Citation Grounding** is a retrieval-augmented generation chatbot built from scratch that answers questions about PDFs (and other text documents) with **citation-grounded receipts**. Every claim in the model's answer is backed by an inline citation of the form `[source: file.pdf#p.N]`, and a Receipts panel lists each cited source with its page number and quoted snippet.

---

## ✨ Features

### 📄 Document Ingestion
- **Multi-pass PDF extraction** — Each page goes through up to three passes: digital text layer (reading-order sorted), table extraction (preserved as Markdown), and OCR fallback for scanned/image-only pages.
- **Per-chunk section detection** — Chunks are classified by section type (body, reference, related_work, table, figure_caption) using topic-based heading patterns (e.g., "REFERENCE", "RELATED WORKS", "Bibliography") and numbered heading detection.
- **Page-level provenance** — Every chunk retains its source filename and page number — never merged into an anonymous blob.
- **Section-aware metadata** — Each chunk stores `section_number`, `section_title`, `parent_section`, and `document_type` for precise section-scoped retrieval.

### 🔍 Hybrid Retrieval
- **Vector search** — Semantic retrieval using BGE embeddings stored in ChromaDB.
- **BM25 keyword search** — Complements vector search for keyword-heavy queries (enabled by default).
- **Cross-encoder reranking** — Re-ranks top candidates for better relevance before passing to the LLM.
- **Confidence-aware section filtering** — Prioritizes chunks from the target section detected in the query (e.g., "references") while falling back gracefully when insufficient matches exist.
- **Section-number awareness** — Extracts section numbers from queries (e.g., "section 3") and matches them against chunk metadata.

### 🤖 Generation with Citations
- **Ollama LLM integration** — Runs local LLMs (default: `llama3.2:3b`) via Ollama at `http://localhost:11434`.
- **Citation-prompted generation** — System prompt instructs the model to cite every claim using `[source: file.pdf#p.N]` format.
- **Section-type-aware context** — Sources are labeled with their section type (body vs. reference vs. related_work) so the model can distinguish between them.
- **Configurable output length** — `max_new_tokens` default of 2048 supports long list-type answers without truncation.

### 🧾 Receipts & Verification
- **Citation parsing** — Extracts `[source: file.pdf#p.N]` citations from model answers.
- **Receipts panel** — Gradio UI shows each cited source with file name, page number, and quoted snippet alongside the answer.
- **Citation validation** — Validates citations against retrieved context to prevent hallucinated references.

### 🖥️ User Interface
- **Gradio web app** — Runs on `http://localhost:7860`.
- **Dynamic model dropdown** — Fetches installed Ollama models automatically at startup.
- **Document upload & indexing** — Upload PDFs/TXTs and click **Index Documents** to build the vector store.
- **Side-by-side layout** — Answer on the left, Receipts panel on the right.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.14 |
| UI Framework | Gradio 4.x |
| RAG Framework | llama-index 0.10 |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector Store | ChromaDB |
| LLM Backend | Ollama (local models) |
| PDF Parsing | PyMuPDF (fitz) |
| OCR | Tesseract + pytesseract (optional) |
| Config | pydantic-compatible dataclass |
| Testing | pytest |
| Linting | ruff |

---

## 🧱 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Gradio UI                            │
│  (Answer panel + Receipts panel, dynamic model dropdown)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      pipeline.py                            │
│          Orchestrates: ingest → index → retrieve →          │
│          generate → cite → render receipts                  │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  ingestion.py │   │  index.py     │   │  ui.py        │
│  PDF/TXT load │   │  ChromaDB     │   │  Gradio chat  │
│  per-chunk    │   │  vector store │   │  receipts UI  │
│  section det. │   │  persistence  │   │  model fetch  │
└───────────────┘   └───────────────┘   └───────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ retrieval.py  │   │ generation.py │   │ citations.py  │
│ Vector + BM25 │   │ Ollama LLM    │   │ Citation      │
│ Reranking     │   │ Citation prompt│   │ parsing +     │
│ Section filter│   │ Section-aware │   │ receipt building│
└───────────────┘   └───────────────┘   └───────────────┘
```

Data flows from the UI into the pipeline, which calls ingestion to load and chunk documents, indexes them in ChromaDB, then on each query performs hybrid retrieval (vector + BM25 + rerank), generates a cited answer via Ollama, parses citations, and renders receipts.

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- Ollama installed and running ([https://ollama.com](https://ollama.com))
- ~1 GB disk space

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-grounded-chatbot.git
cd rag-grounded-chatbot
```

### 2. Create virtual environment
```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e .

# Optional: OCR support for scanned PDFs
pip install ".[ocr]"
```

### 4. Pull an Ollama model
```bash
# Start Ollama server (in a separate terminal)
ollama serve

# Pull the default model
ollama pull llama3.2:3b
```

---

## 🚀 Running the Application

### 1. Start Ollama
```bash
ollama serve
```

### 2. Launch the Gradio app
```bash
python app.py
```
Server runs on `http://localhost:7860`.

### 3. Open the application
Navigate to:
```
http://localhost:7860
```

### 4. Upload and index documents
- Click **Upload** to select one or more PDF/TXT files.
- Click **Index Documents** to build the vector store (ChromaDB persistence under `data/chroma/`).

### 5. Ask a question
- Type your question in the chat input.
- The answer appears with inline citations like `[source: paper.pdf#p.3]`.
- The **Receipts** panel on the right lists each cited source with its page number and quoted snippet.



## 🔄 Core Workflows

| Actor | Action | Description |
|-------|--------|-------------|
| User | Upload Documents | Select PDF/TXT files via Gradio file uploader. |
| User | Index Documents | Triggers ingestion → chunking → section detection → ChromaDB indexing. |
| User | Ask Question | Submits a natural-language question about the indexed documents. |
| System | Hybrid Retrieval | Vector + BM25 search, cross-encoder rerank, section-aware filtering. |
| System | Generate Answer | Ollama LLM produces answer with inline `[source: file.pdf#p.N]` citations. |
| System | Build Receipts | Parses citations, validates against context, builds receipt entries. |
| User | Verify | Reviews answer and Receipts panel side by side. |

---

## ⚙️ How It Works

### Step 1 — Document Ingestion (`ingestion.py`)
Each uploaded PDF/TXT is processed page by page:
1. **Text extraction** — PyMuPDF extracts digital text with reading-order sorting.
2. **Table detection** — Tables are extracted and rendered as Markdown.
3. **OCR fallback** — Image-only pages are OCR'd (requires tesseract).
4. **Chunking** — Pages are split into chunks of `chunk_size` (1024 chars) with `chunk_overlap` (128 chars).
5. **Section detection** — Each chunk is classified by section type using topic-based heading patterns (`REFERENCE`, `RELATED WORKS`, `BIBLIOGRAPHY`, etc.) and numbered heading regex. Section state carries forward across chunks on the same page.
6. **Metadata tagging** — Each chunk stores `file_name`, `page_number`, `section_number`, `section_title`, `parent_section`, and `document_type`.

### Step 2 — Indexing (`index.py`)
Chunks and their metadata are embedded using BGE-small-en-v1.5 and stored in ChromaDB under `data/chroma/`. BM25 indexes are built for keyword search.

### Step 3 — Query Understanding (`retrieval.py`)
When a question is asked:
1. **Section intent detection** — The query is scanned for section keywords (e.g., "references", "related works") and section numbers (e.g., "section 3").
2. **Hybrid retrieval** — Vector search (`top_k=20`) and BM25 retrieve candidate chunks.
3. **Section-aware filtering** — Chunks are prioritized by section match: exact section number > section prefix > section-type-only > all nodes.
4. **Cross-encoder reranking** — Top candidates are reranked; `rerank_top_k=15` chunks pass to the LLM.

### Step 4 — Generation (`generation.py`)
The reranked chunks are formatted into context with section-type labels and sent to Ollama with a system prompt that instructs the model to:
- Answer using only the provided context.
- Cite every claim with `[source: file.pdf#p.N]`.
- Distinguish between body text, references, and related works sections.

### Step 5 — Citation Parsing & Receipts (`citations.py` + `ui.py`)
- Citations are parsed from the model's answer.
- Each citation is validated against retrieved context.
- The Receipts panel displays source file, page number, and quoted snippet for each claim.

---

## ⚙️ Configuration Reference

All settings live in `src/rag_grounded_chatbot/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_model` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `rerank_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `llm_model` | `llama3.2:3b` | Ollama model name |
| `max_new_tokens` | `2048` | Max tokens for LLM output |
| `ollama_host` | `http://localhost:11434` | Ollama API endpoint |
| `chunk_size` | `1024` | Characters per chunk |
| `chunk_overlap` | `128` | Overlap between chunks |
| `top_k` | `20` | Candidates before reranking |
| `rerank_top_k` | `15` | Final chunks to LLM |
| `use_bm25` | `True` | Enable BM25 keyword search |
| `use_query_expansion` | `False` | Enable query expansion |
| `device` | `cpu` | `cpu` or `cuda` |
| `ocr_enabled` | `True` | Attempt OCR on sparse pages |
| `ocr_min_text_chars` | `50` | Min text chars before OCR |
| `ocr_dpi` | `200` | OCR render resolution |
| `citation_format` | `[source: {file}#p{page}]` | Citation string format |

---

## ⚠️ Limitations
- **No persistent chat history** — Conversations are not saved across sessions.
- **OCR dependency** — Scanned/image-only PDFs require tesseract installed system-wide; otherwise OCR is silently skipped.
- **Local-only LLM** — Requires Ollama running locally; no cloud LLM provider support.
- **Single document type focus** — Optimized for research paper PDFs; other formats (DOCX, HTML, EPUB) not yet supported.
- **No section numbers in some PDFs** — Topic-based section detection (REFERENCE, RELATED WORKS) works for unnumbered PDFs, but section-number queries (e.g., "section 3") require numbered headings.

---

## 🗺️ Roadmap

- [x] Per-chunk section detection with topic and numbered headings
- [x] Hybrid retrieval (vector + BM25 + cross-encoder rerank)
- [x] Citation-prompted generation with receipts panel
- [x] Dynamic Ollama model dropdown
- [x] Section-aware confidence filtering
- [x] OCR fallback for scanned PDFs
- [ ] Persist chat history across sessions
- [ ] Highlight cited spans inside answer text
- [ ] "Jump to source" button opening PDF at page
- [ ] Support DOCX, HTML, EPUB formats
- [ ] Evaluation harness (RAGAS / faithfulness + citation recall)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate comments.

---

## ⚠️ Disclaimer

This tool is for research and information retrieval purposes only. Answers are generated by a local LLM and grounded in cited document excerpts, but citations may be incomplete or incorrect. Always verify critical claims against the original document. This tool does not replace professional judgment or expert review.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Prathmesh** – [GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourusername)

---

**Made with ❤️ for verifiable, citation-grounded Q&A.**
