# RAG + Citation Grounding Chatbot --- Detailed Project Description

This document explains how the **RAG + Citation Grounding Chatbot**
works from start to finish. It is a general-purpose, from-scratch
Retrieval-Augmented Generation (RAG) application designed to answer
questions about different types of PDF and text documents. Answers are
supported by source citations and a receipts panel, allowing users to
check where the information came from.

------------------------------------------------------------------------

## 1. High-Level Architecture

The project follows a linear pipeline:

    User PDFs/TXTs

        │

        ▼

    Ingestion (PDF and text-file parsing → chunking → document-structure detection → metadata)

        │

        ▼

    Indexing (ChromaDB vector store + embeddings)

        │

        ▼

    Retrieval (Hybrid: vector search + BM25 + cross-encoder reranking)

        │

        ▼

    Generation (Ollama LLM with citation-aware system prompt)

        │

        ▼

    Citation Grounding (parse citations, validate against context, build receipts)

        │

        ▼

    Gradio UI (chat interface + receipts panel)

Each stage is handled by a dedicated Python module under
`src/rag_grounded_chatbot/`.

------------------------------------------------------------------------

## 2. Module Overview

| Module \| File \| Responsibility \|

\|--------\|------\|----------------\|

| Entry point \| `app.py` \| Launches the Gradio server on port 7860 \|

| Configuration \| `src/.../config.py` \| Central config: paths, models,
  chunk sizes, retrieval settings \|

| Ingestion \| `src/.../ingestion.py` \| PDF and text-file loading,
  per-chunk heading and section detection, metadata extraction, caching
  \|

| Index \| `src/.../index.py` \| ChromaDB vector store, build/load index
  \|

| Retrieval \| `src/.../retrieval.py` \| Hybrid retrieval: vector +
  BM25 + cross-encoder rerank, section-aware filtering \|

| Generation \| `src/.../generation.py` \| Ollama LLM with
  citation-aware system prompt, context building \|

| Citations \| `src/.../citations.py` \| Parse `[source: file.pdf#p.N]`
  citations, validate, build receipts \|

| Pipeline \| `src/.../pipeline.py` \| Orchestrates ingest → index →
  retrieve → generate → cite \|

| UI \| `src/.../ui.py` \| Gradio chat + receipts panel, dynamic model
  dropdown \|

------------------------------------------------------------------------

## 3. Step-by-Step Flow

### Step 1: Start Ollama and Pull a Model

Before running the app, you need Ollama running locally:

``` powershell

ollama serve

ollama pull llama3.2:3b
```

The app connects to Ollama at `http\://localhost:11434` by default
(configurable via `Config.ollama_host`).

### Step 2: Launch the Application

``` powershell

cd a:\Studies\AI\rag-grounded-chatbot

python app.py
```

This starts the Gradio server at `http\://localhost:7860`. The app
initializes:

-   A `Config` object with all settings
-   A `GroundedRAGPipeline` that wires ingestion, indexing, retrieval,
    generation, and citation grounding
-   A `GroundedChatbotUI` that builds the Gradio interface

### Step 3: Upload Documents

In the UI, click **Upload Documents** and select one or more PDF, TXT,
MD, or CSV files. The PDFs do not need to follow a research-paper
format; the chatbot is intended for general document question answering.
Click **Index Documents**.

#### What happens during ingestion (`ingestion.py`):

1.  **File discovery** --- Each uploaded file is read. PDFs are parsed
    with PyMuPDF (`pymupdf`/`fitz`); TXT/MD/CSV files are read as plain
    text.

2\. **Multi-pass PDF extraction** (per page):

   - **Pass 1 --- Digital text layer**: Extracts embedded text with
reading-order sorting (`sort=True`), handling most multi-column layouts.

   - **Pass 2 --- Tables**: Tables detected on the page are extracted
and rendered as Markdown, preserving row/column structure.

   - **Pass 3 --- OCR fallback**: If a page has too little digital text
(below `ocr_min_text_chars`, default 50), it is rendered as an image and
OCR'd using tesseract + pytesseract. OCR is silently disabled if
tesseract is not installed.

3\. **Chunking** --- Extracted page text is split into chunks using
`SentenceSplitter` with `chunk_size=1024` characters and
`chunk_overlap=128` characters. Overlap preserves context across chunk
boundaries.

4\. **Per-chunk section detection** --- Each chunk is scanned for
section headings using regex patterns (`_SECTION_PATTERNS`):

   - `references` --- matches "Reference", "References", "Bibliography",
"Works Cited" (singular and plural)

   - `related_work` --- matches "Related Work", "Related Works",
"Literature Review"

   - `abstract`, `introduction`, `conclusion`, `methodology`, `results`

   Numbered headings like "2.1 Installation" or "3.1 Feature Extraction"
are also detected via `_SECTION_NUMBER_RE`. Numbered bibliography
entries like "\[1\]" or "(1)" are handled using `_REFERENCE_ENTRY_RE` so
they are not mistaken for section headings when processing documents
that contain references.

5\. **Section state carry-forward** --- Within a single page, if a
section heading is detected, subsequent chunks on that page inherit that
section until a new heading appears.

6\. **Metadata assignment** --- Each `IngestedChunk` carries:

   - `text` --- the chunk text

   - `file_name` --- source file name

   - `page_number` --- PDF page number (1-based)

   - `chunk_index` --- index within the document

   - `char_offset` --- character offset within the page

   - `section` --- detected section type or document area (e.g.,
"introduction", "summary", "references", "body")

   - `section_number` --- numbered section ID if detected (e.g., "3.1")

   - `section_title` --- heading text if detected

   - `parent_section` --- parent section number (e.g., "3" for "3.1")

   - `document_type` --- "body", "reference", "table", or
"figure_caption"

7\. **Caching** --- Results are cached in
`data/storage/ingest_cache.json`, keyed by file path + modification
time + size. Unchanged files are skipped on re-ingestion.

8\. **Conversion to LlamaIndex Documents** --- Chunks are converted to
`llama_index.core.schema.Document` objects with all metadata attached.

### Step 4: Build the Index (`index.py`)

The `LocalVectorStore` class manages ChromaDB:

1.  **Embedding model** --- `BAAI/bge-small-en-v1.5` (HuggingFace) runs
    on CPU by default (configurable to CUDA).
2.  **ChromaDB** --- A persistent vector store at `data/chroma/` with a
    collection named `rag_chunks`.
3.  **Index build** --- Documents are embedded and stored in ChromaDB.
    The collection is cleared before each rebuild so deleted documents
    don't linger.
4.  **Load** --- An existing index can be loaded from disk on subsequent
    launches.

### Step 5: Retrieve Relevant Chunks (`retrieval.py`)

When a user asks a question, the `LocalRetriever` performs **hybrid
retrieval**:

1.  **Vector search** --- Semantic similarity search using the BGE
    embedding model. Returns `top_k=20` candidate chunks.

2.  **BM25 keyword search** --- Keyword-based search (`use_bm25=True` by
    default) that excels at matching exact names, acronyms, and
    technical terms (e.g., product names, technical terms, dates,
    identifiers, and author names).

3.  **Cross-encoder reranking** --- A cross-encoder model
    (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks the combined
    candidates to surface the most relevant passages. Final
    `rerank_top_k=15` chunks are passed to the LLM.

4.  **Document-structure-aware filtering** (confidence-aware):

    -   Detects document-section intent from the query when possible
        (for example, a question about a summary, conclusion,
        instructions, or references).
    -   Prioritizes chunks from the relevant detected section or
        document area.
    -   If enough high-priority chunks are not available, a controlled
        fallback expands to related sections and then the rest of the
        document.
    -   This helps keep answers focused while still allowing questions
        that require information from multiple parts of a document.

### Step 6: Generate the Answer (`generation.py`)

The `LocalGenerator` sends the retrieved chunks + user question to the
Ollama LLM (`llama3.2:3b` by default).

**System prompt** instructs the model to:

-   Answer using ONLY the provided context (no outside knowledge).

-   Cite every factual claim inline using `[source: file.pdf#p.N]`.

-   Never invent file names, page numbers, authors, or references.

-   Say "I don't have enough information..." if the context doesn't
    support the question.

-   Use section metadata when it is available to understand where
    information came from:

    -   **body**: the main content of the document
    -   **abstract / summary**: a short overview, when present
    -   **references / bibliography**: cited sources or bibliography
        entries, when present
    -   **other detected sections**: headings such as introduction,
        methods, results, discussion, instructions, or conclusion,
        depending on the document

**Context building** --- Each retrieved chunk is labeled with its source
file, page number, and section type before being fed to the LLM:

    [Source: document.pdf#p5 (section: introduction)]

    This section contains information extracted from the uploaded document...

**Generation parameters**:

-   `temperature=0.1` (low randomness for factual consistency)
-   `max_new_tokens=2048` (generous limit for long answers like
    reference lists)
-   `request_timeout=300.0` seconds

Streaming is supported via `stream_generate()`, which yields token
deltas for real-time UI updates.

### Step 7: Citation Grounding (`citations.py`)

The `CitationGrounder` parses the model's answer for citation markers
like `[source: paper.pdf#p5]`:

1.  **Regex extraction** --- `_CITATION_RE` matches citations in two
    formats: `[source: file.pdf#p3]` and `[source: file.pdf#p.3]`.
2.  **Validation** --- Each cited source is checked against the
    retrieved context. If a file/page doesn't exist in the retrieved
    chunks, the receipt is marked `valid=False` (likely hallucination).
3.  **Deduplication** --- Identical citations are collapsed into one
    receipt.
4.  **Quote extraction** --- A representative snippet (up to 300 chars)
    is pulled from the matching chunk.
5.  **Receipt output** --- Each receipt contains: file name, page
    number, quote, citation string, and validity flag.

Inline citation markers are stripped from the final answer text before
display, but the receipts panel shows them separately.

### Step 8: Display in Gradio UI (`ui.py`)

The interface has three columns:

-   **Left column** --- Status textbox, file upload, "Index Documents"
    button, "Clear All Documents" button, "Clear Chat" button, model
    dropdown (dynamically populated from Ollama's `/api/tags` endpoint).
-   **Center column** --- Chatbot showing the conversation with answers.
-   **Right column** --- Receipts panel showing one card per cited
    source: file name, page number, citation, quoted snippet, and
    validity warnings.

**Streaming chat** --- As tokens arrive from the LLM, the answer updates
in real time. Once generation completes, citations are grounded and
receipts are rendered.

------------------------------------------------------------------------

## 4. Key Design Decisions

| Decision \| Rationale \|

\|----------\|-----------\|

| **Per-chunk heading and section detection** \| Detecting headings
  within chunks helps identify document structure and reduces incorrect
  section labeling across different PDF layouts \|

| **Hybrid retrieval (BM25 + vector)** \| BM25 matches exact
  keywords/names; vector search matches meaning. Together they cover
  both. \|

| **Cross-encoder reranking** \| Moves the most relevant passages to the
  top before sending to the LLM \|

| **Confidence-aware section filtering** \| Prioritizes the most
  relevant detected document area but falls back to other parts of the
  document when needed \|

| **Section-aware guidance in system prompt** \| Helps the model
  interpret retrieved passages according to their document context \|

| **Receipt validation** \| Catches hallucinated citations by checking
  against retrieved context \|

| **File-based ingestion cache** \| Avoids re-parsing unchanged PDFs on
  every launch \|

------------------------------------------------------------------------

## 5. Data Flow Example

**User asks:** "What are the main points covered in this document?"

1.  **Ingestion** (already done): The PDF is parsed and divided into
    chunks. Recognizable headings and document areas are stored as
    metadata where available.
2.  **Retrieval**: The query is matched against the indexed chunks. BM25
    finds important exact terms, vector search finds semantically
    related content, and the reranker selects the most relevant
    passages.
3.  **Generation**: The LLM receives the retrieved context with source
    file, page number, and any detected section information. It prepares
    an answer using the document content and includes source citations.
4.  **Citation grounding**: Citations are parsed and checked against the
    retrieved context. Receipts are built with supporting quotes.
5.  **UI**: The answer is displayed in chat, while the receipts panel
    lists the supporting file, page number, citation, and quoted
    snippet.

------------------------------------------------------------------------

## 6. Configuration (`config.py`)

Key settings and their defaults:

| Setting \| Default \| Description \|

\|---------\|---------\|-------------\|

| `embedding_model` \| `BAAI/bge-small-en-v1.5` \| HuggingFace embedding
  model \|

| `rerank_model` \| `cross-encoder/ms-marco-MiniLM-L-6-v2` \|
  Cross-encoder for reranking \|

| `llm_model` \| `llama3.2:3b` \| Ollama model for generation \|

| `max_new_tokens` \| `2048` \| Max tokens in LLM output \|

| `chunk_size` \| `1024` \| Characters per chunk \|

| `chunk_overlap` \| `128` \| Overlap between chunks \|

| `top_k` \| `20` \| Candidates before reranking \|

| `rerank_top_k` \| `15` \| Final chunks to LLM \|

| `use_bm25` \| `True` \| Enable BM25 keyword search \|

| `use_query_expansion` \| `False` \| Enable query expansion \|

| `device` \| `cpu` \| `cpu` or `cuda` \|

| `ocr_enabled` \| `True` \| Attempt OCR on sparse pages \|

| `citation_format` \| `[source: {file}#p{page}]` \| Citation format
  string \|

------------------------------------------------------------------------

## 7. Testing

Run the test suite from the project root:

``` powershell

pytest
```

Tests cover all backend modules: config, ingestion, index, retrieval,
generation, citations, and pipeline. An integration test exercises the
full flow with mocked LLM. Gradio UI and real Ollama are not tested
(they require a running server).

------------------------------------------------------------------------

## 8. Limitations

-   **OCR requires tesseract** --- Scanned PDFs without tesseract
    installed may produce no extractable text for those pages.
-   **No persistent chat history** --- Conversations reset on app
    restart.
-   **Single collection** --- All documents share one ChromaDB
    collection; no multi-tenant isolation.
-   **Model-dependent accuracy** --- Answer quality depends on the
    Ollama model's ability to follow citation instructions.

------------------------------------------------------------------------

## 9. Project Structure

    rag-grounded-chatbot/

    ├── app.py                          # Gradio entry point

    ├── pyproject.toml                  # Dependencies and build config

    ├── README.md                       # Project overview and setup

    ├── PROJECT_DESCRIPTION.md          # This file

    ├── TESTING.md                      # Testing instructions

    ├── .gitignore                      # Git ignore rules

    ├── data/

    │   ├── documents/                  # Drop your PDFs and supported text documents here (user-added)

    │   ├── storage/                    # Ingestion cache (gitignored)

    │   └── chroma/                     # ChromaDB persistence (gitignored)

    ├── src/

    │   └── rag_grounded_chatbot/

    │       ├── __init__.py

    │       ├── config.py               # Central configuration

    │       ├── ingestion.py            # PDF and text-file loading, chunking, document-structure detection

    │       ├── index.py                # ChromaDB vector store

    │       ├── retrieval.py            # Hybrid retrieval + reranking + section filtering

    │       ├── generation.py           # Ollama LLM with citation prompt

    │       ├── citations.py            # Citation parsing + receipt building

    │       ├── pipeline.py             # End-to-end orchestration

    │       └── ui.py                   # Gradio chat + receipts panel

    └── tests/

        ├── test_*.py                   # Unit tests for each module

        └── test_integration.py         # Full pipeline integration test

------------------------------------------------------------------------

*End of project description.*
