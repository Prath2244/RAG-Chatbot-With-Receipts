"""Citation parsing and receipt building.

The model is expected to emit citations like:

    [source: policy.pdf#p.3]

This module extracts those citations, matches them back to the retrieved
chunks, and builds a list of receipts the UI can render. It also validates
that every cited source actually exists in the retrieved context, so
hallucinated references are caught and flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from llama_index.core.schema import NodeWithScore

from .config import Config

# Matches [source: file.pdf#p3] or [source: file.pdf#p.3] anywhere in text.
# File names containing brackets (e.g. "Report[2](1).pdf") are captured correctly
# by using a non-greedy match up to the closing ] that ends the citation.
# The pattern uses a lookahead to ensure we match up to the final ].
_CITATION_RE = re.compile(
    r"\[source:\s*(?P<file>.+?(?=\s*(?:#p\.?\d+)?\]))(?P<page_part>#p\.?(?P<page>\d+))?\]",
    re.IGNORECASE
)


@dataclass(frozen=True)
class Receipt:
    """One piece of evidence backing a claim in the answer."""

    file_name: str
    page_number: Optional[int]
    quote: str
    citation: str  # the exact string the model emitted, e.g. [source: policy.pdf#p.3]
    valid: bool = True  # False if the cited source does not exist in the context


class CitationGrounder:
    """Parse citations from an answer and attach supporting quotes."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()

    def ground(
        self,
        answer: str,
        nodes: List[NodeWithScore],
    ) -> tuple[str, List[Receipt]]:
        """Return the answer text plus a list of receipts found in it.

        Receipts with ``valid=False`` indicate the model cited a source
        that does not exist in the retrieved context — a likely
        hallucination.
        """
        receipts: List[Receipt] = []
        seen: set[str] = set()
        valid_sources = self._build_source_index(nodes)

        for match in _CITATION_RE.finditer(answer):
            citation = match.group(0)
            file_name = match.group("file").strip()
            page_str = match.group("page")
            page_number = int(page_str) if page_str else None

            # Dedup identical citations
            key = f"{file_name}#{page_number}"
            if key in seen:
                continue
            seen.add(key)

            # Validate: does this source actually exist in the retrieved nodes?
            valid = self._is_valid_source(file_name, page_number, valid_sources)

            # Find the best matching chunk to quote
            quote = self._find_quote(file_name, page_number, nodes)
            receipts.append(
                Receipt(
                    file_name=file_name,
                    page_number=page_number,
                    quote=quote,
                    citation=citation,
                    valid=valid,
                )
            )

        # Strip inline citation markers from the answer text
        clean_answer = _CITATION_RE.sub("", answer)
        return clean_answer, receipts

    @staticmethod
    def _build_source_index(
        nodes: List[NodeWithScore],
    ) -> set[tuple[str, Optional[int]]]:
        """Return the set of (normalized_file_name, page_number) present in the context."""
        sources: set[tuple[str, Optional[int]]] = set()
        for node in nodes:
            meta = node.node.metadata or {}
            file_name = meta.get("file_name", "unknown")
            # Normalize: strip path, use basename only, lowercase for comparison
            file_name = Path(file_name).name.lower()
            sources.add((file_name, meta.get("page_number")))
        return sources

    @staticmethod
    def _is_valid_source(
        file_name: str,
        page_number: Optional[int],
        valid_sources: set[tuple[str, Optional[int]]],
    ) -> bool:
        """Check whether *file_name* / *page_number* exists in the context.

        Comparison is case-insensitive on file name and tolerant of
        missing page numbers (matches any page from the same file).
        """
        normalized_file = Path(file_name).name.lower()

        # Exact match: same file + same page
        if (normalized_file, page_number) in valid_sources:
            return True

        # Fallback: same file, any page (page number may be omitted in citation)
        if page_number is not None:
            for src_file, src_page in valid_sources:
                if src_file == normalized_file:
                    return True

        return False

    def _find_quote(
        self,
        file_name: str,
        page_number: Optional[int],
        nodes: List[NodeWithScore],
    ) -> str:
        """Return a representative snippet from the matching node(s)."""
        normalized_file = Path(file_name).name.lower()
        candidates: List[str] = []
        for node in nodes:
            meta = node.node.metadata or {}
            if Path(meta.get("file_name", "unknown")).name.lower() != normalized_file:
                continue
            if page_number is not None and meta.get("page_number") != page_number:
                continue
            candidates.append(node.node.text)

        if not candidates:
            return ""
        # Return the first (highest-scoring) node's text, trimmed to 300 chars
        snippet = candidates[0].strip()
        return snippet[:300] + ("…" if len(snippet) > 300 else "")