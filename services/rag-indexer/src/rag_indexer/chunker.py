"""
Code-aware document chunker for the Nexus RAG indexer.

Splits source files into semantically meaningful chunks:
- Python files: split on ``def `` / ``class `` boundaries so each chunk
  starts at a function or class declaration.
- Markdown files: split on heading lines (``# `` / ``## ``).
- All other files: sliding window of 1500 chars with 200-char overlap.

Each chunk carries metadata for Pinecone (file path, chunk index, language).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""

    content: str
    source_file: str
    chunk_index: int
    language: str
    char_start: int = 0


# ─── Language detection ───────────────────────────────────────────────────────

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".go": "go",
    ".java": "java",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}

_WINDOW_SIZE = 1500
_OVERLAP = 200


def detect_language(path: Path) -> str:
    """Return the language string for a given file path."""
    return _EXTENSION_TO_LANGUAGE.get(path.suffix.lower(), "unknown")


# ─── Chunkers ─────────────────────────────────────────────────────────────────


def _chunk_python(text: str, source_file: str) -> list[Chunk]:
    """
    Split a Python file on ``def `` and ``class `` declaration boundaries.

    Each chunk starts at the beginning of a top-level or nested function/class
    definition so that the LLM receives complete callable units.
    """
    pattern = re.compile(r"(?=^(?:def |class |\s+def |\s+class ))", re.MULTILINE)
    parts = pattern.split(text)
    # Filter empty strings that can appear at the start of the split
    parts = [p for p in parts if p.strip()]
    if not parts:
        return _chunk_sliding_window(text, source_file, "python")
    return [
        Chunk(
            content=part,
            source_file=source_file,
            chunk_index=idx,
            language="python",
        )
        for idx, part in enumerate(parts)
    ]


def _chunk_markdown(text: str, source_file: str) -> list[Chunk]:
    """
    Split a Markdown file on heading lines (``# `` and ``## `` only).

    Keeps the heading line at the start of each chunk so the context
    is self-contained.
    """
    pattern = re.compile(r"(?=^#{1,2} )", re.MULTILINE)
    parts = pattern.split(text)
    parts = [p for p in parts if p.strip()]
    if not parts:
        return _chunk_sliding_window(text, source_file, "markdown")
    return [
        Chunk(
            content=part,
            source_file=source_file,
            chunk_index=idx,
            language="markdown",
        )
        for idx, part in enumerate(parts)
    ]


def _chunk_sliding_window(text: str, source_file: str, language: str) -> list[Chunk]:
    """
    Generic sliding-window chunker with overlap.

    Args:
        text:        Full file text.
        source_file: Path string for metadata.
        language:    Language label string.

    Returns:
        List of Chunk objects, each of at most ``_WINDOW_SIZE`` characters.
    """
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + _WINDOW_SIZE
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    content=chunk_text,
                    source_file=source_file,
                    chunk_index=idx,
                    language=language,
                    char_start=start,
                )
            )
            idx += 1
        start = end - _OVERLAP
        if start <= 0:
            break
    return chunks


# ─── Public API ───────────────────────────────────────────────────────────────


def chunk_file(path: Path, text: str) -> list[Chunk]:
    """
    Dispatch to the correct chunker based on file extension.

    Args:
        path: Path object pointing to the source file.
        text: Full text content of the file.

    Returns:
        Ordered list of Chunk objects.
    """
    language = detect_language(path)
    source_file = str(path)

    if language == "python":
        return _chunk_python(text, source_file)
    if language == "markdown":
        return _chunk_markdown(text, source_file)
    return _chunk_sliding_window(text, source_file, language)
