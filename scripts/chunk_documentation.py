#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


MARKER_RE = re.compile(r"^=== (.+) ===$")
IMAGE_LINE_RE = re.compile(r"^\s*!\[.*\]\(.*\)\s*(\{.*\})?\s*$")
WIDTH_ATTR_RE = re.compile(r"\{[^}]*width=[^}]*\}")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
RST_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+(toctree|automodule|autoclass|autoattribute)\b")
RST_OPTION_RE = re.compile(r"^\s*:[a-zA-Z0-9_-]+:\s*")
EMPH_RE = re.compile(r"(\*{1,2})([^*]+)\1")   # *text* or **text**
UNDERLINE_RE = re.compile(r"^[-=]{3,}\s*$")   # ==== or ---- lines

ALLOWLIST_PATTERNS: List[str] = [
    "orange3-doc-visual-programming/source/widgets/**/*.md",
    "orange3-doc-visual-programming/source/exporting-*/index.md",
    "orange3-doc-visual-programming/source/report/index.md",
    "orange3-doc-visual-programming/source/learners-as-scorers/index.md",
    "orange3-doc-visual-programming/source/building-workflows/index.md",
    "orange3-doc-visual-programming/source/loading-your-data/index.md",
    "orange3-text/doc/widgets/*.md",
    "orange3-bioinformatics/doc/widgets/*.md",
    "orange3-survival-analysis/doc/widgets/*.md",
    "orange3-single-cell/doc/widgets/*.md",
    "orange3/Orange/distance/distances.md",
    "orange3/doc/data-mining-library/source/tutorial/*.rst",
]


@dataclass
class Section:
    path: str
    text: str


def iter_sections(raw_text: str) -> Iterable[Section]:
    current_path: str | None = None
    current_lines: list[str] = []

    for line in raw_text.splitlines():
        m = MARKER_RE.match(line)
        if m:
            if current_path is not None:
                yield Section(path=current_path, text="\n".join(current_lines).rstrip())
            current_path = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_path is not None:
        yield Section(path=current_path, text="\n".join(current_lines).rstrip())


def is_allowed(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in ALLOWLIST_PATTERNS)


def clean_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if IMAGE_LINE_RE.match(line):
            continue
        stripped = line.strip()
        if UNDERLINE_RE.match(stripped):
            # drop pure underline lines
            continue
        line = WIDTH_ATTR_RE.sub("", line)
        line = MD_LINK_RE.sub(lambda m: m.group(1), line)
        if RST_DIRECTIVE_RE.match(line):
            continue
        if RST_OPTION_RE.match(line):
            continue
        # remove simple *emphasis* / **bold** markers
        line = EMPH_RE.sub(lambda m: m.group(2), line)
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_title(path: str, cleaned_text: str) -> str:
    lines = [ln for ln in cleaned_text.splitlines() if ln.strip()]
    if not lines:
        # fall back to filename
        return Path(path).stem

    # Markdown ATX heading: # Title
    first = lines[0]
    if first.lstrip().startswith("#"):
        return first.lstrip("#").strip()

    # Setext-style: Title\n=== or Title\n---
    if len(lines) >= 2:
        underline = lines[1]
        if re.fullmatch(r"=+", underline.strip()) or re.fullmatch(r"-+", underline.strip()):
            return first.strip()

    return first.strip()


def slugify_path(path: str) -> str:
    lower = path.lower()
    prefix = "doc"
    if lower.startswith("orange3-doc-visual-programming/"):
        prefix = "vp"
    elif lower.startswith("orange3-text/"):
        prefix = "text"
    elif lower.startswith("orange3-bioinformatics/"):
        prefix = "bio"
    elif lower.startswith("orange3-survival-analysis/"):
        prefix = "surv"
    elif lower.startswith("orange3-single-cell/"):
        prefix = "single"
    elif lower.startswith("orange3/"):
        prefix = "core"

    name = Path(path).stem
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{prefix}-{slug}" if slug else prefix


def tokenize(text: str) -> list[str]:
    # Simple whitespace tokenization; for these docs 1 word ~ 1 token.
    return text.split()


def split_by_headings(cleaned_text: str) -> list[str]:
    """
    Split a long document into sections by approximate headings.

    We treat a line as a heading if:
      - It starts with '#' (Markdown ATX), OR
      - It is a non-empty line following a blank line, with no leading
        list/numbering markers and reasonably short length.
    """
    lines = cleaned_text.splitlines()
    if not lines:
        return []

    heading_idxs: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading_idxs.append(i)
            continue
        if i == 0 or not lines[i - 1].strip():
            # no obvious list markers, short-ish line: treat as heading
            if not stripped.startswith(("-", "*", "+")) and not stripped[:2].isdigit():
                if len(stripped) <= 80:
                    heading_idxs.append(i)

    if not heading_idxs:
        return [cleaned_text]

    # ensure first line is a start
    if heading_idxs[0] != 0:
        heading_idxs.insert(0, 0)

    sections: list[str] = []
    for start, end in zip(heading_idxs, heading_idxs[1:] + [len(lines)]):
        chunk_lines = lines[start:end]
        section = "\n".join(chunk_lines).strip()
        if section:
            sections.append(section)

    return sections or [cleaned_text]


def chunk_by_size(tokens: list[str], max_tokens: int, overlap: int) -> list[str]:
    if not tokens:
        return []
    chunks: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        end = min(n, i + max_tokens)
        chunk_tokens = tokens[i:end]
        chunks.append(" ".join(chunk_tokens))
        if end >= n:
            break
        i = max(0, end - overlap)
    return chunks


def split_into_chunks(cleaned_text: str) -> list[str]:
    """
    Apply the requested chunking rules:

      - if chunk < 300 tokens: keep as-is (no overlap)
      - if 300–600 tokens: split with 30 token overlap
      - if >600 tokens: split by headings first, then apply above per section
    """
    top_tokens = tokenize(cleaned_text)
    n_tokens = len(top_tokens)

    if n_tokens == 0:
        return []

    # helper to chunk a single logical section
    def chunk_section(text: str) -> list[str]:
        toks = tokenize(text)
        n = len(toks)
        if n == 0:
            return []
        if n < 300:
            return [text]
        if n <= 600:
            return chunk_by_size(toks, max_tokens=300, overlap=30)
        # if a subsection is still very long, fall back to 300-token windows
        return chunk_by_size(toks, max_tokens=300, overlap=30)

    if n_tokens <= 600:
        return chunk_section(cleaned_text)

    # >600: split by headings, then chunk each subsection
    all_chunks: list[str] = []
    for section in split_by_headings(cleaned_text):
        all_chunks.extend(chunk_section(section))
    return all_chunks


def build_records(raw_text: str) -> list[dict]:
    records: list[dict] = []
    total = 0
    kept_sections = 0

    for sec in iter_sections(raw_text):
        total += 1
        if not is_allowed(sec.path):
            continue

        cleaned = clean_text(sec.text)
        if not cleaned:
            continue

        base_title = extract_title(sec.path, cleaned)
        base_pmid = slugify_path(sec.path)

        chunks = split_into_chunks(cleaned)
        if not chunks:
            continue

        if len(chunks) == 1:
            records.append(
                {
                    "pmid": base_pmid,
                    "title": base_title,
                    "abstract": chunks[0],
                    "source_path": sec.path,
                }
            )
        else:
            for idx, chunk_text in enumerate(chunks, start=1):
                pmid = f"{base_pmid}-{idx}"
                title = f"{base_title} (part {idx})"
                records.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": chunk_text,
                        "source_path": sec.path,
                    }
                )

        kept_sections += 1

    print(f"Total sections: {total}")
    print(f"Kept sections:  {kept_sections}")
    print(f"Total chunks:   {len(records)}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Chunk Orange documentation text into JSONL for BM25/dense indexing."
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to all_documentation.txt",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write JSONL (one JSON object per line).",
    )
    args = ap.parse_args()

    raw_text = args.input.read_text(encoding="utf-8")
    records = build_records(raw_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

