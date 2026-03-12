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
NUMBERED_LIST_RE = re.compile(r"^\d+[\.\)]\s")  # 1. ..., 2) ...

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
    "orange3-timeseries/doc/widgets/*.md",
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


MIN_SECTION_TOKENS = 50      # minimum tokens when merging heading sections
SMALL_CHUNK_TOKENS = 50      # minimum tokens for final chunks (merge smaller)


def split_by_headings(cleaned_text: str) -> list[str]:
    """
    Split a long document into sections by approximate headings,
    then merge any tiny sections (< MIN_SECTION_TOKENS) into
    the following section so we don't get degenerate 1-token chunks.
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
            # skip obvious list items / numbered lines
            if stripped.startswith(("-", "*", "+")):
                continue
            if NUMBERED_LIST_RE.match(stripped):
                continue
            if len(stripped) <= 80:
                heading_idxs.append(i)

    if not heading_idxs:
        return [cleaned_text]

    if heading_idxs[0] != 0:
        heading_idxs.insert(0, 0)

    raw_sections: list[str] = []
    for start, end in zip(heading_idxs, heading_idxs[1:] + [len(lines)]):
        chunk_lines = lines[start:end]
        section = "\n".join(chunk_lines).strip()
        if section:
            raw_sections.append(section)

    if not raw_sections:
        return [cleaned_text]

    # merge small sections into the next one
    merged: list[str] = []
    buf = ""
    for sec in raw_sections:
        if buf:
            buf = buf + "\n\n" + sec
        else:
            buf = sec
        if len(tokenize(buf)) >= MIN_SECTION_TOKENS:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buf
        else:
            merged.append(buf)

    return merged or [cleaned_text]


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


def merge_small_chunks(chunks: list[str], min_tokens: int = SMALL_CHUNK_TOKENS) -> list[str]:
    """
    Merge chunks that are shorter than min_tokens into their preceding chunk.
    """
    merged: list[str] = []
    for chunk in chunks:
        toks = tokenize(chunk)
        if not merged:
            merged.append(chunk)
            continue
        if len(toks) < min_tokens:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)
    return merged


def split_into_chunks(cleaned_text: str) -> list[str]:
    """
    Apply the chunking rules:

      - if whole section < 300 tokens: keep as a single chunk
      - otherwise:
          * split by headings
          * within each heading section:
              - if < 300 tokens: keep as-is
              - if ≥ 300 tokens: 300-token windows with 30-token overlap
          * finally, merge any chunks shorter than SMALL_CHUNK_TOKENS
            into their predecessors.
    """
    top_tokens = tokenize(cleaned_text)
    if not top_tokens:
        return []

    # If the whole section is short, keep it as a single chunk
    if len(top_tokens) < 300:
        return [cleaned_text]

    # For longer sections, use headings + size rules
    sections = split_by_headings(cleaned_text)

    per_section_chunks: list[str] = []

    for section in sections:
        toks = tokenize(section)
        n = len(toks)
        if n == 0:
            continue
        if n < 300:
            per_section_chunks.append(section)
        else:
            per_section_chunks.extend(chunk_by_size(toks, max_tokens=300, overlap=30))

    if not per_section_chunks:
        return []

    # Global pass: merge small chunks into their predecessors
    return merge_small_chunks(per_section_chunks, min_tokens=SMALL_CHUNK_TOKENS)


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
                    "abstract": f"{base_title}\n\n{chunks[0]}",
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
                        "abstract": f"{base_title}\n\n{chunk_text}",
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

