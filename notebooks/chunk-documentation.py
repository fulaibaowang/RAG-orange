# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: orange RAG (Python 3.14 venv)
#     language: python
#     name: ornagerag-py314
# ---

# %% [markdown]
# # Orange Documentation Chunking Exploration
#
# This notebook explores `data/all_documentation.txt`, inspects per-file sections, and prototypes cleaning and chunking rules before we lock them into a script.

# %%
from pathlib import Path
import re
from dataclasses import dataclass
from typing import List

import pandas as pd

BASE_DIR = Path("..").resolve()
DOC_PATH = BASE_DIR / "data" / "all_documentation.txt"
DOC_PATH


# %%
@dataclass
class Section:
    path: str
    text: str


MARKER_RE = re.compile(r"^=== (.+) ===$")


def parse_sections(raw_text: str) -> List[Section]:
    sections: List[Section] = []
    current_path: str | None = None
    current_lines: List[str] = []

    for line in raw_text.splitlines():
        m = MARKER_RE.match(line)
        if m:
            # flush previous
            if current_path is not None:
                sections.append(Section(path=current_path, text="\n".join(current_lines).rstrip()))
            current_path = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_path is not None:
        sections.append(Section(path=current_path, text="\n".join(current_lines).rstrip()))

    return sections


raw = DOC_PATH.read_text(encoding="utf-8")
sections = parse_sections(raw)
len(sections)

# %%
df = pd.DataFrame({
    "path": [s.path for s in sections],
    "text": [s.text for s in sections],
    "n_chars": [len(s.text) for s in sections],
})
df

# %%
df["top_dir"] = df["path"].str.split("/", n=1).str[0]
df.groupby("top_dir").size().sort_values(ascending=False)

# %%
IMAGE_LINE_RE = re.compile(r"^\s*!\[.*\]\(.*\)\s*(\{.*\})?\s*$")
WIDTH_ATTR_RE = re.compile(r"\{[^}]*width=[^}]*\}")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
RST_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+(toctree|automodule|autoclass|autoattribute)\b")
RST_OPTION_RE = re.compile(r"^\s*:[a-zA-Z0-9_-]+:\s*")
EMPH_RE = re.compile(r"(\*{1,2})([^*]+)\1")   # *text* or **text**
UNDERLINE_RE = re.compile(r"^[-=]{3,}\s*$")   # ==== or ---- lines
NUMBERED_LIST_RE = re.compile(r"^\d+[\.\)]\s")  # 1. ..., 2) ...


def clean_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if IMAGE_LINE_RE.match(line):
            continue
        stripped = line.strip()
        if UNDERLINE_RE.match(stripped):
            # drop pure underline lines (e.g. ====, ----)
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


sample_paths = [
    "orange3-doc-visual-programming/source/widgets/unsupervised/PCA.md",
    "orange3-doc-visual-programming/source/widgets/unsupervised/louvainclustering.md",
    "orange3-single-cell/doc/widgets/filter.md",
]

df_samples = df[df["path"].isin(sample_paths)].copy()
df_samples["cleaned"] = df_samples["text"].map(clean_text)
df_samples[["path", "n_chars"]]

# %%
for _, row in df_samples.iterrows():
    print("== PATH ==", row["path"])
    print("\n--- RAW (first 40 lines) ---")
    print("\n".join(row["text"].splitlines()[:40]))
    print("\n--- CLEANED (first 40 lines) ---")
    print("\n".join(row["cleaned"].splitlines()[:40]))
    print("\n" + "=" * 80 + "\n")

# %%
import fnmatch

ALLOWLIST_PATTERNS = [
    # Core widget docs (visual programming)
    "orange3-doc-visual-programming/source/widgets/**/*.md",
    # User guides
    "orange3-doc-visual-programming/source/exporting-*/index.md",
    "orange3-doc-visual-programming/source/report/index.md",
    "orange3-doc-visual-programming/source/learners-as-scorers/index.md",
    "orange3-doc-visual-programming/source/building-workflows/index.md",
    "orange3-doc-visual-programming/source/loading-your-data/index.md",
    # Add-on widget docs
    "orange3-text/doc/widgets/*.md",
    "orange3-bioinformatics/doc/widgets/*.md",
    "orange3-survival-analysis/doc/widgets/*.md",
    "orange3-single-cell/doc/widgets/*.md",
    "orange3-timeseries/doc/widgets/*.md",
    # Core reference & tutorials
    "orange3/Orange/distance/distances.md",
    "orange3/doc/data-mining-library/source/tutorial/*.rst",
]


def is_allowed(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in ALLOWLIST_PATTERNS)


df["allowed"] = df["path"].map(is_allowed)
df["allowed"].value_counts()

# %%
# Detailed inspection of allowed vs dropped sections

allowed_df = df[df["allowed"]].copy()
dropped_df = df[~df["allowed"]].copy()

print("Allowed by top_dir (count):")
display(allowed_df.groupby("top_dir").size().sort_values(ascending=False))

print("\nDropped by top_dir (count):")
display(dropped_df.groupby("top_dir").size().sort_values(ascending=False))

print("\nSample KEPT paths per top_dir (first 5 each):")
for top in allowed_df["top_dir"].unique():
    subset = allowed_df[allowed_df["top_dir"] == top]
    print(f"\n[KEPT] top_dir={top}")
    print(subset["path"].head(5).to_string(index=False))

print("\nSample DROPPED paths per top_dir (first 5 each):")
for top in dropped_df["top_dir"].unique():
    subset = dropped_df[dropped_df["top_dir"] == top]
    print(f"\n[DROPPED] top_dir={top}")
    print(subset["path"].head(5).to_string(index=False))

# Borderline: anything under widgets/ that is currently dropped
borderline = df[(~df["allowed"]) & df["path"].str.contains("widgets/")].copy()
if not borderline.empty:
    print("\n\nBorderline cases: paths containing 'widgets/' but currently DROPPED:")
    print(borderline[["top_dir", "path"]].sort_values(["top_dir", "path"]).to_string(index=False))
else:
    print("\n\nNo borderline 'widgets/' paths are being dropped.")


# %%
def slugify_path(path: str) -> str:
    # Very similar to what the script will do: prefix by area + filename
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


preview = []
for _, row in df[df["allowed"]].head(10).iterrows():
    cleaned = clean_text(row["text"])
    title = cleaned.splitlines()[0] if cleaned else Path(row["path"]).stem
    preview.append({
        "pmid": slugify_path(row["path"]),
        "title": title.strip("# "),
        "abstract": cleaned,
        "source_path": row["path"],
    })

preview

# %%
# Prototype chunking logic in the notebook

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

      - always split by headings first (if any headings are found)
      - for each heading section:
          * if < 350 tokens: keep as-is (no overlap)
          * if ≥ 350 tokens: use 350-token windows with 30-token overlap
      - after all sections are chunked, merge any final chunks smaller
        than SMALL_CHUNK_TOKENS into their neighbors.
    """
    top_tokens = tokenize(cleaned_text)
    if not top_tokens:
        return []

    # If the whole section is short, keep it as a single chunk
    if len(top_tokens) < 350:
        return [cleaned_text]

    # For longer sections, use headings + size rules
    sections = split_by_headings(cleaned_text)

    per_section_chunks: list[str] = []

    for section in sections:
        toks = tokenize(section)
        n = len(toks)
        if n == 0:
            continue
        if n < 350:
            per_section_chunks.append(section)
        else:
            per_section_chunks.extend(chunk_by_size(toks, max_tokens=350, overlap=30))

    if not per_section_chunks:
        return []

    # Global pass: merge small chunks into their predecessors
    return merge_small_chunks(per_section_chunks, min_tokens=SMALL_CHUNK_TOKENS)



# %%
# Build chunk-level DataFrame directly in the notebook

allowed_df = df[df["allowed"]].copy()
allowed_df["cleaned"] = allowed_df["text"].map(clean_text)

chunk_rows = []
for _, row in allowed_df.iterrows():
    path = row["path"]
    cleaned = row["cleaned"]
    base_title = cleaned.splitlines()[0] if cleaned else Path(path).stem
    base_title = base_title.strip("# ").strip()
    base_pmid = slugify_path(path)

    chunks = split_into_chunks(cleaned)
    if not chunks:
        continue

    if len(chunks) == 1:
            chunk_rows.append(
                {
                    "pmid": base_pmid,
                    "title": base_title,
                    "abstract": f"{base_title}\n\n{chunks[0]}",  # prepend title
                    "source_path": path,
                }
            )
    else:
        for idx, chunk_text in enumerate(chunks, start=1):
            pmid = f"{base_pmid}-{idx}"
            title = f"{base_title} (part {idx})"
            chunk_rows.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": f"{base_title}\n\n{chunk_text}",  # prepend title to every chunk
                    "source_path": path,
                }
            )

chunks_df = pd.DataFrame(chunk_rows)
chunks_df["word_count"] = chunks_df["abstract"].str.split().str.len()

print("Total allowed sections:", len(allowed_df))
print("Total chunks (notebook prototype):", len(chunks_df))
print("\nWord count summary for notebook chunks:")
print(chunks_df["word_count"].describe())

chunks_df[["pmid", "title", "word_count"]]

# %%
# Inspect chunks with fewer than 50 tokens

small_chunks = chunks_df[chunks_df["word_count"] < 50].copy()
print("Number of chunks with < 50 tokens:", len(small_chunks))
if not small_chunks.empty:
    print("\nSample of small chunks (<50 tokens):")
    for _, row in small_chunks.head(20).iterrows():
        text = row["abstract"]
        toks = tokenize(text)
        preview = " ".join(toks[:])
        print("\n---")
        print(f"pmid: {row['pmid']}")
        print(f"source_path: {row['source_path']}")
        print(f"word_count: {row['word_count']}")
        print(f"preview: {preview}...")

# %%
chunks_df = chunks_df[chunks_df["word_count"] >= 30].reset_index(drop=True)

# %%
# Histogram of chunk word counts

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
chunks_df["word_count"].hist(bins=30)
plt.xlabel("Words per chunk")
plt.ylabel("Frequency")
plt.title("Chunk word-count distribution (notebook chunks, >=30 words)")
plt.tight_layout()
plt.show()

# %%
# Histogram of section word counts BEFORE chunking (allowed sections)

allowed_df["section_word_count"] = allowed_df["cleaned"].str.split().str.len()

plt.figure(figsize=(8, 4))
allowed_df["section_word_count"].hist(bins=100)
plt.xlabel("Words per section (before chunking)")
plt.ylabel("Frequency")
plt.title("Section word-count distribution (allowed sections, before chunking)")
plt.tight_layout()
plt.show()

# %%
# Print detailed chunking examples for a few docs

example_paths = [
    "orange3-doc-visual-programming/source/widgets/unsupervised/PCA.md",
    "orange3-doc-visual-programming/source/widgets/unsupervised/tsne.md",
    "orange3-timeseries/doc/widgets/line_chart.md",
]

for path in example_paths:
    sect = allowed_df[allowed_df["path"] == path]
    if sect.empty:
        continue
    cleaned = sect.iloc[0]["cleaned"]
    n_tokens = len(tokenize(cleaned))
    print("\n" + "=" * 80)
    print(f"Source path: {path}")
    print(f"Section tokens (before chunking): {n_tokens}")

    ex_chunks = chunks_df[chunks_df["source_path"] == path].copy()
    if ex_chunks.empty:
        print("No chunks found for this path (likely filtered out).")
        continue

    print(f"Number of chunks: {len(ex_chunks)}")
    for i, row in ex_chunks.sort_values("pmid").iterrows():
        text = row["abstract"]
        tokens = tokenize(text)
        preview = " ".join(tokens[:40])
        print(f"\n  Chunk pmid: {row['pmid']}")
        print(f"  Title: {row['title']}")
        print(f"  Tokens in chunk: {len(tokens)}")
        print(f"  Preview: {preview}...")


# %%
# Broader inspection: show chunking for many multi-chunk docs

print("\n" + "#" * 80)
print("Broader chunking overview for multi-chunk docs")

grouped = chunks_df.groupby("source_path")
multi_paths = [p for p, g in grouped if len(g) > 1]

for path in multi_paths[:20]:
    sect = allowed_df[allowed_df["path"] == path]
    if sect.empty:
        continue
    cleaned = sect.iloc[0]["cleaned"]
    n_tokens = len(tokenize(cleaned))
    doc_chunks = grouped.get_group(path).sort_values("pmid")

    print("\n" + "=" * 80)
    print(f"Source path: {path}")
    print(f"Section tokens (before chunking): {n_tokens}")
    print(f"Number of chunks: {len(doc_chunks)}")

    for _, row in doc_chunks.iterrows():
        text = row["abstract"]
        tokens = tokenize(text)
        preview = " ".join(tokens[:])
        print(f"\n  Chunk pmid: {row['pmid']}")
        print(f"  Tokens in chunk: {len(tokens)}")
        print(f"  Preview: {preview}...")


# %%

# %%
