#!/usr/bin/env python3
"""
Convert Orange QA MCQ-style JSONL (actually a JSON array) into BioASQ-style JSON.

Input format (per element of the top-level list):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": ".... Question: ... Answers: ..."},
        {"role": "assistant", "content": "D"}
    ]
}

Output format:
{
  "questions": [
    {
      "id": "<stem>_0",
      "body": "<question text>",
      "type": "factoid"
    },
    ...
  ]
}

The question ID scheme is:
    qid = f"{Path(input_path).stem}_{index}"

This same scheme is assumed by `build_rag_eval_dataset.py` when mapping
pipeline outputs back onto the original MCQ dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _build_ground_truth_maps(
    qa_full_path: Path,
    chunks_path: Path,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Return (question_to_file, sourcepath_to_pmids) for ground-truth lookup."""
    with qa_full_path.open("r", encoding="utf-8") as f:
        qa_full = json.load(f)
    question_to_file: Dict[str, str] = {}
    for entry in qa_full:
        question_to_file[entry["question"].strip()] = entry["file"]

    sourcepath_to_pmids: Dict[str, List[str]] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            sourcepath_to_pmids.setdefault(chunk["source_path"], []).append(chunk["pmid"])

    return question_to_file, sourcepath_to_pmids


def convert_mcq_to_bioasq(
    input_path: Path,
    qa_full_path: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load MCQ JSON array and convert to BioASQ-style questions dict.

    When *qa_full_path* and *chunks_path* are provided, each output question
    gets a ``documents`` list of chunk pmids derived from the source file.
    """
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {input_path}, got {type(data)}")

    question_to_file: Dict[str, str] = {}
    sourcepath_to_pmids: Dict[str, List[str]] = {}
    if qa_full_path and chunks_path:
        question_to_file, sourcepath_to_pmids = _build_ground_truth_maps(
            qa_full_path, chunks_path,
        )

    stem = input_path.stem
    questions: List[Dict[str, Any]] = []
    docs_matched = 0

    for i, item in enumerate(data):
        msgs = item.get("messages") or []
        if len(msgs) < 2:
            raise ValueError(f"Item {i} in {input_path} has fewer than 2 messages")

        user_msg = msgs[1]
        if user_msg.get("role") != "user":
            print(
                f"Warning: item {i} user message has role={user_msg.get('role')!r}, "
                f"expected 'user'",
                file=sys.stderr,
            )
        # For MCQ, we keep the full user prompt (instructions + question + options)
        # in `body` so downstream components (e.g., generation) see the exact same text,
        # and also expose a question-only field `body_query` (used as retrieval query).
        raw_content = str(user_msg.get("content", ""))
        full_prompt = raw_content.strip()

        question_only = full_prompt
        q_marker = "Question:"
        a_marker = "Answers:"
        try:
            q_idx = raw_content.find(q_marker)
            a_idx = raw_content.find(a_marker, q_idx + len(q_marker)) if q_idx != -1 else -1
            if q_idx != -1 and a_idx != -1:
                question_only = raw_content[q_idx + len(q_marker) : a_idx].strip()
            else:
                question_only = full_prompt
                print(
                    f"Warning: item {i} in {input_path} is missing expected 'Question:'/'Answers:' markers; "
                    "using full prompt as body_query.",
                    file=sys.stderr,
                )
        except Exception as exc:  # pragma: no cover - defensive
            question_only = full_prompt
            print(
                f"Warning: failed to parse question-only text for item {i} in {input_path}: {exc!r}; "
                "using full prompt as body_query.",
                file=sys.stderr,
            )

        qid = f"{stem}_{i}"

        q_obj: Dict[str, Any] = {
            "id": qid,
            "body": full_prompt,
            "body_query": question_only,
            "type": "MCQ",
        }

        if question_to_file:
            src_file = question_to_file.get(question_only)
            if src_file:
                pmids = sourcepath_to_pmids.get(src_file, [])
                q_obj["documents"] = pmids
                if pmids:
                    docs_matched += 1
                else:
                    print(
                        f"Warning: item {i} (qid={qid}) source file {src_file!r} "
                        "has no chunks in the index.",
                        file=sys.stderr,
                    )
            else:
                q_obj["documents"] = []
                print(
                    f"Warning: item {i} (qid={qid}) body_query not found in --qa-full; "
                    "documents will be empty.",
                    file=sys.stderr,
                )

        questions.append(q_obj)

    if question_to_file:
        print(
            f"Ground-truth documents: {docs_matched}/{len(questions)} questions matched.",
            file=sys.stderr,
        )

    return {"questions": questions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Orange QA MCQ JSON array to BioASQ-style JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to MCQ JSONL file (actually a JSON array).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write BioASQ-style JSON.",
    )
    parser.add_argument(
        "--qa-full",
        type=Path,
        default=None,
        help="Path to orange_qa_full.json for ground-truth document mapping.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Path to orange_docs_chunks.jsonl for source_path -> pmid mapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.qa_full) != bool(args.chunks):
        print(
            "Error: --qa-full and --chunks must be provided together.",
            file=sys.stderr,
        )
        sys.exit(1)

    bioasq_obj = convert_mcq_to_bioasq(args.input, args.qa_full, args.chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(bioasq_obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

