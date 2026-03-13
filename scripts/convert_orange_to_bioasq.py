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
from typing import Any, Dict, List


def convert_mcq_to_bioasq(input_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load MCQ JSON array and convert to BioASQ-style questions dict."""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {input_path}, got {type(data)}")

    stem = input_path.stem
    questions: List[Dict[str, Any]] = []

    for i, item in enumerate(data):
        msgs = item.get("messages") or []
        if len(msgs) < 2:
            raise ValueError(f"Item {i} in {input_path} has fewer than 2 messages")

        user_msg = msgs[1]
        if user_msg.get("role") != "user":
            # Be tolerant but log via stderr
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
                # Fall back to full prompt if we cannot cleanly parse markers.
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

        questions.append(
            {
                "id": qid,
                "body": full_prompt,
                "body_query": question_only,
                # Explicitly mark all Orange QA items as multiple-choice questions.
                "type": "MCQ",
            }
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bioasq_obj = convert_mcq_to_bioasq(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(bioasq_obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

