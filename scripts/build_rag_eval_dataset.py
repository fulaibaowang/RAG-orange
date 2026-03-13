#!/usr/bin/env python3
"""
Build an evaluation dataset with retrieved contexts injected into Orange MCQ questions.

Inputs:
  1) Contexts JSON produced by build_contexts_from_documents.py, e.g.
     evidence_baseline/<split>_contexts.json
     {
       "questions": [
         {
           "id": "<stem>_0",
           "body": "...",
           "contexts": [
             {"id": "vp-index-1", "doc": "http://www.ncbi.nlm.nih.gov/pubmed/vp-index", "text": "..."},
             ...
           ]
         },
         ...
       ]
     }

  2) Original MCQ JSON array, e.g. data/train_test_dataset/orange_qa_MCQ_test.jsonl
     [
       {
         "messages": [
           {"role": "system", "content": "..."},
           {"role": "user", "content": "Answer the following question... Question: ... Answers: ..."},
           {"role": "assistant", "content": "D"}
         ]
       },
       ...
     ]

We assume the same ID scheme as convert_orange_to_bioasq.py:
    qid = f"{Path(original_mcq_path).stem}_{index}"

For each MCQ item, we:
  - Look up contexts for its qid.
  - If any exist, prepend a "Context:" block ahead of the "Question:" section
    in the user message content.

Output:
  - JSON array in the same messages format, suitable for evaluate_model().
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_contexts(path: Path, top_k: int) -> Dict[str, List[str]]:
    """
    Return mapping qid -> list of context texts (up to top_k each).

    The input can be either:
      - a *_contexts.json from build_contexts_from_documents.py, or
      - a *_answers.json from generate_answers.py, which has the same
        question/contexts structure plus answer fields.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions") or []
    contexts_by_qid: Dict[str, List[str]] = {}

    for q in questions:
        qid = str(q.get("id"))
        ctxs = q.get("contexts") or []
        texts: List[str] = []
        for ctx in ctxs:
            text = ctx.get("text")
            if not text:
                continue
            texts.append(str(text))
            if len(texts) >= top_k:
                break
        if texts:
            contexts_by_qid[qid] = texts

    return contexts_by_qid


def inject_context_into_user_message(content: str, contexts: List[str]) -> str:
    """Inject a 'Context:' block before the 'Question:' section in the user content."""
    if not contexts:
        return content

    context_lines = ["Context:"]
    for i, txt in enumerate(contexts, start=1):
        context_lines.append(f"[{i}] {txt}")
    context_block = "\n".join(context_lines) + "\n\n"

    marker = "Question:"
    idx = content.find(marker)
    if idx == -1:
        # Fallback: prepend context at the beginning
        return context_block + content

    prefix = content[:idx].rstrip() + "\n\n"
    suffix = content[idx:]
    return prefix + context_block + suffix


def build_rag_eval_dataset(
    contexts_path: Path,
    original_mcq_path: Path,
    top_k: int,
    restrict_to_contexts: bool,
) -> List[Dict[str, Any]]:
    """Combine contexts with the original MCQ dataset."""
    contexts_by_qid = load_contexts(contexts_path, top_k=top_k)

    with original_mcq_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {original_mcq_path}, got {type(data)}")

    stem = original_mcq_path.stem
    augmented: List[Dict[str, Any]] = []

    for i, item in enumerate(data):
        qid = f"{stem}_{i}"
        contexts = contexts_by_qid.get(qid)
        if not contexts:
            if restrict_to_contexts:
                # Skip questions that were not part of the pipeline subset.
                continue
            # No contexts for this question; keep it unchanged.
            augmented.append(item)
            continue

        msgs = item.get("messages") or []
        if len(msgs) < 2:
            augmented.append(item)
            continue

        user_msg = msgs[1]
        if user_msg.get("role") != "user":
            # Be tolerant, but do not change content if structure is unexpected.
            augmented.append(item)
            continue

        original_content = str(user_msg.get("content", ""))
        new_content = inject_context_into_user_message(original_content, contexts)

        # Construct a shallow copy so we do not mutate the original list in-place.
        new_item = dict(item)
        new_messages = list(msgs)
        new_user_msg = dict(user_msg)
        new_user_msg["content"] = new_content
        new_messages[1] = new_user_msg
        new_item["messages"] = new_messages
        augmented.append(new_item)

    return augmented


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build RAG evaluation dataset by injecting retrieved contexts into "
            "Orange MCQ user messages."
        )
    )
    parser.add_argument(
        "--contexts-json",
        type=Path,
        required=True,
        help=(
            "Path to questions JSON with contexts: either *_contexts.json from "
            "build_contexts_from_documents.py or *_answers.json from generate_answers.py."
        ),
    )
    parser.add_argument(
        "--original-mcq",
        type=Path,
        required=True,
        help="Path to original MCQ JSON array (e.g. orange_qa_MCQ_test.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write augmented MCQ JSON array.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of contexts to inject per question.",
    )
    parser.add_argument(
        "--restrict-to-contexts",
        action="store_true",
        help=(
            "If set, only include questions that have contexts in --contexts-json "
            "(useful when the pipeline was run on a subset)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    augmented = build_rag_eval_dataset(
        contexts_path=args.contexts_json,
        original_mcq_path=args.original_mcq,
        top_k=args.top_k,
        restrict_to_contexts=args.restrict_to_contexts,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

