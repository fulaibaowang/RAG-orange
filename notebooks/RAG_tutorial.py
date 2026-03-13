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

# %% [markdown] id="9cd7c256"
# # RAG Retrieval and Evaluation Tutorial on Orange3 QA & MCQ dataset

# %% [markdown] id="01a32a07"
# ### Preparing the environment

# %% colab={"base_uri": "https://localhost:8080/"} id="22242794" outputId="c79f1cfe-8599-43ca-8e79-68bc232561e8"
# @title 🛠️ Setup: Optional Clone & Install (commented out)
import os
import sys

# Optional: clone original tutorial repo (kept here for reference, commented out)
REPO_URL = "https://github.com/fulaibaowang/RAG-orange.git"
REPO_NAME = "RAG-orange"
branch = "main"

# if os.path.isdir(REPO_NAME):
#     print(f"🔄 Updating {REPO_NAME}...")
#     # !cd {REPO_NAME} && git pull origin {branch}
# else:
#     print(f"📥 Cloning {REPO_NAME}...")
#     # !git clone {REPO_URL}

# if REPO_NAME not in sys.path:
#     sys.path.append(os.path.abspath(REPO_NAME))

# Optional: install dependencies from the cloned repo (commented out)
# print("📦 Installing dependencies from cloned repo...")
# #!pip install -q -r {REPO_NAME}/requirements.txt

print("Using local project files. External clone/install is kept above but commented out.")

# %% id="12a95f4e"
import os
import sys
import json
import textwrap

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('../src')
from evaluation_function import evaluate_model

# %% [markdown] id="19405c6b"
# ## PART 1: Baseline evaluation with Qwen3-0.6B
#
# ### Defining the model config

# %% colab={"base_uri": "https://localhost:8080/"} id="839c457e" outputId="be3afa32-7457-4fa5-fa1d-101e950d3725"
MODEL_CONFIG = {
    "base_model": "Qwen/Qwen3-0.6B",
    "batch_size": 8,
}

print("Model configuration:")
for key, value in MODEL_CONFIG.items():
    print(f"   {key}: {value}")

# %% [markdown] id="e7799f7a"
# ## Test dataset and paths

# %% id="d3c8e619"
if "__file__" in globals():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'train_test_dataset')
TESTDATA_MCQ_FILE = os.path.join(DATA_DIR, 'orange_qa_MCQ_test.jsonl')
TESTDATA_MCQ_CON_FILE = os.path.join(DATA_DIR, 'orange_qa_MCQ-con_test.jsonl')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

with open(TESTDATA_MCQ_FILE, "r") as f:
    test_mcq_dataset = json.load(f)
with open(TESTDATA_MCQ_CON_FILE, "r") as f:
    test_mcq_con_dataset = json.load(f)

print(f"MCQ test set: {len(test_mcq_dataset)} questions")
print(f"MCQ-con test set: {len(test_mcq_con_dataset)} questions")

# %% [markdown] id="19d98174"
# ### Load the model and tokenizer
#
# Let's get familiar with the model and tokenizer. Specifically look at what layers the model has.

# %% id="84f54aca"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['base_model'], padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG['base_model'],
    dtype=torch.float16,
    device_map="auto"
)

# %%
print("Running baseline evaluation on MCQ and MCQ-connection test sets (no RAG context)...")
accuracy_mcq_base, se_mcq_base = evaluate_model(model, tokenizer, test_mcq_dataset, batch_size=MODEL_CONFIG['batch_size'])
accuracy_mcq_con_base, se_mcq_con_base = evaluate_model(model, tokenizer, test_mcq_con_dataset, batch_size=MODEL_CONFIG['batch_size'])

baseline_results = {
    "accuracy_mcq": accuracy_mcq_base,
    "se_mcq": se_mcq_base,
    "accuracy_mcq_con": accuracy_mcq_con_base,
    "se_mcq_con": se_mcq_con_base,
}
print("Baseline Evaluation Results:", baseline_results)

# %% [markdown] id="01ea7d7b"
# This is the baseline results by Qwen/Qwen3-0.6B before fine tuning:
#
#     Evaluation Results: {'accuracy_mcq': 5.0, 'se_mcq': 1.54, 'accuracy_mcq_con': 13.5, 'se_mcq_con': 2.42}
#
# Fine tuning and token injection results (for reference, not re-run here):
#
# - after fine tuning:
#   {'accuracy_mcq': 64.0, 'se_mcq': 3.39, 'accuracy_mcq_con': 13.5, 'se_mcq_con': 2.42}
#   
# - after token injection and fine tuning:
#   {'accuracy_mcq': 60.0, 'se_mcq': 3.46, 'accuracy_mcq_con': 17.0, 'se_mcq_con': 2.66}

# %% [markdown] id="64671df3"
# ## PART 2: RAG
#
# In this section we explore how Retrieval-Augmented Generation (RAG) affects
# the model's ability to answer Orange3 MCQ questions.
#
# 1. We first visually inspect how different retrieval methods (BM25, dense,
#    hybrid, rerank) produce different top results for the same question.
# 2. We then evaluate pre-generated RAG answers (produced by llama3.3).
# 3. Finally, we run Qwen3-0.6B with injected RAG context for a fair comparison.

# %% [markdown]
# ### 2.1 Loading retrieval results and document chunks
#
# The retrieval pipeline (run separately via `run_retrieval_rerank_pipeline.sh`)
# produced run files for each stage: BM25, dense, hybrid (RRF fusion of BM25 +
# dense), rerank (cross-encoder on hybrid candidates), and rerank_hybrid (RRF
# of rerank + hybrid).

# %%
def load_run_tsv(path, columns=None):
    """Load a TREC-style run TSV file into a DataFrame."""
    if columns is None:
        columns = ['qid', 'docno', 'rank', 'score']
    df = pd.read_csv(path, sep='\t', names=columns, header=0)
    df['rank'] = df['rank'].astype(int)
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
    return df

bm25_runs = load_run_tsv(os.path.join(OUTPUT_DIR, 'bm25/runs/BM25__orange_qa_MCQ_test_bioasq__top80.tsv'))
dense_runs = load_run_tsv(os.path.join(OUTPUT_DIR, 'dense/runs/dense_orange_qa_MCQ_test_bioasq.tsv'))
hybrid_runs = load_run_tsv(
    os.path.join(OUTPUT_DIR, 'hybrid/runs/best_rrf_orange_qa_MCQ_test_bioasq_top80.tsv'),
    columns=['qid', 'rank', 'docno', 'score'],
)
rerank_runs = load_run_tsv(os.path.join(OUTPUT_DIR, 'rerank/runs/best_rrf_orange_qa_MCQ_test_bioasq_top80.tsv'))
rerank_hybrid_runs = load_run_tsv(
    os.path.join(OUTPUT_DIR, 'rerank_hybrid/runs/best_rrf_orange_qa_MCQ_test_bioasq_top80_rrf_poolR50_poolH50_k60.tsv'),
    columns=['qid', 'docno', 'rank'],
)

chunks_by_id = {}
with open(os.path.join(PROJECT_ROOT, 'data', 'orange_docs_chunks.jsonl')) as f:
    for line in f:
        doc = json.loads(line)
        chunks_by_id[doc['pmid']] = doc

print(f"Loaded {len(chunks_by_id)} document chunks")
print(f"BM25 runs: {len(bm25_runs)} rows, Dense: {len(dense_runs)}, Hybrid: {len(hybrid_runs)}")

# %%
def get_top_hits(runs_df, qid, n=5):
    """Return top-n hits for a given question ID."""
    q = runs_df[runs_df['qid'] == qid].sort_values('rank').head(n)
    return q[['docno', 'rank'] + (['score'] if 'score' in q.columns and q['score'].notna().any() else [])]

def show_chunk_text(docno, max_chars=1000):
    """Display a chunk's title and text."""
    chunk = chunks_by_id.get(docno)
    if chunk is None:
        print(f"  [{docno}] -- not found in chunks JSONL")
        return
    title = chunk.get('title', '')
    text = chunk.get('abstract', '')[:max_chars]
    print(f"  [{docno}] {title}")
    for line in textwrap.wrap(text, width=90):
        print(f"    {line}")

def show_retrieval_comparison(qid, question_text, runs_dict, top_n=3):
    """Print a side-by-side comparison of retrieval results for one question."""
    print(f"Question: {question_text}")
    print(f"ID: {qid}\n")
    for method_name, runs_df in runs_dict.items():
        hits = get_top_hits(runs_df, qid, n=top_n)
        print(f"--- {method_name} top-{top_n} ---")
        for _, row in hits.iterrows():
            score_str = f"  (score: {row['score']:.4f})" if 'score' in row and pd.notna(row.get('score')) else ""
            print(f"  rank {int(row['rank'])}: {row['docno']}{score_str}")
        print()
    unique_top1 = {}
    for method_name, runs_df in runs_dict.items():
        top1 = runs_df[runs_df['qid'] == qid].sort_values('rank').iloc[0]['docno']
        if top1 not in unique_top1:
            unique_top1[top1] = method_name
    print("--- Top-1 chunk text ---")
    for docno, first_method in unique_top1.items():
        print(f"\n  Top-1 for {first_method}:")
        show_chunk_text(docno)

# Helper to get the full MCQ prompt (question + multiple-choice options)
def get_mcq_prompt_with_choices(qid, dataset):
    """
    Return the full user message (question + options) for a given MCQ id.

    The qid follows the pattern '<stem>_<index>', where <stem> matches the
    TESTDATA_MCQ_FILE stem, and index is the 0-based position in the dataset.
    """
    try:
        stem, idx_str = qid.rsplit("_", 1)
        idx = int(idx_str)
    except Exception:
        raise ValueError(f"Unexpected qid format: {qid}")
    return dataset[idx]["messages"][1]["content"]

# %% [markdown]
# ### 2.2 Example 1: BM25 vs Dense vs Hybrid
#
# Question: *"What is the source of the data in GEO Data Sets?"*
#
# Here BM25 top-1 is `vp-sqltable-3` (about SQL/PostgreSQL -- wrong domain),
# while dense retrieval correctly finds `bio-geo-data-sets-1`.
# Hybrid (RRF fusion) also ranks `bio-geo-data-sets-1` first, showing that the
# dense signal rescues the result despite BM25's keyword mismatch.

# %%
qid_ex1 = "orange_qa_MCQ_test_19"
question_ex1 = get_mcq_prompt_with_choices(qid_ex1, test_mcq_dataset)
show_retrieval_comparison(
    qid_ex1,
    question_ex1,
    {
        "BM25": bm25_runs,
        "Dense": dense_runs,
        "Hybrid (BM25+Dense RRF)": hybrid_runs,
    },
    top_n=5,
)

# %% [markdown]
# ### 2.3 Example 2: Rerank corrects hybrid retrieval
#
# Question: *"Which calibration method uses Isotonic Regression?"*
#
# BM25 top-1 is `vp-calibratedlearner` (the correct doc about calibration
# methods), while dense top-1 is `vp-calibrationplot-2` (the plot widget, not
# the learner). Hybrid follows dense and ranks `vp-calibrationplot-2` first.
#
# The cross-encoder reranker re-scores the hybrid candidates and promotes
# `vp-calibratedlearner` back to rank 1 -- demonstrating how reranking can fix
# errors introduced by the initial retrieval fusion.

# %%
qid_ex2 = "orange_qa_MCQ_test_152"
question_ex2 = get_mcq_prompt_with_choices(qid_ex2, test_mcq_dataset)
show_retrieval_comparison(
    qid_ex2,
    question_ex2,
    {
        "BM25": bm25_runs,
        "Dense": dense_runs,
        "Hybrid (BM25+Dense RRF)": hybrid_runs,
        "Rerank (cross-encoder)": rerank_runs,
        "Rerank+Hybrid RRF": rerank_hybrid_runs,
    },
    top_n=3,
)

# %% [markdown]
# ### 2.4 Evaluate llama3.3 RAG answers
#
# The RAG pipeline generated answers using llama3.3 via Ollama. The results
# were converted into the same chat-message format as our MCQ test set.
# We score them by comparing the assistant's answer letter to the ground truth.
#
# **Note:** This comparison is not apples-to-apples with the Qwen3-0.6B baseline
# since llama3.3 is a much larger model. Section 2.5 addresses this.

# %%
def score_pregenerated_answers(pred_path, gold_dataset):
    """Score pre-generated answers against gold labels, return (accuracy, n_correct, n_total)."""
    with open(pred_path, "r") as f:
        predictions = json.load(f)
    n_correct = 0
    n_total = min(len(predictions), len(gold_dataset))
    for pred, gold in zip(predictions, gold_dataset):
        pred_answer = pred['messages'][2]['content'].strip()
        gold_answer = gold['messages'][2]['content'].strip()
        if pred_answer == gold_answer:
            n_correct += 1
    accuracy = n_correct / n_total * 100 if n_total > 0 else 0.0
    return accuracy, n_correct, n_total

mcq_pred_path = os.path.join(OUTPUT_DIR, 'generation_baseline_converted', 'orange_qa_MCQ_test_bioasq_answers.json')
mcq_con_pred_path = os.path.join(OUTPUT_DIR, 'generation_baseline_converted', 'orange_qa_MCQ-con_test_bioasq_answers.json')

acc_llama_mcq, correct_mcq, total_mcq = score_pregenerated_answers(mcq_pred_path, test_mcq_dataset)
acc_llama_mcq_con, correct_mcq_con, total_mcq_con = score_pregenerated_answers(mcq_con_pred_path, test_mcq_con_dataset)

print(f"llama3.3 + RAG context (MCQ):     {acc_llama_mcq:.1f}% ({correct_mcq}/{total_mcq})")
print(f"llama3.3 + RAG context (MCQ-con): {acc_llama_mcq_con:.1f}% ({correct_mcq_con}/{total_mcq_con})")

# %% [markdown]
# ### 2.5 Evaluate Qwen3-0.6B with RAG context
#
# For a fair comparison we inject the same retrieved contexts into the MCQ test
# questions and run `evaluate_model` with Qwen3-0.6B. This uses the
# `build_rag_eval_dataset` script which prepends a `Context:` block to each
# user message.

# %%
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
from build_rag_eval_dataset import build_rag_eval_dataset

evidence_mcq_path = Path(OUTPUT_DIR) / 'evidence_baseline' / 'orange_qa_MCQ_test_bioasq_contexts.json'
evidence_mcq_con_path = Path(OUTPUT_DIR) / 'evidence_baseline' / 'orange_qa_MCQ-con_test_bioasq_contexts.json'

rag_mcq_dataset = build_rag_eval_dataset(
    contexts_path=evidence_mcq_path,
    original_mcq_path=Path(TESTDATA_MCQ_FILE),
    top_k=3,
    restrict_to_contexts=False,
)
rag_mcq_con_dataset = build_rag_eval_dataset(
    contexts_path=evidence_mcq_con_path,
    original_mcq_path=Path(TESTDATA_MCQ_CON_FILE),
    top_k=3,
    restrict_to_contexts=False,
)

print(f"RAG-augmented MCQ dataset: {len(rag_mcq_dataset)} questions")
print(f"RAG-augmented MCQ-con dataset: {len(rag_mcq_con_dataset)} questions")

# %% [markdown]
# The original MCQ prompts say *"based on your knowledge"* and the system
# message has no instruction to use the provided context. The RAG pipeline
# (llama3.3) uses `scripts/prompts/system.txt` which explicitly says
# *"You MUST answer using ONLY the provided Evidence Contexts."*
#
# We patch the messages so Qwen3-0.6B gets the same kind of instruction.

# %%
RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that can answer questions about the Orange Data Mining software.\n"
    "You MUST answer using ONLY the provided Context.\n"
    "Answer with a single letter (A, B, C, or D) corresponding to the correct answer."
)

OLD_USER_INSTRUCTION = (
    "Answer the following question based on your knowledge of the Orange Data Mining software.\n"
    "Make sure you answer the question with a single letter corresponding to the correct answer."
)
NEW_USER_INSTRUCTION = (
    "Answer the following question using ONLY the provided context about the Orange Data Mining software.\n"
    "Answer with a single letter (A, B, C, or D)."
)

def patch_rag_prompts(dataset):
    """Replace system and user instructions to align with the RAG pipeline prompts."""
    for item in dataset:
        msgs = item['messages']
        msgs[0]['content'] = RAG_SYSTEM_PROMPT
        msgs[1]['content'] = msgs[1]['content'].replace(OLD_USER_INSTRUCTION, NEW_USER_INSTRUCTION)
    return dataset

rag_mcq_dataset = patch_rag_prompts(rag_mcq_dataset)
rag_mcq_con_dataset = patch_rag_prompts(rag_mcq_con_dataset)

# %%
print("Patched system message:")
print(rag_mcq_dataset[0]['messages'][0]['content'])
print("\nSample: RAG-augmented user message (first 5000 chars):")
print(rag_mcq_dataset[0]['messages'][1]['content'][:5000])

# %% [markdown]
# #### 2.5.1 Inspect raw Qwen3-0.6B generations (debug)
#
# Before trusting the accuracy numbers, it is useful to look at a few raw
# generations from Qwen3-0.6B to see:
# - whether the model outputs just a letter (A/B/C/...), or a longer sentence,
# - whether it follows the `/no_think` instruction and whether any `<think>` tags
#   appear in the output,
# - how often the extracted answer (after stripping any thinking) matches the
#   gold letter.
#
# The helper below prints a small sample of MCQ items together with:
# - the (truncated) user prompt,
# - the gold answer letter,
# - the raw decoded model output,
# - the extracted prediction that `evaluate_model` compares against the gold.

# %%
def debug_qwen_outputs(sample_dataset, num_examples=5):
    model.eval()
    with torch.no_grad():
        batch = sample_dataset[:num_examples]
        batch_texts = []
        gold_answers = []
        user_messages = []

        for item in batch:
            system_user_messages = [
                {"role": item["messages"][0]["role"], "content": item["messages"][0]["content"]},
                {"role": item["messages"][1]["role"], "content": item["messages"][1]["content"] + "\n /no_think"},
            ]
            gold_answers.append(item["messages"][2]["content"])
            user_messages.append(item["messages"][1]["content"])

            text = tokenizer.apply_chat_template(
                system_user_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_tracking=False,
            )
            batch_texts.append(text)

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        for i, (output, gold, user_msg) in enumerate(zip(outputs, gold_answers, user_messages)):
            input_length = inputs.input_ids[i].shape[0]
            generated_token_ids = output[input_length:]
            decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

            if "</think>" in decoded_text:
                prediction = decoded_text.split("</think>")[1].strip()
            else:
                prediction = decoded_text.strip()

            print("=" * 80)
            print(f"Example {i}")
            print("- User message (truncated):")
            print(user_msg[:400] + ("..." if len(user_msg) > 400 else ""))
            print("\n- Gold answer:", repr(gold))
            print("- Raw decoded output:")
            print(decoded_text)
            print("- Extracted prediction (compared to gold):", repr(prediction))
            print()

# %%
print("Evaluating Qwen3-0.6B with RAG context on MCQ...")
accuracy_mcq_rag, se_mcq_rag = evaluate_model(model, tokenizer, rag_mcq_dataset, batch_size=MODEL_CONFIG['batch_size'])
print(f"MCQ with RAG: accuracy={accuracy_mcq_rag:.1f}%, SE={se_mcq_rag:.2f}")

print("\nEvaluating Qwen3-0.6B with RAG context on MCQ-con...")
accuracy_mcq_con_rag, se_mcq_con_rag = evaluate_model(model, tokenizer, rag_mcq_con_dataset, batch_size=MODEL_CONFIG['batch_size'])
print(f"MCQ-con with RAG: accuracy={accuracy_mcq_con_rag:.1f}%, SE={se_mcq_con_rag:.2f}")

# %%
# Uncomment to inspect a few raw generations from Qwen3-0.6B.
# Start with the RAG-augmented MCQ dataset; you can also try the original
# (non-RAG) dataset by passing `test_mcq_dataset` instead.
#
# debug_qwen_outputs(rag_mcq_dataset, num_examples=5)

# %% [markdown]
# ### 2.6 Summary
#
# | Setting | MCQ Accuracy | MCQ-con Accuracy |
# |---|---|---|
# | Qwen3-0.6B (no context) | 5.0% | 13.5% |
# | Qwen3-0.6B + RAG context | *run above* | *run above* |
# | llama3.3 + RAG context | *run above* | *run above* |
# | Qwen3-0.6B + LoRA fine-tuning (for reference) | 64.0% | 13.5% |

# %%
print("=" * 65)
print(f"{'Setting':<35} {'MCQ':>10} {'MCQ-con':>10}")
print("-" * 65)
print(f"{'Qwen3-0.6B (no context)':<35} {'5.0%':>10} {'13.5%':>10}")
print(f"{'Qwen3-0.6B + RAG context':<35} {accuracy_mcq_rag:>9.1f}% {accuracy_mcq_con_rag:>9.1f}%")
print(f"{'llama3.3 + RAG context':<35} {acc_llama_mcq:>9.1f}% {acc_llama_mcq_con:>9.1f}%")
print(f"{'Qwen3-0.6B + LoRA (reference)':<35} {'64.0%':>10} {'13.5%':>10}")
print("=" * 65)
