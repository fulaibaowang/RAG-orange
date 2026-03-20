# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RAG Retrieval and Evaluation Tutorial on the Orange3 QA & MCQ Dataset
#
# This tutorial walks through two main topics:
#
# 1. **Baseline evaluation** -- measure how Qwen3-0.6B performs on Orange3
#    multiple-choice questions *without* any retrieval or fine-tuning.
# 2. **RAG evaluation** -- explore retrieval results produced by the
#    `run_retrieval_rerank_pipeline.sh` pipeline, then evaluate answer quality
#    with and without fine-tuning.

# %% [markdown]
# ### Preparing the environment
#
# This notebook works both **locally** (inside the cloned repo) and on
# **Google Colab**. The cell below detects the runtime, sets up paths, and
# -- on Colab -- clones the repo and installs dependencies automatically.

# %%
import os
import sys

IN_COLAB = False
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except ImportError:
    pass

REPO_URL = "https://github.com/fulaibaowang/RAG-orange.git"
REPO_NAME = "RAG-orange"
BRANCH = "main"

if IN_COLAB:
    if not os.path.isdir(REPO_NAME):
        print(f"Cloning {REPO_NAME} from GitHub...")
        get_ipython().system(f"git clone -b {BRANCH} {REPO_URL}")
    os.chdir(REPO_NAME)

    req_path = os.path.join(os.getcwd(), "requirements.txt")
    if os.path.isfile(req_path):
        print("Installing dependencies from requirements.txt...")
        get_ipython().system("pip install -q -r requirements.txt")

if "__file__" in globals():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    _cwd = os.path.abspath(os.getcwd())
    # Walk up from cwd to find the repo root (contains .git)
    _candidate = _cwd
    while _candidate != os.path.dirname(_candidate):
        if os.path.isdir(os.path.join(_candidate, ".git")):
            break
        _candidate = os.path.dirname(_candidate)
    PROJECT_ROOT = _candidate

for _subdir in ("src", "scripts"):
    _p = os.path.join(PROJECT_ROOT, _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"Environment: {'Google Colab' if IN_COLAB else 'local'}")
print(f"Project root: {PROJECT_ROOT}")

# %%
import json
import textwrap

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluation_function import evaluate_model

# %% [markdown]
# ## PART 1: Baseline Evaluation with Qwen3-0.6B
#
# Before adding any retrieval context, we establish a baseline by running
# Qwen3-0.6B on the Orange3 MCQ test sets without any external knowledge.
#
# ### 1.1 Model configuration

# %%
MODEL_CONFIG = {
    "base_model": "Qwen/Qwen3-0.6B",
    "batch_size": 8,
}

print("Model configuration:")
for key, value in MODEL_CONFIG.items():
    print(f"   {key}: {value}")

# %% [markdown]
# ### 1.2 Test datasets and paths

# %%
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

# %% [markdown]
# ### 1.3 Load the model and tokenizer

# %%
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

# %% [markdown]
# ### 1.4 Baseline results
#
# Qwen3-0.6B baseline **before** RAG or fine-tuning:
#
#     {'accuracy_mcq': 5.0, 'se_mcq': 1.54, 'accuracy_mcq_con': 13.5, 'se_mcq_con': 2.42}
#
# For reference, the
# [LoRA fine-tuning tutorial](https://github.com/MartinSpendl/LoRA-finetuning-tutorial/blob/main/notebooks/0.0-lora-tutorial.ipynb)
# reports the following results (not re-run here):
#
# | Setting | MCQ | MCQ-con |
# |---|---|---|
# | After LoRA fine-tuning | 64.0 % | 13.5 % |
# | After token injection + fine-tuning | 60.0 % | 17.0 % |

# %% [markdown]
# ## PART 2: Retrieval-Augmented Generation (RAG)
#
# In this part we examine how providing retrieved documentation context affects
# the model's ability to answer Orange3 MCQ questions. **We do not run the
# retrieval pipeline itself in this notebook.** The full retrieval, reranking,
# evidence extraction, and answer generation pipeline was executed beforehand
# via `scripts/shared_scripts/run_retrieval_rerank_pipeline.sh`, which
# orchestrates the following stages:
#
# 1. **BM25** -- sparse keyword retrieval (with optional RM3 query expansion).
# 2. **Dense** -- embedding-based retrieval using a bi-encoder model.
# 3. **Hybrid** -- Reciprocal Rank Fusion (RRF) of BM25 and Dense results.
# 4. **Rerank** -- a cross-encoder re-scores the hybrid candidates.
# 5. **Hybrid + Rerank RRF** -- a second RRF fusion combining rerank scores
#    with the original hybrid scores for the final ranked list.
#
# The pipeline writes TREC-style run files and context JSONs to `output/`.
# In the cells below we:
#
# - visually compare how different retrieval stages rank documents for the
#   same question,
# - score pre-generated answers produced by llama3.3 (a much larger model),
# - evaluate Qwen3-0.6B with injected RAG context for a fair comparison, and
# - optionally fine-tune Qwen3-0.6B with LoRA/DoRA and re-evaluate.

# %% [markdown]
# ### 2.1 Loading retrieval results and document chunks
#
# We load the run files produced by each pipeline stage together with the
# chunked Orange3 documentation (one chunk per JSONL line).

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

# %% [markdown]
# ### 2.1.1 Retrieval quality: MRR curve
#
# The bioasq test file includes a `documents` field listing the ground-truth
# chunk pmids for each question (derived from the source file in
# `orange_qa_full.json`). We compute MRR@k across all five retrieval stages
# to see how retrieval quality improves through the pipeline.
#
# **MRR@k** = mean reciprocal rank of the first relevant chunk in the top-k.
# This metric is unaffected by the number of ground-truth chunks per question,
# making it suitable when ground truth is at the file level.

# %%
import matplotlib.pyplot as plt

bioasq_path = os.path.join(
    PROJECT_ROOT, 'data', 'train_test_dataset_bioasq_style',
    'orange_qa_MCQ_test_bioasq.json',
)
with open(bioasq_path) as f:
    bioasq_data = json.load(f)

gold_map = {}
for q in bioasq_data['questions']:
    docs = q.get('documents', [])
    if docs:
        gold_map[q['id']] = set(docs)

print(f"Ground-truth loaded: {len(gold_map)} questions with documents")

run_files = {
    'BM25': os.path.join(OUTPUT_DIR, 'bm25/runs/BM25__orange_qa_MCQ_test_bioasq__top80.tsv'),
    'Dense': os.path.join(OUTPUT_DIR, 'dense/runs/dense_orange_qa_MCQ_test_bioasq.tsv'),
    'Hybrid': os.path.join(OUTPUT_DIR, 'hybrid/runs/best_rrf_orange_qa_MCQ_test_bioasq_top80.tsv'),
    'Rerank': os.path.join(OUTPUT_DIR, 'rerank/runs/best_rrf_orange_qa_MCQ_test_bioasq_top80.tsv'),
    'Rerank+Hybrid': os.path.join(OUTPUT_DIR, 'rerank_hybrid/runs/best_rrf_orange_qa_MCQ_test_bioasq_top80_rrf_poolR50_poolH50_k60.tsv'),
}

run_dfs = {}
for name, path in run_files.items():
    df = pd.read_csv(path, sep='\t', header=0)
    df['rank'] = df['rank'].astype(int)
    run_dfs[name] = df
    print(f"  {name}: {len(df)} rows, columns={list(df.columns)}")


def compute_mrr(runs_df, gold_map, k):
    scores = []
    for qid, gold in gold_map.items():
        ranked = runs_df[runs_df['qid'] == qid].sort_values('rank')['docno'].tolist()[:k]
        rr = 0.0
        for i, doc in enumerate(ranked):
            if doc in gold:
                rr = 1.0 / (i + 1)
                break
        scores.append(rr)
    return np.mean(scores)


ks = [1, 3, 5, 10, 20]
mrr_results = {
    name: [compute_mrr(df, gold_map, k) for k in ks]
    for name, df in run_dfs.items()
}

# %%
fig, ax = plt.subplots(figsize=(8, 5))
for name, values in mrr_results.items():
    ax.plot(ks, values, marker='o', label=name)
ax.set_xlabel('k')
ax.set_ylabel('MRR@k')
ax.set_title('MRR@k by Retrieval Stage (MCQ)')
ax.set_xticks(ks)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

# %%
print(f"{'Method':<20} " + " ".join(f"{'MRR@'+str(k):>7}" for k in ks))
print("-" * (20 + 8 * len(ks)))
for name in run_dfs:
    mrr_vals = " ".join(f"{v:7.3f}" for v in mrr_results[name])
    print(f"{name:<20} {mrr_vals}")

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

def get_mcq_prompt_with_choices(qid, dataset):
    """Return the full user message (question + options) for a given qid.

    The qid follows the pattern ``<stem>_<index>`` where *index* is the
    0-based position in the dataset list.
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
# *"What is the source of the data in GEO Data Sets?"*
#
# BM25 ranks `vp-sqltable-3` (about SQL/PostgreSQL) first -- a keyword match
# on the wrong domain. Dense retrieval correctly surfaces `bio-geo-data-sets-1`.
# Hybrid (RRF) also ranks the correct chunk first, showing how the dense signal
# can rescue the result despite BM25's keyword mismatch.

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
# ### 2.3 Example 2: Reranking corrects hybrid retrieval
#
# *"Which calibration method uses Isotonic Regression?"*
#
# BM25 correctly finds `vp-calibratedlearner`, but dense retrieval ranks
# `vp-calibrationplot-2` (the plot widget, not the learner) higher. Because
# hybrid fuses both lists, the wrong chunk ends up at rank 1.
#
# The cross-encoder reranker re-scores the hybrid candidates and promotes
# `vp-calibratedlearner` back to rank 1, demonstrating how reranking can
# correct errors introduced by the initial retrieval fusion.

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
# ### 2.4 Scoring pre-generated llama3.3 RAG answers
#
# The retrieval pipeline also generated answers using **llama3.3** (via Ollama)
# with the retrieved contexts. Those answers were converted into the same
# chat-message format as our MCQ test set. We score them here by comparing
# the predicted answer letter to the ground truth.
#
# **Note:** This is *not* an apples-to-apples comparison with the Qwen3-0.6B
# baseline because llama3.3 is a much larger model. Section 2.5 provides a
# fairer comparison by running the same small model with RAG context.

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
# ### 2.5 Evaluating Qwen3-0.6B with RAG context
#
# For a fair comparison with the baseline, we inject the same retrieved contexts
# into the MCQ user prompts and let Qwen3-0.6B answer. The helper
# `build_rag_eval_dataset` prepends a `Context:` block (built from the top-k
# retrieved chunks) to each user message.

# %%
from build_rag_eval_dataset import build_rag_eval_dataset

evidence_mcq_path = Path(OUTPUT_DIR) / 'evidence_baseline' / 'orange_qa_MCQ_test_bioasq_contexts.json'
evidence_mcq_con_path = Path(OUTPUT_DIR) / 'evidence_baseline' / 'orange_qa_MCQ-con_test_bioasq_contexts.json'

# Build RAG-augmented datasets for top_k=1 and top_k=3.
rag_mcq_dataset_k1 = build_rag_eval_dataset(
    contexts_path=evidence_mcq_path,
    original_mcq_path=Path(TESTDATA_MCQ_FILE),
    top_k=1,
    restrict_to_contexts=False,
)
rag_mcq_con_dataset_k1 = build_rag_eval_dataset(
    contexts_path=evidence_mcq_con_path,
    original_mcq_path=Path(TESTDATA_MCQ_CON_FILE),
    top_k=1,
    restrict_to_contexts=False,
)
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

print(f"RAG-augmented MCQ (top_k=1): {len(rag_mcq_dataset_k1)} questions")
print(f"RAG-augmented MCQ-con (top_k=1): {len(rag_mcq_con_dataset_k1)} questions")
print(f"RAG-augmented MCQ (top_k=3): {len(rag_mcq_dataset)} questions")
print(f"RAG-augmented MCQ-con (top_k=3): {len(rag_mcq_con_dataset)} questions")

# %% [markdown]
# The original MCQ prompts tell the model to answer *"based on your knowledge"*.
# The RAG pipeline instead instructs the model to answer using *only* the
# provided context. We patch the system and user messages so Qwen3-0.6B
# receives the same kind of instruction.

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

rag_mcq_dataset_k1 = patch_rag_prompts(rag_mcq_dataset_k1)
rag_mcq_con_dataset_k1 = patch_rag_prompts(rag_mcq_con_dataset_k1)
rag_mcq_dataset = patch_rag_prompts(rag_mcq_dataset)
rag_mcq_con_dataset = patch_rag_prompts(rag_mcq_con_dataset)

# %%
print("Patched system message:")
print(rag_mcq_dataset[0]['messages'][0]['content'])
print("\nSample: RAG-augmented user message (first 5000 chars):")
print(rag_mcq_dataset[0]['messages'][1]['content'][:5000])

# %% [markdown]
# #### 2.5.1 Inspecting raw Qwen3-0.6B generations (debug)
#
# Before trusting the accuracy numbers it is useful to eyeball a few raw
# generations. The helper below prints, for a small sample:
#
# - the (truncated) user prompt,
# - the gold answer letter,
# - the raw decoded output (does the model emit just a letter, or a full
#   sentence? do any `<think>` tags leak through despite `/no_think`?),
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
# Uncomment to inspect raw generations (try rag_mcq_dataset_k1 or test_mcq_dataset too):
# debug_qwen_outputs(rag_mcq_dataset, num_examples=5)

# %%
print("Evaluating Qwen3-0.6B with RAG context (top_k=1)...")
accuracy_mcq_rag_k1, se_mcq_rag_k1 = evaluate_model(model, tokenizer, rag_mcq_dataset_k1, batch_size=MODEL_CONFIG['batch_size'])
accuracy_mcq_con_rag_k1, se_mcq_con_rag_k1 = evaluate_model(model, tokenizer, rag_mcq_con_dataset_k1, batch_size=MODEL_CONFIG['batch_size'])
print(f"  MCQ: {accuracy_mcq_rag_k1:.1f}%, MCQ-con: {accuracy_mcq_con_rag_k1:.1f}%")

print("\nEvaluating Qwen3-0.6B with RAG context (top_k=3)...")
accuracy_mcq_rag, se_mcq_rag = evaluate_model(model, tokenizer, rag_mcq_dataset, batch_size=MODEL_CONFIG['batch_size'])
accuracy_mcq_con_rag, se_mcq_con_rag = evaluate_model(model, tokenizer, rag_mcq_con_dataset, batch_size=MODEL_CONFIG['batch_size'])
print(f"  MCQ: {accuracy_mcq_rag:.1f}%, MCQ-con: {accuracy_mcq_con_rag:.1f}%")

# %% [markdown]
# ### 2.6 Fine-tuned Qwen3-0.6B + RAG (optional)
#
# So far we have compared:
#
# - base Qwen3-0.6B without context,
# - base Qwen3-0.6B with RAG context, and
# - llama3.3 with RAG context.
#
# To see whether domain-specific fine-tuning helps *in addition to* RAG, we
# train a lightweight LoRA/DoRA adapter on the Orange QA training set and
# re-evaluate on the same RAG-augmented MCQ datasets. The adapter configuration
# follows the
# [LoRA fine-tuning tutorial](https://github.com/MartinSpendl/LoRA-finetuning-tutorial).
#
# **This step is GPU-intensive and optional.** Skip it if you do not have
# sufficient resources.

# %%
from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, IA3Config, get_peft_model
from trl import SFTTrainer

FT_CONFIG = {
    "use_dora": True,
    "n_epochs": 1,
    "lora_r": 8,
    "lora_alpha": 16,
    "lr": 5e-4,
    "batch_size": MODEL_CONFIG["batch_size"],
    "lora_projections": "qvko",
    "lora_dropout": 0.05,
    "use_ia3": False,
}

PROJECTIONS = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "g": "gate_proj",
    "d": "down_proj",
    "u": "up_proj",
}
target_modules = [PROJECTIONS[p] for p in list(FT_CONFIG["lora_projections"])]

print("Setting up LoRA/DoRA configuration for fine-tuning...")
peft_config = LoraConfig(
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
    use_dora=FT_CONFIG["use_dora"],
    r=FT_CONFIG["lora_r"],
    lora_alpha=FT_CONFIG["lora_alpha"],
    lora_dropout=FT_CONFIG["lora_dropout"],
)
if FT_CONFIG["use_ia3"]:
    peft_config = IA3Config(
        peft_type="IA3",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

ft_model = get_peft_model(model, peft_config)

train_data_path = os.path.join(DATA_DIR, "orange_qa_train.jsonl")
print(f"Loading training data from {train_data_path} ...")
train_dataset = load_dataset("json", data_files=train_data_path, split="train")

ft_output_dir = os.path.join(PROJECT_ROOT, "models", "Qwen3-0.6B_DoRA_qvko_r8_alpha16_rag")
training_args = TrainingArguments(
    output_dir=ft_output_dir,
    num_train_epochs=FT_CONFIG["n_epochs"],
    per_device_train_batch_size=FT_CONFIG["batch_size"],
    gradient_accumulation_steps=1,
    learning_rate=FT_CONFIG["lr"],
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    save_strategy="epoch",
    report_to=[],
)

trainer = SFTTrainer(
    model=ft_model,
    train_dataset=train_dataset,
    args=training_args,
    processing_class=tokenizer,
)

print("Starting fine-tuning (this may take a while)...")
trainer.train()
print("Fine-tuning complete.")

# %%
print("Evaluating fine-tuned Qwen3-0.6B with RAG context on MCQ...")
accuracy_mcq_rag_ft, se_mcq_rag_ft = evaluate_model(ft_model, tokenizer, rag_mcq_dataset, batch_size=FT_CONFIG['batch_size'])
print(f"Fine-tuned MCQ with RAG: accuracy={accuracy_mcq_rag_ft:.1f}%, SE={se_mcq_rag_ft:.2f}")

print("\nEvaluating fine-tuned Qwen3-0.6B with RAG context on MCQ-con...")
accuracy_mcq_con_rag_ft, se_mcq_con_rag_ft = evaluate_model(ft_model, tokenizer, rag_mcq_con_dataset, batch_size=FT_CONFIG['batch_size'])
print(f"Fine-tuned MCQ-con with RAG: accuracy={accuracy_mcq_con_rag_ft:.1f}%, SE={se_mcq_con_rag_ft:.2f}")

# %% [markdown]
# ### 2.7 Summary
#
# The table below collects all settings evaluated in this tutorial.
# The base model is tested with RAG at top_k = 1 and top_k = 3; the
# fine-tuned variant uses top_k = 3 only.

# %%
print("=" * 75)
print(f"{'Setting':<45} {'MCQ':>10} {'MCQ-con':>10}")
print("-" * 75)
print(f"{'Qwen3-0.6B (no context)':<45} {'5.0%':>10} {'13.5%':>10}")
print(f"{'Qwen3-0.6B + LoRA (no RAG)':<45} {'64.0%':>10} {'13.5%':>10}")
print(f"{'Qwen3-0.6B + RAG (top_k=1)':<45} {accuracy_mcq_rag_k1:>9.1f}% {accuracy_mcq_con_rag_k1:>9.1f}%")
print(f"{'Qwen3-0.6B + RAG (top_k=3)':<45} {accuracy_mcq_rag:>9.1f}% {accuracy_mcq_con_rag:>9.1f}%")
print(f"{'Qwen3-0.6B + RAG (fine-tuned, top_k=3)':<45} {accuracy_mcq_rag_ft:>9.1f}% {accuracy_mcq_con_rag_ft:>9.1f}%")
print(f"{'llama3.3 + RAG context':<45} {acc_llama_mcq:>9.1f}% {acc_llama_mcq_con:>9.1f}%")
print("=" * 75)

# %%
