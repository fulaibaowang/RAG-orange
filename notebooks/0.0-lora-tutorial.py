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

# %% [markdown] id="9cd7c256"
# # Hands-on LoRA finetuning tutorial on Orange3 QA & MCQ dataset

# %% [markdown] id="01a32a07"
# ### Preparing the environment

# %% colab={"base_uri": "https://localhost:8080/"} id="22242794" outputId="c79f1cfe-8599-43ca-8e79-68bc232561e8"
# @title 🛠️ Setup: Optional Clone & Install (commented out)
import os
import sys

# Optional: clone original tutorial repo (kept here for reference, commented out)
REPO_URL = "https://github.com/MartinSpendl/LoRA-finetuning-tutorial.git"
REPO_NAME = "LoRA-finetuning-tutorial"
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
# !pip install -q -r {REPO_NAME}/requirements.txt

print("Using local project files. External clone/install is kept above but commented out.")

# %% id="12a95f4e"
import os
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel, IA3Config
from trl import SFTTrainer

import sys
sys.path.append('../src')
from store_load_results import store_results, load_results
from evaluation_function import evaluate_model
from utils import define_model_name

# %% [markdown] id="19405c6b"
# ## PART 1: Basic Finetuning with LoRA using PEFT
#
# ### Defining the model config

# %% colab={"base_uri": "https://localhost:8080/"} id="839c457e" outputId="be3afa32-7457-4fa5-fa1d-101e950d3725"
MODEL_CONFIG = {
    "base_model": "Qwen/Qwen3-0.6B",
    "finetuning": True,
    "use_dora": True,
    "n_epochs": 1,
    "lora_r": 8,
    "lora_alpha": 16,
    "lr": 5e-4,
    "batch_size": 8,
    "lora_projections": "qvko",
    "lora_dropout": 0.05,
    "new_tokens_path": None,
    "new_tokens_init": "random",
    "new_tokens_train": True,
    "use_ia3": False
}

PROJECTIONS = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "g": "gate_proj",
    "d": "down_proj",
    "u": "up_proj"
}

projections = [PROJECTIONS[p] for p in list(MODEL_CONFIG["lora_projections"])]
MODEL_CONFIG['lora_projections'] = projections
model_name, OUTPUT_DIR = define_model_name(MODEL_CONFIG)

wandb_project = "qwen3-lora-finetuning"
wandb_run_name = model_name
os.environ["WANDB_PROJECT"] = wandb_project

print("Model configuration:")
for key, value in MODEL_CONFIG.items():
    print(f"   {key}: {value}")

# %% [markdown] id="e7799f7a"
# ## Training dataset
#
# Let's look at the training dataset, and what kind of data we have.

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["cdbf08f80d0344368a9cb95d320dc7ac", "ea204f2a3efd4a13a34d4b1743d695b3", "36d2ccbcb97d4f6c8438ed8d6e726e92", "7eff912eb22e4865ab243808598d34cd", "573501d1c0224731a8d295accb8c3af5", "52d28c1bcc8a4823a6c458614984b07c", "f920c26d712d44b08afee1e0210ac58e", "ddaa9883c69b4041a3d048336da3c087", "2ca8c861960c4d62a7c18da534ba813a", "58add836a5204360913fa03329da132d", "3e3ea36b8293457994403f76bfb6cf54"]} id="d3c8e619" outputId="26c5fffc-df0a-4ce6-f15b-1e8af990741b"
if "__file__" in globals():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'train_test_dataset')
TRAINDATA_FILE = os.path.join(DATA_DIR, 'orange_qa_train.jsonl')
TESTDATA_MCQ_FILE = os.path.join(DATA_DIR, 'orange_qa_MCQ_test.jsonl')
TESTDATA_MCQ_CON_FILE = os.path.join(DATA_DIR, 'orange_qa_MCQ-con_test.jsonl')

dataset = load_dataset("json", data_files=TRAINDATA_FILE, split="train")

# %% colab={"base_uri": "https://localhost:8080/"} id="f0b21b65" outputId="132b14fc-a91b-459f-e2d9-034fcb2da00c"
# check different question types:
# QA: 7
# MCQ: 579
# QA - connection: 0
# MCQ - conection: 1012

ID = 0
sample = dataset[ID]['messages']
for message in sample:
    print("Role:", message['role'])
    print(message['content'])
    print()

# %% [markdown] id="19d98174"
# ### Load the model and tokenizer
#
# Let's get familiar with the model and tokenizer. Specifically look at what layers the model has.

# %% colab={"base_uri": "https://localhost:8080/", "height": 104, "referenced_widgets": ["606189ccdd40491c8d66c7398eda639a", "8f2aea3fa811405ab7706523f6d395d4", "e4936d5af6ba4b7dbd76a0298d8579db", "006c1db209d14fcf964191109641e2d7", "a4c8d768e60942baa696cc23cd1701f7", "b6c06fd7eb644bdf80ad8d0370c9f274", "fc16105b0f3d4006ab4c0ae20d71751a", "332435ca20d6477c86b55233e5074215", "47e19c62ab3740ab92caaa72cd84fd3c", "17010d24d9df4d11bbb9e21ae11dd9c0", "04e945c95e894c988a0fb7a69af02276"]} id="84f54aca" outputId="499d4e80-356c-419c-f6d8-ef0f4c108658"
# 4. Load Model & Tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['base_model'])
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG['base_model'],
    dtype=torch.float16,       # Use float16 to save memory
    device_map="auto"          # Auto-selects GPU or CPU
)

# Baseline evaluation before LoRA fine-tuning
print("Running baseline evaluation on MCQ and MCQ-connection test sets before fine-tuning...")
with open(TESTDATA_MCQ_FILE, "r") as f:
    test_mcq_dataset = json.load(f)
accuracy_mcq_base, se_mcq_base = evaluate_model(model, tokenizer, test_mcq_dataset, batch_size=MODEL_CONFIG['batch_size'])

with open(TESTDATA_MCQ_CON_FILE, "r") as f:
    test_mcq_con_dataset = json.load(f)
accuracy_mcq_con_base, se_mcq_con_base = evaluate_model(model, tokenizer, test_mcq_con_dataset, batch_size=MODEL_CONFIG['batch_size'])

baseline_results = {
    "accuracy_mcq": accuracy_mcq_base,
    "se_mcq": se_mcq_base,
    "accuracy_mcq_con": accuracy_mcq_con_base,
    "se_mcq_con": se_mcq_con_base,
}
print("Baseline Evaluation Results:", baseline_results)
baseline_config = {**MODEL_CONFIG, "finetuning": False}
store_results(baseline_results, baseline_config)
print("Baseline results stored successfully.")

# %% colab={"base_uri": "https://localhost:8080/"} id="8693ad72" outputId="7a281b1b-40b8-483c-b528-93aff446184e"
model

# %% colab={"base_uri": "https://localhost:8080/"} id="2998bb2f" outputId="6b18a292-1bb4-4835-d730-9b04c77f157f"
tokenizer

# %% colab={"base_uri": "https://localhost:8080/"} id="01ea7d7b" outputId="0dede908-859c-4538-b59d-01e0968e1556"
# "Orange data mining", "widget", "Hierarchical Clustering", "-->", "-x->"
tokenizer(["Orange Data Mining", "orange data mining"])

# %% [markdown] id="ed26506a"
# ### PEFT Module for LoRA/DoRA

# %% id="3221edb7"
peft_config = LoraConfig(
    bias="none",
    task_type="CAUSAL_LM",
    # from config
    target_modules=MODEL_CONFIG['lora_projections'],
    use_dora=MODEL_CONFIG['use_dora'],
    r=MODEL_CONFIG['lora_r'],
    lora_alpha=MODEL_CONFIG['lora_alpha'],
    lora_dropout=MODEL_CONFIG['lora_dropout'],
    modules_to_save=None,
)
if MODEL_CONFIG['use_ia3']:
    peft_config = IA3Config(
        peft_type="IA3",
        task_type="CAUSAL_LM",
        target_modules=MODEL_CONFIG['lora_projections'],
    )
    MODEL_CONFIG['model_name'] = MODEL_CONFIG['model_name'].replace("LoRA", "IA3").replace("DoRA", "IA3")
model = get_peft_model(model, peft_config)

# %% colab={"base_uri": "https://localhost:8080/"} id="3019c0c9" outputId="737102f5-f685-4b5b-f7b9-f32621c01432"
model

# %% colab={"base_uri": "https://localhost:8080/"} id="161045fe" outputId="29812b57-e7de-48c9-db84-ea48e646b305"
for name, param in model.base_model.model.model.layers[0].self_attn.named_parameters():
    print(name, param.requires_grad)


# %% [markdown] id="64671df3"
# ## Setup training arguments and Train

# %% colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["7936b10a9050473fa7371cc729cfcf69", "bae69cac699649d9b187a461cda0492f", "25f4af23b984445591dc9c6a4f7486bf", "4bdb685ca8724b859c77a5ecedf5e807", "6133b1fb9f6f47f8bd51aacd71a49c4f", "ca15b1c04b2345b69e35661d7ff5ec7e", "a120fbf7b852409fa5d4d20c9a1cd728", "a9500e8c9c014cfc90c979b96410575b", "c07c20eabc8b4382862fbaf67e63d417", "cba73513653749c4b6cdf84a44a0c79c", "7be7aa3a69174ef2a0bd1784e7db182d", "3c2f9dae3ea74d3797d49873e9edcdc9", "5fd5e4fff8e74b0da541c071dacc562c", "17d5094da2f44171bf09d3c093fa1eee", "749ce64d00534ed6ae1cbbff8711d57b", "8f82a19cc7ca4a149d16847041a24d67", "f4106f62005c49d7a2088c7ed02b651b", "f026cdd16e1042289f3480ff5341c0cc", "e2a0ad1d2e7b4f639db0bd82d529df3d", "fd55c45b8ffb4ae1bb8a5a86e292a23c", "2e6e2de744de42a5bcb62ca658532471", "bbd1ed2222844a3ab546faaaa290a537"]} id="6cd7e261" outputId="15bbb1d8-4005-42d9-aea7-4d92e55ccccb"
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=MODEL_CONFIG['n_epochs'],          # How many times to read the docs
    per_device_train_batch_size=MODEL_CONFIG['batch_size'],
    gradient_accumulation_steps=1,
    learning_rate=MODEL_CONFIG['lr'],
    fp16=True,                   # Use mixed precision
    logging_steps=10,
    optim="adamw_torch",
    save_strategy="epoch",       # Save a checkpoint every epoch
    report_to=["wandb"],         # Enable wandb logging
    run_name=wandb_run_name,     # Set run name to model name
)

# 8. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="09ec7cc8" outputId="be76396c-ddee-4679-b6a8-9fed780a2f84"
trainer.train()

# %% [markdown] id="8c5f4bf0"
# ### Evaluate the model on MCQ and MCQ-connection test sets

# %% colab={"base_uri": "https://localhost:8080/"} id="7a8b75f1" outputId="529a01c2-e556-4938-ce0a-701aa996cc71"
# 12. Evaluate and store results
with open(TESTDATA_MCQ_FILE, "r") as f:
    test_mcq_dataset = json.load(f)
accuracy_mcq, se_mcq = evaluate_model(model, tokenizer, test_mcq_dataset, batch_size=MODEL_CONFIG['batch_size'])

with open(TESTDATA_MCQ_CON_FILE, "r") as f:
    test_mcq_con_dataset = json.load(f)
accuracy_mcq_con, se_mcq_con = evaluate_model(model, tokenizer, test_mcq_con_dataset, batch_size=MODEL_CONFIG['batch_size'])
results = {
    "accuracy_mcq": accuracy_mcq,
    "se_mcq": se_mcq,
    "accuracy_mcq_con": accuracy_mcq_con,
    "se_mcq_con": se_mcq_con,
}
print("Evaluation Results:", results)
store_results(results, MODEL_CONFIG)
print("Results stored successfully.")

# %% [markdown] id="5e0add05"
# ## PART 2: Token-injection of Orange3 widgets
#
# If you are running out of GPU memory, you can restart the session and run the following cell again.
#
# Just make sure to also run cells from here onwards.

# %% id="09220ef6"
## Delete if still in memory to free up space
del model
del tokenizer
torch.cuda.empty_cache()

# %% id="aae3e10c"
import os
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

import sys
sys.path.append('LoRA-finetuning-tutorial/src')
sys.path.append('../src')
from store_load_results import store_results, load_results
from evaluation_function import evaluate_model
from utils import define_model_name

# %% colab={"base_uri": "https://localhost:8080/"} id="4802526f" outputId="a6a86a4c-8de7-4c35-c88d-39c438736bf2"
MODEL_CONFIG = {
    "base_model": "Qwen/Qwen3-0.6B",
    "finetuning": True,
    "use_dora": True,
    "n_epochs": 1,
    "lora_r": 8,
    "lora_alpha": 16,
    "lr": 5e-4,
    "batch_size": 8,
    "lora_projections": "qvko",
    "lora_dropout": 0.05,
    "new_tokens_path": "LoRA-finetuning-tutorial/data/injected_tokens.json", ## path to new tokens file
    "new_tokens_init": "random",
    "new_tokens_train": True,
}

PROJECTIONS = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "g": "gate_proj",
    "d": "down_proj",
    "u": "up_proj"
}

projections = [PROJECTIONS[p] for p in list(MODEL_CONFIG["lora_projections"])]
MODEL_CONFIG['lora_projections'] = projections
model_name, OUTPUT_DIR = define_model_name(MODEL_CONFIG)

wandb_project = "qwen3-lora-finetuning"
wandb_run_name = model_name
os.environ["WANDB_PROJECT"] = wandb_project

print("Model configuration:")
for key, value in MODEL_CONFIG.items():
    print(f"   {key}: {value}")

# %% id="f7dab907"
if "__file__" in globals():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'train_test_dataset')
TRAINDATA_FILE = os.path.join(DATA_DIR, 'orange_qa_train.jsonl')
TESTDATA_MCQ_FILE = os.path.join(DATA_DIR, 'orange_qa_MCQ_test.jsonl')
TESTDATA_MCQ_CON_FILE = os.path.join(DATA_DIR, 'orange_qa_MCQ-con_test.jsonl')

dataset = load_dataset("json", data_files=TRAINDATA_FILE, split="train")

# %% colab={"base_uri": "https://localhost:8080/", "height": 104, "referenced_widgets": ["6118f09e035a4b72afe73b2f17de7b39", "85c1215ffd8f4dce8a06ab89165938ae", "b6218684477f4d0ca03d3b87c31006d4", "de157ddb743a4dd8a18900be78f81018", "a748d2b756be4cff9f326115fb0df6e4", "f2a680783a8f41009cff077770289e09", "f145be1053b14297aa0061f8be38e0e0", "a9b016c23c21404c86502dc87b5a338b", "f8006fa5bb2243afb26cca84e25a1a29", "d19610502209490fa38f39b6d3ce6f1c", "ca819913c1674641b6bbe078efa674eb"]} id="5bb93bf3" outputId="bb25c5a1-2927-4869-faf9-9b4d660248ef"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['base_model'])
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG['base_model'],
    dtype=torch.float16,       # Use float16 to save memory
    device_map="auto",          # Auto-selects GPU or CPU
)

# %% [markdown] id="eb4895a5"
# ### Lets check the tokenizer again, and how it handles some common tokens

# %% colab={"base_uri": "https://localhost:8080/"} id="f7eaaf20" outputId="ab459446-77df-435f-c8f2-a6157a5d9836"
tokenizer

# %% colab={"base_uri": "https://localhost:8080/"} id="16fcbad3" outputId="5fdc3085-f846-45b2-bcc0-13cd1466a7fb"
tokenizer(["Hierarchical Clustering", "-->", "-x->", "Logistic Regression", "Hierarchical Clustering -x-> Logistic Regression"])

# %% [markdown] id="b936c5b2"
# ### Adding new tokens to tokenizer & Extending model embeddings

# %% colab={"base_uri": "https://localhost:8080/"} id="cfe1bf17" outputId="1144dbd4-d73a-4036-ae02-984b5b58fe05"
from transformers import AddedToken

# 5. Extend token library (tokenizer + model)
if MODEL_CONFIG['new_tokens_path'] is not None:
    print("Adding new tokens from:", MODEL_CONFIG['new_tokens_path'])
    DEFAULT_TOKEN_NUM = len(tokenizer)

    with open(MODEL_CONFIG['new_tokens_path'], 'r') as f:
        new_tokens_text = json.load(f)
    new_token_tokenized = [tokenizer.encode(token) for token in new_tokens_text]
    new_tokens = [
        AddedToken(token, lstrip=True, rstrip=True)
        for token in new_tokens_text
    ]

    ## extend tokenizer and model
    print("Number of default tokenizer tokens:", DEFAULT_TOKEN_NUM)
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print("Number of new tokenizer tokens:", len(tokenizer))

    new_tokens_ids = [
        tokenizer.encode(token)[0]
        for token in new_tokens_text
    ]
    old_tokens_ids = [id for id in range(DEFAULT_TOKEN_NUM) if id not in new_tokens_ids]

    ## initialize new token embeddings
    model.model.embed_tokens = model.model.embed_tokens.float()

    if MODEL_CONFIG['new_tokens_init'] == "average":
        with torch.no_grad():
            existing_embeddings = model.model.embed_tokens.weight
            for new_id, previous_token_ids in zip(new_tokens_ids, new_token_tokenized):
                model.model.embed_tokens.weight[new_id, :] = existing_embeddings[previous_token_ids].mean(dim=0)
    elif MODEL_CONFIG['new_tokens_init'] == "zero":
        with torch.no_grad():
            model.model.embed_tokens[DEFAULT_TOKEN_NUM:, :].fill_(0.0)
    elif MODEL_CONFIG['new_tokens_init'] == "random":
        pass  # already quazi randomly initialized
    else:
        raise ValueError(f"Unknown new_tokens_init method: {MODEL_CONFIG['new_tokens_init']}")

    ## setup trainable weights for new tokens
    if MODEL_CONFIG['new_tokens_train']:
        def zero_out_old_token_grads(grad):
            new_grad = grad.clone()
            new_grad[old_tokens_ids, :] = 0.0
            return new_grad
        model.model.embed_tokens.weight.requires_grad = True
        model.model.embed_tokens.weight.register_hook(zero_out_old_token_grads)
    model.lm_head.weight.requires_grad = True

# %% [markdown] id="44706a9a"
# ### Let's test our tokenizer with new tokens

# %% colab={"base_uri": "https://localhost:8080/"} id="c5c065ce" outputId="ed06dd4b-05ee-40f0-8e9e-30dce7b688e7"
tokenizer

# %% colab={"base_uri": "https://localhost:8080/"} id="e38acdbf" outputId="54d60799-af4f-4b1a-8e89-8e7501713d9d"
tokenizer(["Hierarchical Clustering", "-->", "-x->", "Logistic Regression", "Hierarchical Clustering -x-> Logistic Regression"])

# %% colab={"base_uri": "https://localhost:8080/"} id="2692db3b" outputId="caadf25b-dcf5-4b9e-efed-1355927330d2"
if MODEL_CONFIG['new_tokens_path'] is not None:
    modules_to_save = ["lm_head"] + (["embed_tokens"] if MODEL_CONFIG['new_tokens_train'] else [])
else:
    modules_to_save = None

peft_config = LoraConfig(
    bias="none",
    task_type="CAUSAL_LM",
    # from config
    target_modules=MODEL_CONFIG['lora_projections'],
    use_dora=MODEL_CONFIG['use_dora'],
    r=MODEL_CONFIG['lora_r'],
    lora_alpha=MODEL_CONFIG['lora_alpha'],
    lora_dropout=MODEL_CONFIG['lora_dropout'],
    modules_to_save=modules_to_save,
)
model = get_peft_model(model, peft_config)

# %% colab={"base_uri": "https://localhost:8080/"} id="049d89cc" outputId="37f42483-746d-482a-9633-a3d865e05601"
print("LM head module requires grad: ", model.base_model.model.lm_head.modules_to_save.default.weight.requires_grad)
print("Embed tokens module requires grad: ", model.base_model.model.model.embed_tokens.modules_to_save.default.weight.requires_grad)

# %% colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["93d4f56024ad45308664be4332395e95", "99635bea8d5d4d1ca33b4ddd4369d3cb", "4927960e76f344889c790b20215b690a", "5c69a16850aa45ff970abfa48055bffe", "ddac808fa0c74d129c6ff57cfa303d67", "2f1b7e170064420495baf2f59d5ad91f", "aa88f251faf4489fb120e27624348e68", "f4e3e289e2bf42dfa37a41cc7254ae90", "93cf779e31324377b57b1a3bc0cfe227", "8b4fc0daf2b943a5ab000f16d092549c", "771e5f963fba4ed7be6212cc34114566", "a289534dd00f4aeeaa7dc54524aab0aa", "f208f05e6b1d45a4b21271d4456b0421", "e39a3ada664f4aedb56ad494153a220c", "a305b753174246e6aa8eb586cf810dae", "e18e82ff166940c8aea3cfaf49129462", "d58a2f7b6d134a1a9098566ea8598824", "1327ed4317444fbea856d8ddc5078b06", "80ae7c9ae28344b08d85f819c3d2d0b8", "b653bd5799034717a455c193134d37bb", "76b6f78c0796476992419a8421eb86f4", "5c2c53360af64e928667601895ac6950"]} id="e9563256" outputId="21d4b06d-d693-4390-ad52-96fd8ac8fce3"
wandb_project = MODEL_CONFIG.get('wandb_project') or os.environ.get("WANDB_PROJECT", "qwen3-lora-finetuning")
wandb_run_name = MODEL_CONFIG['model_name']

# Set wandb project via environment variable
os.environ["WANDB_PROJECT"] = wandb_project

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=MODEL_CONFIG['n_epochs'],          # How many times to read the docs
    per_device_train_batch_size=MODEL_CONFIG['batch_size'],
    gradient_accumulation_steps=1,
    learning_rate=MODEL_CONFIG['lr'],
    fp16=True,                   # Use mixed precision
    logging_steps=10,
    optim="adamw_torch",
    save_strategy="epoch",       # Save a checkpoint every epoch
    report_to=["wandb"],         # Enable wandb logging
    run_name=wandb_run_name,     # Set run name to model name
)

# 8. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="3f2a9fdc" outputId="555c89a6-c61c-48f9-ca04-61042d1f3f4f"
trainer.train()

# %% id="0bf0cea2"
## Training in FP32, now we have to cast back to FP16 to evaluate
model.base_model.model.model.embed_tokens.modules_to_save.default = model.base_model.model.model.embed_tokens.modules_to_save.default.half()
model.base_model.model.lm_head.modules_to_save.default = model.base_model.model.lm_head.modules_to_save.default.half()

# %% colab={"base_uri": "https://localhost:8080/"} id="5e99c829" outputId="731df299-8efd-42a4-83eb-6e05dfb40353"
# 12. Evaluate and store results
with open(TESTDATA_MCQ_FILE, "r") as f:
    test_mcq_dataset = json.load(f)
accuracy_mcq, se_mcq = evaluate_model(model, tokenizer, test_mcq_dataset, batch_size=MODEL_CONFIG['batch_size'])

with open(TESTDATA_MCQ_CON_FILE, "r") as f:
    test_mcq_con_dataset = json.load(f)
accuracy_mcq_con, se_mcq_con = evaluate_model(model, tokenizer, test_mcq_con_dataset, batch_size=MODEL_CONFIG['batch_size'])
results = {
    "accuracy_mcq": accuracy_mcq,
    "se_mcq": se_mcq,
    "accuracy_mcq_con": accuracy_mcq_con,
    "se_mcq_con": se_mcq_con,
}
print("\nEvaluation Results:", results)
store_results(results, MODEL_CONFIG)
print("Results stored successfully.")

# %% id="1oAMPEKflhxj"
