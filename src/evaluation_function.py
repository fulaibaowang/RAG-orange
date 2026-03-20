import re

import numpy as np
import torch
from tqdm import tqdm

_ANSWER_LETTER_RE = re.compile(r'^([A-Da-d])(?:[^a-zA-Z]|$)')


def _extract_answer_letter(text: str) -> str:
    """Extract the leading answer letter (A-D) from model output.

    Handles common formatting like ``"D: Yes, as .svg or .png"`` by returning
    just ``"D"``.  If no leading letter is found, the full text is returned so
    strict comparison can still work.
    """
    m = _ANSWER_LETTER_RE.match(text)
    if m:
        return m.group(1).upper()
    return text


def evaluate_model(
    model,
    tokenizer,
    dataset,
    batch_size: int = 32,
    return_lenient: bool = False,
):
    """Run greedy generation on *dataset* and compute accuracy.

    Returns
    -------
    (strict_accuracy, strict_se) when *return_lenient* is False (default).
    (strict_accuracy, strict_se, lenient_accuracy, lenient_se) when True.

    *strict*  — the full decoded prediction must equal the gold answer exactly.
    *lenient* — only the leading answer letter (A–D) is compared.
    """
    strict_correct = []
    lenient_correct = []
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

    model.eval()

    with torch.no_grad():
        for batch in tqdm(batches, desc="Evaluating model"):
            batch_texts = []
            batch_assistant_messages = []

            for item in batch:
                system_user_messages = [
                    {"role": item["messages"][0]["role"], "content": item["messages"][0]["content"]},
                    {"role": item["messages"][1]["role"], "content": item["messages"][1]["content"] + "\n /no_think"}
                ]
                assistant_messages = item["messages"][2]
                batch_assistant_messages.append(assistant_messages)

                text = tokenizer.apply_chat_template(
                    system_user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_tracking=False
                )
                batch_texts.append(text)

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            for i, (output, assistant_messages) in enumerate(zip(outputs, batch_assistant_messages)):
                input_length = inputs.input_ids[i].shape[0]
                generated_token_ids = output[input_length:]
                decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

                prediction = decoded_text.split("</think>")[1].strip() if "</think>" in decoded_text else decoded_text.strip()
                gold = assistant_messages["content"]

                strict_correct.append(prediction == gold)
                lenient_correct.append(_extract_answer_letter(prediction) == _extract_answer_letter(gold))

    strict_correct = np.array(strict_correct) * 100
    accuracy = np.mean(strict_correct)
    se = np.std(strict_correct) / np.sqrt(len(strict_correct))

    if not return_lenient:
        return accuracy, se

    lenient_correct = np.array(lenient_correct) * 100
    lenient_accuracy = np.mean(lenient_correct)
    lenient_se = np.std(lenient_correct) / np.sqrt(len(lenient_correct))
    return accuracy, se, lenient_accuracy, lenient_se
