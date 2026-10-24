"""Shared paths, constants and generation helpers for 03_dataset_generation_task.py and 04_answer_generation_task.py."""

import os
from pathlib import Path

import pandas as pd
import torch
from jinja2.exceptions import TemplateError
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR / "dataset"
MODEL_BASE_DIR = SCRIPT_DIR / "model_weights"

SAMPLED_PATH = DATASETS_DIR / "sampled_dataset_with_types.csv"
MANUAL_PATH = DATASETS_DIR / "manual_entries_completed.xlsx"
GENERATION_TASK_PATH = DATASETS_DIR / "generation_task.csv"
ANSWERS_DIR = DATASETS_DIR / "generated_answers"  # one csv per answering model
FULL_DATASET_PATH = DATASETS_DIR / "full_dataset_generated.csv"

SEED = 67
TEMPERATURE = 0.7
TOP_P = 0.9

CONTEXT_MODEL = "llama-3.1-70B-instruct"
ANSWER_MODELS = [
    "gemma-3-27B-instruct",
    "llama-3.1-70B-instruct",
    "llama-3.1-8B-instruct",
    "mistral-7B-instruct",
    "qwen-2.5-72B-instruct",
    "qwen-2.5-7B-instruct",
]
MANUAL_LABELS = ["manual", "manual+claude"]

ANSWER_COLUMNS = [
    "true_non-misleading_answer",
    "false_answer",
    "misleading_non-false_answer",
    "unrelated_or_nonsense",
]
TEXT_COLUMNS = ["context_with_question", *ANSWER_COLUMNS, "manual_or_generated", "generator_model"]


def read_dataset(path):
    """Read a dataset csv. Columns that are still completely empty would otherwise be parsed as float."""
    df = pd.read_csv(path)
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df


def save_csv(df, path):
    """Write to a temporary file first, so that a job killed mid-write never leaves a broken csv behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def load_model(model_name):
    model_path = MODEL_BASE_DIR / model_name
    print(f"Loading model: {model_path.name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"  # decoder-only models need left padding for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except ValueError:
        # gemma-3-27B-instruct is a multimodal checkpoint that some transformers versions
        # do not map to AutoModelForCausalLM
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return model, tokenizer


def format_chat(tokenizer, messages):
    """Apply the chat template and return the prompt as text."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TemplateError:
        if messages[0]["role"] != "system":
            raise
        # Some chat templates (e.g. older Mistral versions) reject the system role.
        # In that case, the system prompt is put in front of the first user message.
        folded = [dict(m) for m in messages[1:]]
        folded[0]["content"] = messages[0]["content"] + "\n\n" + folded[0]["content"]
        return tokenizer.apply_chat_template(folded, tokenize=False, add_generation_prompt=True)


def clean_output(text):
    text = text.strip()
    # remove quotation marks around the whole output (single quotes are kept, they often mark words, e.g. 'luz')
    for left, right in [('"', '"'), ("“", "”"), ("„", "“")]:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right) and text.count(left) + text.count(right) == 2:
            text = text[1:-1].strip()
    return text


def generate_batch(model, tokenizer, prompts, max_new_tokens):
    # the chat template already contains the special tokens (e.g. <|begin_of_text|>)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return [clean_output(t) for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def run_generation(model, tokenizer, prompts, max_new_tokens, batch_size, on_batch, desc):
    """Generate for a list of (row_index, prompt_text) pairs in batches.

    on_batch(row_indices, outputs) is called after every batch, so the caller can store and save results.
    If a batch does not fit into GPU memory, the batch size is halved and the batch is retried.
    """
    # Longest prompts first: similar lengths in a batch means little padding, and an
    # out-of-memory error shows up in the first batch rather than hours into the job.
    ordered = sorted(prompts, key=lambda p: len(p[1]), reverse=True)
    start = 0
    while start < len(ordered):
        batch = ordered[start:start + batch_size]
        try:
            outputs = generate_batch(model, tokenizer, [p for _, p in batch], max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f"Out of GPU memory, reducing batch size to {batch_size}", flush=True)
            continue
        on_batch([i for i, _ in batch], outputs)
        start += len(batch)
        print(f"{desc}: {start}/{len(ordered)}", flush=True)
