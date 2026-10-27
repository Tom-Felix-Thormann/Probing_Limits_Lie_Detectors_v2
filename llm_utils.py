"""Shared paths and constants of the dataset construction scripts, and the generation helpers for 02_context_generation_task.py."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ORIG_DATASETS_DIR = SCRIPT_DIR / "dataset" / "original_datasets"
DATASETS_DIR = SCRIPT_DIR / "dataset" / "dataset_construction_steps"
MODEL_DIR = SCRIPT_DIR / "model_weights" / "llama-3.1-70B-instruct"

SAMPLED_PATH = DATASETS_DIR / "full_dataset_sampled.csv"            # written by 01
MANUAL_SAMPLED_PATH = DATASETS_DIR / "manual_entries_sampled.xlsx"  # written by 01, to be filled in by hand
MANUAL_PATH = DATASETS_DIR / "manual_entries_completed.xlsx"        # the filled-in version, read by 02
FULL_DATASET_PATH = DATASETS_DIR / "full_dataset_generated.csv"     # written by 02

SEED = 67
TEMPERATURE = 0.7
TOP_P = 0.9


def load_model():
    print(f"Loading model: {MODEL_DIR.name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer.padding_side = "left"  # decoder-only models need left padding for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    model.eval()
    return model, tokenizer


def format_chat(tokenizer, messages):
    """Apply the chat template and return the prompt as text."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def run_generation(model, tokenizer, prompts, max_new_tokens, batch_size, desc):
    """Generate a reply for every prompt text, in batches. Returns the replies in the order of `prompts`.

    Longest prompts go first: similar lengths in a batch means little padding, and an out-of-memory
    error shows up in the first batch. If a batch does not fit into GPU memory, the batch size is
    halved and the batch is retried.
    """
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    replies = [None] * len(prompts)
    start = 0
    while start < len(order):
        batch = order[start:start + batch_size]
        try:
            outputs = generate_batch(model, tokenizer, [prompts[i] for i in batch], max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size //= 2
            print(f"Out of GPU memory, reducing batch size to {batch_size}", flush=True)
            continue
        for i, output in zip(batch, outputs):
            replies[i] = output
        start += len(batch)
        print(f"{desc}: {start}/{len(order)}", flush=True)
    return replies