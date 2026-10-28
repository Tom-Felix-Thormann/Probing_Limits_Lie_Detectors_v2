"""Shared paths, constants and LLM helpers of all scripts (01-06)."""

from pathlib import Path

import numpy as np
import torch
from jinja2.exceptions import TemplateError
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ORIG_DATASETS_DIR = SCRIPT_DIR / "dataset" / "original_datasets"
DATASETS_DIR = SCRIPT_DIR / "dataset" / "dataset_construction_steps"
MODEL_BASE_DIR = SCRIPT_DIR / "model_weights"
ACTIVATIONS_DIR = SCRIPT_DIR / "activations"
RESULTS_DIR = SCRIPT_DIR / "results"

SAMPLED_PATH = DATASETS_DIR / "full_dataset_sampled.csv"            # written by 01
MANUAL_SAMPLED_PATH = DATASETS_DIR / "manual_entries_sampled.xlsx"  # written by 01, to be filled in by hand
MANUAL_PATH = DATASETS_DIR / "manual_entries_completed.xlsx"        # the filled-in version, read by 02 and 04
FULL_DATASET_PATH = DATASETS_DIR / "full_dataset_generated.csv"     # written by 02, read by 05
STATEMENTS_PATH = ACTIVATIONS_DIR / "statements.csv"                # written by 05: one row per activation vector

GENERATION_MODEL = "llama-3.1-70B-instruct"  # writes the contexts and false answers (02)
MODELS = [  # the models that are tested (04-06)
    "gemma-3-27B-instruct",
    "llama-3.1-70B-instruct",
    "llama-3.1-8B-instruct",
    "mistral-7B-instruct",
    "qwen-2.5-72B-instruct",
    "qwen-2.5-7B-instruct",
]
# the models whose activations are extracted and probed (05-06): without the two 7B models, which do
# not select the misleading answer reliably in the forced choice task (04)
PROBE_MODELS = [m for m in MODELS if m not in ("mistral-7B-instruct", "qwen-2.5-7B-instruct")]

SEED = 67
TEMPERATURE = 0.7
TOP_P = 0.9


def load_model(model_name):
    model_path = MODEL_BASE_DIR / model_name
    print(f"Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"  # left padding: the last position of every row is the last real token
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


def format_chat(tokenizer, messages, add_generation_prompt=True):
    """Apply the chat template and return the conversation as text."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except TemplateError:
        # Some chat templates (e.g. older Mistral versions) reject the system role.
        # In that case, the system prompt is put in front of the first user message.
        folded = [dict(m) for m in messages[1:]]
        folded[0]["content"] = messages[0]["content"] + "\n\n" + folded[0]["content"]
        return tokenizer.apply_chat_template(folded, tokenize=False, add_generation_prompt=add_generation_prompt)


def clean_output(text):
    text = text.strip()
    # remove quotation marks around the whole output (single quotes are kept, they often mark words, e.g. 'luz')
    for left, right in [('"', '"'), ("“", "”"), ("„", "“")]:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right) and text.count(left) + text.count(right) == 2:
            text = text[1:-1].strip()
    return text


def run_in_batches(texts, batch_size, batch_fn, desc):
    """Apply batch_fn(list of texts) -> list of results to all texts. Returns the results in the order of `texts`.

    Longest texts go first: similar lengths in a batch means little padding, and an out-of-memory
    error shows up in the first batch. If a batch does not fit into GPU memory, the batch size is
    halved and the batch is retried.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    results = [None] * len(texts)
    start = 0
    while start < len(order):
        batch = order[start:start + batch_size]
        try:
            outputs = batch_fn([texts[i] for i in batch])
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size //= 2
            print(f"Out of GPU memory, reducing batch size to {batch_size}", flush=True)
            continue
        for i, output in zip(batch, outputs):
            results[i] = output
        start += len(batch)
        print(f"{desc}: {start}/{len(order)}", flush=True)
    return results


def generate_batch(model, tokenizer, prompts, max_new_tokens, greedy=False):
    # the chat template already contains the special tokens (e.g. <|begin_of_text|>)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    if greedy:  # None overrides the sampling defaults in the model's generation_config
        sampling = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        sampling = dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, **sampling
        )
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return [clean_output(t) for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def run_generation(model, tokenizer, prompts, max_new_tokens, batch_size, desc, greedy=False):
    """Generate a reply for every prompt text, in batches. Returns the replies in the order of `prompts`."""
    batch_fn = lambda texts: generate_batch(model, tokenizer, texts, max_new_tokens, greedy)
    return run_in_batches(prompts, batch_size, batch_fn, desc)


def mcnemar_greater(x, y):
    """One-sided exact McNemar test for paired binary outcomes, H1: P(x) > P(y).

    Returns (b, c, p): b = pairs with x but not y, c = pairs with y but not x.
    """
    x, y = np.asarray(x, dtype=bool), np.asarray(y, dtype=bool)
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    p = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c > 0 else 1.0
    return b, c, p


def wilson_ci(x, confidence_level=0.95):
    """Wilson confidence interval of the share of True in the binary outcomes x. Returns (low, high)."""
    x = np.asarray(x, dtype=bool)
    ci = binomtest(int(x.sum()), len(x)).proportion_ci(confidence_level=confidence_level, method="wilson")
    return ci.low, ci.high


def holm(p_values):
    """Holm-adjusted p-values (family-wise error rate), in the order of p_values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    # the i-th smallest p-value (from 0) is multiplied by m - i; the running maximum keeps the order of the adjusted values
    adjusted_sorted = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted