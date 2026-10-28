"""Step 5: extract the activations of all statements for the models in PROBE_MODELS (all but the two 7B models).

Statements (one row each in activations/statements.csv):
    generated rows   true  = original_true_statement (label 1), false = false_answer (label 0)
    manual rows      true  = true_non-misleading_answer (label 1), false = false_answer (label 0),
                     misleading = misleading_non-false_answer (label 0)
The manual rows are split 180 / 180 into train and test, so that every topic, statement type and
deception type is split evenly (see split_manual_rows).

Every statement is passed through the model in two formats:
    raw       the statement alone
    context   a chat-formatted dialogue: the user asks context_with_question, the assistant replies with the statement
As in Truth_is_Universal, the activation is the residual stream (output of one decoder layer) at the
last token of the statement, i.e. its final period; chat tokens after the statement are cut off.

The layer is selected per model as in Truth_is_Universal: the layer with the largest separation of true
and false statements (between-class / within-class variance), averaged over the raw statements of
cities, neg_cities, sp_en_trans and neg_sp_en_trans (generated rows).

Results: activations/statements.csv, activations/<model>.npz (raw, context, layer),
         results/layer_selection/<model>.csv

Usage:
    python 05_activation_extraction.py                # all models in PROBE_MODELS one after another
    python 05_activation_extraction.py --model NAME   # one model only (parallel cluster jobs)
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from llm_utils import (
    ACTIVATIONS_DIR,
    FULL_DATASET_PATH,
    PROBE_MODELS,
    RESULTS_DIR,
    SEED,
    STATEMENTS_PATH,
    format_chat,
    load_model,
    run_in_batches,
)

BATCH_SIZE = 16  # halved automatically if it does not fit into GPU memory
SELECTION_TOPICS = ["cities", "sp_en_trans"]
SELECTION_STATEMENT_TYPES = ["affirmative", "neg"]
LAYER_SELECTION_DIR = RESULTS_DIR / "layer_selection"


def split_manual_rows(manual):
    """Assign the manual rows to "train" or "test", 180 each.

    There is one manual row per topic x statement type x deception type. The topic x statement type
    cells are colored like a chessboard (3 of 6 per topic and per statement type); in one color the
    first half of the deception types goes to train, in the other the second half. So every topic,
    statement type and deception type, and every pair of them, is split evenly.
    """
    rng = np.random.default_rng(SEED)
    topics = list(rng.permutation(sorted(manual["topic"].unique())))
    types = list(rng.permutation(sorted(manual["statement_type"].unique())))
    deception_types = list(rng.permutation(sorted(manual["deception_type"].unique())))
    first_half = set(deception_types[: len(deception_types) // 2])

    black_cell = (manual["topic"].map(topics.index) + manual["statement_type"].map(types.index)) % 2 == 0
    in_first_half = manual["deception_type"].isin(first_half)
    return np.where(black_cell == in_first_half, "train", "test")


def build_statements(df):
    """One row per statement, with its label, the context and the metadata of its dataset row."""
    meta = ["id", "topic", "statement_type", "deception_type", "context_with_question"]
    generated = df[df["manual_or_generated"] == "generated"]
    manual = df[df["manual_or_generated"] != "generated"].copy()
    manual["split"] = split_manual_rows(manual)

    parts = []
    for rows, source, kinds in [
        (generated, "generated", [("true", "original_true_statement", 1), ("false", "false_answer", 0)]),
        (manual, "manual", [
            ("true", "true_non-misleading_answer", 1),
            ("false", "false_answer", 0),
            ("misleading", "misleading_non-false_answer", 0),
        ]),
    ]:
        for kind, column, label in kinds:
            part = rows[meta + (["split"] if source == "manual" else [])].copy()
            part["source"], part["kind"], part["label"] = source, kind, label
            part["statement"] = rows[column].astype(str).str.strip()
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def context_text(tokenizer, context, statement):
    """The chat-formatted dialogue, cut off after the statement, so that its last token is the statement's last token."""
    messages = [{"role": "user", "content": context}, {"role": "assistant", "content": statement}]
    text = format_chat(tokenizer, messages, add_generation_prompt=False)
    start = text.rfind(statement)
    if start < 0:  # e.g. a chat template that changes the message; would otherwise cut the text silently at a wrong place
        raise ValueError(f"Statement not found in the chat-formatted text: {statement!r}")
    return text[: start + len(statement)]


def last_token_acts(model, tokenizer, texts, add_special_tokens, layer, desc):
    """Residual stream at the last token of every text: [n, d] for one layer, or [n, layers, d] for layer=None.

    hidden_states[i + 1] is the output of decoder layer i; the last entry is left out, as it has the
    final norm applied.
    """
    def batch_fn(batch):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens).to(model.device)
        position_ids = (inputs["attention_mask"].cumsum(-1) - 1).clamp(min=0)  # positions ignore the left padding
        with torch.no_grad():
            hidden = model(**inputs, position_ids=position_ids, output_hidden_states=True, use_cache=False).hidden_states
        if layer is None:
            acts = torch.stack([h[:, -1].float().cpu() for h in hidden[1:-1]], dim=1)
        else:
            acts = hidden[layer + 1][:, -1].float().cpu()
        return list(acts.numpy())

    return np.stack(run_in_batches(texts, BATCH_SIZE, batch_fn, desc))


def separation(acts, labels):
    """Between-class / within-class variance of true and false statements per layer (acts: [n, layers, d])."""
    acts = torch.from_numpy(acts)
    labels = torch.tensor(labels)
    true_acts, false_acts = acts[labels == 1], acts[labels == 0]
    within = (true_acts.var(dim=0).mean(dim=-1) + false_acts.var(dim=0).mean(dim=-1)) / 2
    overall = acts.mean(dim=0)
    between = ((true_acts.mean(dim=0) - overall).pow(2) + (false_acts.mean(dim=0) - overall).pow(2)).mean(dim=-1) / 2
    return (between / within).numpy()


def select_layer(model, tokenizer, statements, model_name):
    curves = {}
    for topic in SELECTION_TOPICS:
        for statement_type in SELECTION_STATEMENT_TYPES:
            subset = statements[
                (statements["source"] == "generated")
                & (statements["topic"] == topic)
                & (statements["statement_type"] == statement_type)
            ]
            name = f"{topic}_{statement_type}"
            acts = last_token_acts(
                model, tokenizer, subset["statement"].tolist(), True, None, desc=f"{model_name} layer selection {name}"
            )
            curves[name] = separation(acts, subset["label"].to_numpy())

    curves = pd.DataFrame(curves)
    curves["mean"] = curves.mean(axis=1)
    curves.index.name = "layer"
    curves.to_csv(LAYER_SELECTION_DIR / f"{model_name}.csv")
    layer = int(curves["mean"].idxmax())
    print(f"{model_name}: selected layer {layer} of {len(curves) + 1}", flush=True)
    return layer


def save_statements(statements, tag):
    """Write statements.csv via a temporary file and an atomic rename.

    Parallel jobs (--model) all write the same content; the rename ensures that no job ever reads a
    half-written file.
    """
    tmp = STATEMENTS_PATH.with_name(f"{STATEMENTS_PATH.stem}.{tag}.tmp")
    statements.to_csv(tmp, index=False)
    os.replace(tmp, STATEMENTS_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Step 5: activation extraction.")
    parser.add_argument("--model", choices=PROBE_MODELS, help="extract the activations of this model only (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    LAYER_SELECTION_DIR.mkdir(parents=True, exist_ok=True)

    statements = build_statements(pd.read_csv(FULL_DATASET_PATH))
    save_statements(statements, args.model or "all")
    print(statements.groupby(["source", "kind"]).size().to_string(), flush=True)

    models = [args.model] if args.model else PROBE_MODELS
    for model_name in models:
        model, tokenizer = load_model(model_name)
        layer = select_layer(model, tokenizer, statements, model_name)

        raw_texts = statements["statement"].tolist()
        context_texts = [
            context_text(tokenizer, context, statement)
            for context, statement in zip(statements["context_with_question"], statements["statement"])
        ]
        raw = last_token_acts(model, tokenizer, raw_texts, True, layer, desc=f"{model_name} raw")
        # the chat template already contains the special tokens (e.g. <|begin_of_text|>)
        context = last_token_acts(model, tokenizer, context_texts, False, layer, desc=f"{model_name} context")
        np.savez_compressed(ACTIVATIONS_DIR / f"{model_name}.npz", raw=raw, context=context, layer=layer)

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()