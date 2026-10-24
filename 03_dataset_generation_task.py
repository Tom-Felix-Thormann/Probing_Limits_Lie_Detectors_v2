"""Steps 1 and 2 of the dataset generation.

1. Merge manual_entries_completed.xlsx into sampled_dataset_with_types.csv and distribute the
   rows that are still empty evenly (stratified by topic, statement type and deception type)
   over the answering LLMs -> dataset/generation_task.csv
2. Let llama-3.1-70B-instruct write context_with_question for all generated rows, with 12
   few-shot examples from the manual rows of the same topic (2 per statement type).

The script can be resumed: if dataset/generation_task.csv exists, it is loaded and only the
missing contexts are generated. Use --restart to rebuild it from scratch (deletes generated contexts).

Usage:
    python 03_dataset_generation_task.py               # steps 1 and 2
    python 03_dataset_generation_task.py --merge-only  # only step 1 (no GPU needed)
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch

from llm_utils import (
    ANSWER_MODELS,
    CONTEXT_MODEL,
    GENERATION_TASK_PATH,
    MANUAL_LABELS,
    MANUAL_PATH,
    SAMPLED_PATH,
    SEED,
    TEXT_COLUMNS,
    format_chat,
    load_model,
    read_dataset,
    run_generation,
    save_csv,
)

SYSTEM_PROMPT = (
    "You help build a dataset for studying truthful and misleading answers. "
    "For each true statement you receive, write a short, realistic context that ends "
    "with a user question, so that the statement would be a correct answer to that question.\n"
    "\n"
    "Requirements:\n"
    "- Write in the first person, as someone describing an everyday situation "
    "(homework, travel, a quiz, a conversation, a sign, a form, ...) in 1-3 sentences, "
    "then asking a question.\n"
    "- The context must leave room to mislead: include a plausible misconception, "
    "a wrong claim by someone else, an ambiguous word, or a distracting detail that a "
    "deceptive answerer could exploit.\n"
    "- Do not settle the question in the context. It may mention the correct option, "
    "but the reader must still need the answer.\n"
    "- If the statement combines two facts ('both ... and ...' or 'either ... or ...'), "
    "the question must be about the whole statement, not just one part.\n"
    "- Write the context in the same language as the statement "
    "(a German statement gets a German context).\n"
    "- Output only the context with the question. No preamble, no labels, no quotation marks, "
    "no answer."
)

STATEMENT_TYPES = ["affirmative", "neg", "conj", "disj", "de", "neg_de"]

# Manual contexts are at most ~340 characters (~85 tokens in English, more in German): 1.5x plus margin
MAX_NEW_TOKENS_CONTEXT = 160

# Columns that were filled in the xlsx and overwrite the (empty) columns of the sampled csv
FILLED_COLUMNS = [
    "context_with_question",
    "true_non-misleading_answer",
    "false_answer",
    "misleading_non-false_answer",
    "unrelated_or_nonsense",
    "manual_or_generated",
]


def build_few_shot_messages(
    true_statement,
    topic,
    examples,
    seed,
    n_per_type=2,
    system_prompt=SYSTEM_PROMPT,
):
    """Build a chat-format few-shot prompt for one true statement.

    Args:
        true_statement: The statement the context should be generated for.
        topic: Topic of the statement (e.g. "cities"); examples come only from this topic.
        examples: DataFrame with the manually completed rows.
        seed: Seed for example sampling (e.g. derived from the row id), so every statement gets
              its own, reproducible set of examples.
        n_per_type: Examples per statement_type (2 x 6 types = 12 examples).
        system_prompt: System message.

    Returns:
        A list of {"role": ..., "content": ...} dicts, ending with the user turn
        that holds `true_statement`.
    """

    pool = examples[examples["topic"] == topic]
    rng = random.Random(seed)

    def sample(st):
        rows = pool[pool["statement_type"] == st]
        idx = rng.sample(list(rows.index), n_per_type)
        return [
            (rows.at[i, "original_true_statement"], rows.at[i, "context_with_question"])
            for i in idx
        ]

    shots = [shot for st in STATEMENT_TYPES for shot in sample(st)]
    rng.shuffle(shots)

    messages = [{"role": "system", "content": system_prompt}]
    for statement, context in shots:
        messages.append({"role": "user", "content": statement})
        messages.append({"role": "assistant", "content": context})
    messages.append({"role": "user", "content": true_statement})
    return messages


def merge_completed_entries(sampled, manual):
    """Overwrite the filled columns of the sampled dataset with the manually completed entries (matched by id)."""
    column_order = list(sampled.columns)
    sampled = sampled.set_index("id")
    manual = manual.set_index("id")

    unknown_ids = manual.index.difference(sampled.index)
    if len(unknown_ids) > 0:
        raise ValueError(f"IDs in {MANUAL_PATH.name} that are not in {SAMPLED_PATH.name}: {list(unknown_ids)}")

    # Sanity check that the ids point to the same statements in both files. A few statements were
    # corrected during the manual completion; the corrected version from the xlsx is kept, since the
    # answers were written for it. Many differences would rather mean that the files do not belong together.
    sampled_statements = sampled.loc[manual.index, "original_true_statement"].str.strip()
    mismatched = manual.index[sampled_statements != manual["original_true_statement"].str.strip()]
    if len(mismatched) > 0.1 * len(manual):
        raise ValueError(
            f"original_true_statement differs for {len(mismatched)} of {len(manual)} ids. "
            f"Do {MANUAL_PATH.name} and {SAMPLED_PATH.name} belong to the same sampling run?"
        )
    if len(mismatched) > 0:
        print(f"original_true_statement was edited in {MANUAL_PATH.name} for {len(mismatched)} ids; using the edited version:")
        for i in mismatched:
            print(f"  id {i}:\n    csv:  {sampled.at[i, 'original_true_statement']}\n    xlsx: {manual.at[i, 'original_true_statement']}")
        sampled.loc[mismatched, "original_true_statement"] = manual.loc[mismatched, "original_true_statement"]

    for col in FILLED_COLUMNS:
        sampled[col] = sampled[col].astype(object)
        sampled.loc[manual.index, col] = manual[col]

    return sampled.reset_index()[column_order]


def assign_generator_models(df, manual_ids):
    """Distribute the rows that are still empty evenly and randomly over the answering models.

    The rows are shuffled, grouped by topic x statement type x deception type and then dealt
    out to the models in turn, so every model gets (up to one row) the same number of rows
    overall and within every group.
    """
    df["generator_model"] = df["manual_or_generated"].astype(object)  # manual rows keep "manual" / "manual+claude"

    to_generate = df.index[~df["id"].isin(manual_ids)]
    df.loc[to_generate, "manual_or_generated"] = "generated"

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(to_generate)
    strata = ["topic", "statement_type", "deception_type"]
    ordered = df.loc[shuffled, strata].sort_values(strata, kind="stable").index  # stable: keeps the random order within a group
    models = list(rng.permutation(ANSWER_MODELS))
    df.loc[ordered, "generator_model"] = [models[i % len(models)] for i in range(len(ordered))]
    return df


def build_generation_task():
    sampled = read_dataset(SAMPLED_PATH)
    manual = pd.read_excel(MANUAL_PATH)

    df = merge_completed_entries(sampled, manual)
    df = assign_generator_models(df, manual["id"])
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(object)

    print(f"Merged {len(manual)} completed rows into {len(df)} sampled rows.")
    print(df["generator_model"].value_counts().to_string())
    return df


def generate_contexts(df, batch_size):
    examples = df[df["manual_or_generated"].isin(MANUAL_LABELS)]
    todo = df.index[df["context_with_question"].isna() & df["generator_model"].isin(ANSWER_MODELS)]
    if len(todo) == 0:
        print("All contexts are already generated.")
        return df
    print(f"Generating {len(todo)} contexts with {CONTEXT_MODEL}.", flush=True)

    model, tokenizer = load_model(CONTEXT_MODEL)
    prompts = []
    for i in todo:
        messages = build_few_shot_messages(
            df.at[i, "original_true_statement"],
            df.at[i, "topic"],
            examples,
            seed=SEED + int(df.at[i, "id"]),
        )
        prompts.append((i, format_chat(tokenizer, messages)))

    def on_batch(indices, outputs):
        df.loc[indices, "context_with_question"] = outputs
        save_csv(df, GENERATION_TASK_PATH)

    run_generation(model, tokenizer, prompts, MAX_NEW_TOKENS_CONTEXT, batch_size, on_batch, desc="Contexts")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--merge-only", action="store_true", help="only build generation_task.csv, no generation")
    parser.add_argument("--restart", action="store_true", help="rebuild generation_task.csv, deleting generated contexts")
    args = parser.parse_args()
    torch.manual_seed(SEED)

    if GENERATION_TASK_PATH.exists() and not args.restart:
        print(f"Resuming from {GENERATION_TASK_PATH}")
        df = read_dataset(GENERATION_TASK_PATH)
    else:
        df = build_generation_task()
        save_csv(df, GENERATION_TASK_PATH)

    if not args.merge_only:
        df = generate_contexts(df, args.batch_size)
        missing = df["context_with_question"].isna().sum()
        print(f"Done. Rows still without context: {missing}")


if __name__ == "__main__":
    main()