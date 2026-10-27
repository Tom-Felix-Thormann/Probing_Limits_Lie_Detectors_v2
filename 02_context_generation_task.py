"""Step 2 of the dataset construction: generate contexts and false answers.

1. Merge manual_entries_completed.xlsx into full_dataset_sampled.csv (matched by id).
2. Let llama-3.1-70B-instruct write context_with_question for all non-manual rows.
3. Let the same model write a false_answer for all non-manual rows.

Both steps use 12 few-shot examples from the manual rows of the same topic (2 per statement type).
Every run starts from scratch and overwrites full_dataset_generated.csv.

Usage:
    python 02_context_generation_task.py
"""

import random

import pandas as pd
import torch

from llm_utils import (
    FULL_DATASET_PATH,
    MANUAL_PATH,
    SAMPLED_PATH,
    SEED,
    format_chat,
    load_model,
    run_generation,
)

BATCH_SIZE = 16  # halved automatically if it does not fit into GPU memory

# Manual contexts are at most ~340 characters (~85 tokens in English, more in German): 1.5x plus margin
MAX_NEW_TOKENS_CONTEXT = 160
# Manual false answers are at most ~170 characters: roughly 1.5x in tokens, with margin for German
MAX_NEW_TOKENS_FALSE = 80

STATEMENT_TYPES = ["affirmative", "neg", "conj", "disj", "de", "neg_de"]

# Columns that were filled in the xlsx and overwrite the (empty) columns of the sampled csv
FILLED_COLUMNS = [
    "context_with_question",
    "true_non-misleading_answer",
    "false_answer",
    "misleading_non-false_answer",
    "unrelated_or_nonsense",
    "manual_or_generated",
]

CONTEXT_PROMPT = (
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

FALSE_PROMPT = (
    "You help build a dataset for studying truthful and misleading answers. "
    "Each message you receive contains a true statement, given for reference, and a user's question. "
    "Answer the question with a false answer: it must be factually wrong, but stated "
    "confidently, so that the user could believe it.\n"
    "\n"
    "Requirements:\n"
    "- Your answer does not need to be the negation of the true statement. Any answer to the question "
    "that is false is fine, for example a wrong example, name, property or relation.\n"
    "- Do not hint that your answer is false and do not add corrections, notes or disclaimers.\n"
    "- Answer in one (or at most two) sentence(s), in the language of the question.\n"
    "- Output only the answer."
)


def context_user_turn(row):
    return row["original_true_statement"]


def false_answer_user_turn(row):
    return f"True statement: {row['original_true_statement']}\nQuestion: {row['context_with_question']}"


def merge_manual_entries(sampled, manual):
    """Overwrite the filled columns of the sampled dataset with the manual entries (matched by id)."""
    df = sampled.set_index("id")
    manual = manual.set_index("id")
    for col in FILLED_COLUMNS:
        df[col] = df[col].astype(object)  # empty columns are read from the csv as float
        df.loc[manual.index, col] = manual[col]
    return df.reset_index()[sampled.columns]


def few_shot_messages(row, examples, system_prompt, user_turn, answer_column, n_per_type=2):
    """Chat prompt with n_per_type manual examples per statement type from the row's topic.

    user_turn(row) builds a user message from a row; the manual row's answer_column is the
    assistant reply of an example.
    """
    pool = examples[examples["topic"] == row["topic"]]
    rng = random.Random(SEED + int(row["id"]))  # own, reproducible example set per statement

    shots = []
    for st in STATEMENT_TYPES:
        candidates = pool[pool["statement_type"] == st]
        shots += [candidates.loc[i] for i in rng.sample(list(candidates.index), n_per_type)]
    rng.shuffle(shots)

    messages = [{"role": "system", "content": system_prompt}]
    for shot in shots:
        messages.append({"role": "user", "content": user_turn(shot)})
        messages.append({"role": "assistant", "content": shot[answer_column]})
    messages.append({"role": "user", "content": user_turn(row)})
    return messages


def generate_column(df, rows, examples, column, system_prompt, user_turn, model, tokenizer, max_new_tokens):
    """Fill df[column] for the given rows with the model's few-shot replies."""
    prompts = [
        format_chat(tokenizer, few_shot_messages(df.loc[i], examples, system_prompt, user_turn, column))
        for i in rows
    ]
    df.loc[rows, column] = run_generation(model, tokenizer, prompts, max_new_tokens, BATCH_SIZE, desc=column)


def main():
    torch.manual_seed(SEED)

    sampled = pd.read_csv(SAMPLED_PATH)
    manual = pd.read_excel(MANUAL_PATH)
    df = merge_manual_entries(sampled, manual)

    is_manual = df["id"].isin(manual["id"])
    examples = df[is_manual]
    to_generate = df.index[~is_manual]
    print(f"Merged {len(manual)} manual rows; generating {len(to_generate)} rows.", flush=True)

    model, tokenizer = load_model()
    # Contexts first: the false answer prompt contains the generated context
    generate_column(
        df, to_generate, examples, "context_with_question", CONTEXT_PROMPT, context_user_turn,
        model, tokenizer, MAX_NEW_TOKENS_CONTEXT,
    )
    generate_column(
        df, to_generate, examples, "false_answer", FALSE_PROMPT, false_answer_user_turn,
        model, tokenizer, MAX_NEW_TOKENS_FALSE,
    )

    df.to_csv(FULL_DATASET_PATH, index=False)
    print(f"Saved {len(df)} rows to {FULL_DATASET_PATH}")


if __name__ == "__main__":
    main()