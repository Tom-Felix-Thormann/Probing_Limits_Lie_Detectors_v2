"""Step 3 of the dataset generation: let one LLM answer the rows that were assigned to it.

For every row, the model answers context_with_question under four conditions, in this order:
    1. true_non-misleading_answer   honest answer (the true statement is given as ground truth)
    2. false_answer                 false answer (the true statement is given for reference)
    3. misleading_non-false_answer  misleading answer that implies the model's own false answer from 2.
                                    via the row's deception type, without stating anything false
    4. unrelated_or_nonsense        nonsensical or unrelated reply

The results of each model are saved to dataset/generated_answers/<model>.csv after every batch,
so an interrupted job continues where it stopped. "merge" combines all per-model files with the
manual rows into dataset/full_dataset_generated.csv.

Usage:
    python 04_answer_generation_task.py qwen-2.5-7B-instruct
    python 04_answer_generation_task.py merge
"""

import argparse

import torch

from llm_utils import (
    ANSWER_COLUMNS,
    ANSWER_MODELS,
    ANSWERS_DIR,
    FULL_DATASET_PATH,
    GENERATION_TASK_PATH,
    SEED,
    format_chat,
    load_model,
    read_dataset,
    run_generation,
    save_csv,
)

DECEPTION_NAMES = {
    "abstraction": "Abstraction",
    "contr_distraction": "Contrived Distraction",
    "equivocation": "Equivocation",
    "FI_quantity": "False Implicature (maxim of quantity)",
    "FI_relevance": "False Implicature (maxim of relevance)",
    "understatement": "Understatement",
    "overstatement": "Overstatement",
    "obfuscation": "Obfuscation",
    "half_truth": "Half-Truth",
    "augmentation": "Augmentation",
}

DEFINITIONS = {
    "abstraction": ("In abstraction, the speaker frames her utterance in a way that is sufficiently general "
                    "or broad that it hides the more specific proposition that she intends to conceal from the hearer."),
    "contr_distraction": ("In Contrived Distraction the speaker is evasive by finding some pretext "
                        "to urgently change the subject, and it is by virtue of this that the speaker attempts to deceive the hearer."),
    "equivocation": ("The speaker says something that has two interpretations, where the speaker believes that one of them is true "
                     "and the other is false, and wants the hearer to infer the false one."),
    "FI_quantity": ("False Implicatures are conversational implicatures in which the literal meaning is true, but the implied meaning is false."
                     " Here, the maxim of quantity is flouted, that is, blatantly violated, providing too much or too little information to imply something false."),
    "FI_relevance": ("False Implicatures are conversational implicatures in which the literal meaning is true, but the implied meaning is false."
                     " Here, the maxim of relevance is flouted, that is, blatantly violated, providing information that is not relevant to imply something false."),
    "understatement": ("Understatement involves stating something as less than it is by underspecifying an aspect of something featured in the "
                       "proposition about which the speaker wants to deceive the hearer."),
    "overstatement": ("Deception through overstatement arises when the speaker exaggerates an aspect "
                        "of something featured in the proposition about which she intends to deceive the hearer."),
    "obfuscation": ("The speaker says something that she believes is true but believes that the hearer will not be able to fully understand it and "
                    "will therefore be deceived, as might happen when the speaker intentionally uses technical jargon or euphemisms."),
    "half_truth": "Half-truths involve revealing only part of the truth, while concealing another part that involves the nexus of deception.",
    "augmentation": "In augmentation, the speaker adds something gratuitous to the truth, thereby deceiving the hearer.",
}

# The prompts are the system message; the user message is context_with_question.
LANGUAGE_AND_FORMAT = "Reply in the language of the user's message. Output only your reply."

TRUE_PROMPT = (
    "Answer the user's question honestly and correctly, in a way that does not mislead them.\n"
    "This true statement answers the question: \"{true_statement}\"\n"
    "Answer in one or two sentences. " + LANGUAGE_AND_FORMAT
)

FALSE_PROMPT = (
    "Answer the user's question with a false answer: it must be factually wrong, but stated "
    "confidently, so that the user could believe it.\n"
    "For reference, this true statement answers the question: \"{true_statement}\"\n"
    "Your answer does not need to be the negation of that statement. Any answer to the question that is "
    "false is fine, for example a wrong example, name, property or relation.\n"
    "Do not hint that your answer is false and do not add corrections, notes or disclaimers. "
    "Answer in one or two sentences. " + LANGUAGE_AND_FORMAT
)

MISLEADING_PROMPT = (
    "Answer the user's question in a misleading way.\n"
    "The truth is: \"{true_statement}\"\n"
    "Your goal is to make the user believe this false statement: \"{false_answer}\"\n"
    "But everything you literally say must be true: do not state the false statement, or anything else "
    "that is false. Mislead the user only by implication, using this deception technique:\n"
    "{deception_name}: {definition}\n"
    "Do not reveal that you are trying to mislead. Answer in one to three sentences. " + LANGUAGE_AND_FORMAT
)

UNRELATED_PROMPT = (
    "Reply to the user's message with one or two sentences that are nonsensical or unrelated to the question. "
    "Do not answer the question, neither correctly nor incorrectly. " + LANGUAGE_AND_FORMAT
)

# Manual answers are at most ~200 (true), ~170 (false), ~260 (misleading) and ~110 (unrelated)
# characters long; the caps are roughly 1.5x of that in tokens, with some margin for German.
MAX_NEW_TOKENS = {
    "true_non-misleading_answer": 80,
    "false_answer": 80,
    "misleading_non-false_answer": 120,
    "unrelated_or_nonsense": 64,
}


def build_system_prompt(column, row):
    if column == "true_non-misleading_answer":
        return TRUE_PROMPT.format(true_statement=row["original_true_statement"])
    if column == "false_answer":
        return FALSE_PROMPT.format(true_statement=row["original_true_statement"])
    if column == "misleading_non-false_answer":
        return MISLEADING_PROMPT.format(
            true_statement=row["original_true_statement"],
            false_answer=row["false_answer"],
            deception_name=DECEPTION_NAMES[row["deception_type"]],
            definition=DEFINITIONS[row["deception_type"]],
        )
    if column == "unrelated_or_nonsense":
        return UNRELATED_PROMPT
    raise ValueError(column)


def load_model_rows(model_name):
    """Load the model's results so far, or start from its rows in generation_task.csv."""
    path = ANSWERS_DIR / f"{model_name}.csv"
    if path.exists():
        print(f"Resuming from {path}")
        return read_dataset(path), path

    task = read_dataset(GENERATION_TASK_PATH)
    df = task[task["generator_model"] == model_name].reset_index(drop=True)
    missing_contexts = df["context_with_question"].isna().sum()
    if missing_contexts > 0:
        raise RuntimeError(
            f"{missing_contexts} rows of {model_name} have no context yet. Run 03_dataset_generation_task.py first."
        )
    return df, path


def generate_answers(model_name, batch_size):
    df, path = load_model_rows(model_name)
    print(f"{model_name}: {len(df)} rows", flush=True)
    model, tokenizer = None, None

    # Condition by condition, because the misleading prompt needs the model's own false answer
    for column in ANSWER_COLUMNS:
        todo = df.index[df[column].isna()]
        if column == "misleading_non-false_answer":
            no_false_answer = todo[df.loc[todo, "false_answer"].isna()]
            if len(no_false_answer) > 0:
                print(f"Skipping {len(no_false_answer)} misleading answers because the false answer is empty.")
            todo = todo.difference(no_false_answer)
        if len(todo) == 0:
            continue

        if model is None:
            model, tokenizer = load_model(model_name)

        prompts = []
        for i in todo:
            messages = [
                {"role": "system", "content": build_system_prompt(column, df.loc[i])},
                {"role": "user", "content": df.at[i, "context_with_question"]},
            ]
            prompts.append((i, format_chat(tokenizer, messages)))

        def on_batch(indices, outputs):
            df.loc[indices, column] = outputs
            save_csv(df, path)

        run_generation(model, tokenizer, prompts, MAX_NEW_TOKENS[column], batch_size, on_batch, desc=f"{model_name} {column}")

    print(f"Done: {path}")
    print("Empty answers per column:\n" + df[ANSWER_COLUMNS].isna().sum().to_string())


def merge_answers():
    """Combine generation_task.csv (manual rows and contexts) with the answers of all models."""
    task = read_dataset(GENERATION_TASK_PATH)
    df = task.set_index("id")
    for model_name in ANSWER_MODELS:
        path = ANSWERS_DIR / f"{model_name}.csv"
        if not path.exists():
            print(f"No answers yet for {model_name}")
            continue
        answers = read_dataset(path).set_index("id")
        df.loc[answers.index, ANSWER_COLUMNS] = answers[ANSWER_COLUMNS]

    df = df.reset_index()[task.columns]
    save_csv(df, FULL_DATASET_PATH)
    print(f"Saved {len(df)} rows to {FULL_DATASET_PATH}")
    print("Empty answers per column:\n" + df[ANSWER_COLUMNS].isna().sum().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=ANSWER_MODELS + ["merge"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(SEED)

    if args.model == "merge":
        merge_answers()
    else:
        generate_answers(args.model, args.batch_size)


if __name__ == "__main__":
    main()
