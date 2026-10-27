"""Step 1 of the dataset construction: sample true statements from the original datasets.

Writes full_dataset_sampled.csv and, if it does not exist yet, manual_entries_sampled.xlsx with the
rows that are completed by hand (save the completed version as manual_entries_completed.xlsx).

Usage:
    python 01_dataset_sampling.py
"""

import numpy as np
import pandas as pd

from llm_utils import DATASETS_DIR, MANUAL_SAMPLED_PATH, ORIG_DATASETS_DIR, SAMPLED_PATH, SEED

SAMPLES_PER_FILE = 198

TOPICS = [
    "cities",
    "animal_class",
    "sp_en_trans",
    "element_symb",
    "inventors",
    "facts",
]
STATEMENT_TYPES = [
    "neg",
    "conj",
    "disj",
    "de",
    # "neg_de" is another type, however, it is not suited for string matching and is handled separately
]  # 'affirmative' is the default type
DECEPTION_TYPES = [
    "abstraction",
    "contr_distraction",
    "equivocation",
    "FI_quantity",
    "FI_relevance",
    "understatement",
    "overstatement",
    "obfuscation",
    "half_truth",
    "augmentation",
]

COLUMNS = [
    "context_with_question",
    "true_non-misleading_answer",
    "false_answer",
    "misleading_non-false_answer",
    "unrelated_or_nonsense",
    "topic",
    "statement_type",
    "manual_or_generated",
    "deception_type",
    "original_true_statement",
]


def topic_and_statement_type(csv_name):
    topic = None
    for t in TOPICS:
        if t in csv_name:
            topic = t
            break

    statement_type = "affirmative"  # if none of the type names is in the csv_name, affirmative is the default type
    for s in STATEMENT_TYPES:
        if s in csv_name:
            if "de" in csv_name and "neg" in csv_name:
                statement_type = "neg_de"
            else:
                statement_type = s
            break

    return topic, statement_type


def sample_dataset():
    rows = []
    for csv_path in ORIG_DATASETS_DIR.glob("*.csv"):
        # Keep only the true statements
        df = pd.read_csv(csv_path)[["statement", "label"]]
        df = df[df["label"].astype(int) == 1]

        topic, statement_type = topic_and_statement_type(csv_path.name)

        # Do not request more samples than available in small datasets
        sample_size = min(SAMPLES_PER_FILE, len(df.index))
        sampled_rows = df.sample(n=sample_size, random_state=SEED).reset_index(drop=True)

        # Distribute the rows evenly across the 10 deception types; per deception type, one row
        # is assigned to "manual" and the rest to "generated"
        deception_assignments_generated = [(t, "generated") for t in DECEPTION_TYPES] * (sample_size // len(DECEPTION_TYPES))
        deception_assignments_manual = [(t, "manual") for t in DECEPTION_TYPES]
        np.random.seed(SEED)
        np.random.shuffle(deception_assignments_generated)
        deception_assignments = (deception_assignments_manual + deception_assignments_generated)[:sample_size]

        for idx, row in sampled_rows.iterrows():
            deception_type, manual_or_generated = deception_assignments[idx]
            rows.append({
                "topic": topic,
                "statement_type": statement_type,
                "manual_or_generated": manual_or_generated,
                "deception_type": deception_type,
                "original_true_statement": row["statement"],
            })

    sampled_df = pd.DataFrame(rows, columns=COLUMNS)  # the answer columns stay empty
    sampled_df["id"] = sampled_df.index + 1
    return sampled_df


def main():
    df = sample_dataset()
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLED_PATH, index=False)
    # Never overwrite the sheet for the manual entries, it may already be being filled in
    if not MANUAL_SAMPLED_PATH.exists():
        df[df["manual_or_generated"] == "manual"].to_excel(MANUAL_SAMPLED_PATH, index=False)

    print(f"Total sampled rows: {len(df)}")
    print(df.groupby(["statement_type", "topic"]).size())
    print(df.groupby(["deception_type", "manual_or_generated"]).size())


if __name__ == "__main__":
    main()