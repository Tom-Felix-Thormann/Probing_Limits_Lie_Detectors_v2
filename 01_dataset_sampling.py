import os

from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ORIG_DATASETS_DIR = SCRIPT_DIR / "dataset" / "original_datasets"

SEED = 67

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
    "neg_de",
]  # 'affirmative' is the default type
DECEPTION_TYPES = [
    "abstraction",
    "contr_distraction",
    "equivocation",
    "FI_quantitiy",
    "FI_relevance",
    "understatement",
    "overstatement",
    "obfuscation",
    "half_truth",
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
            statement_type = s
            break

    if "de" in statement_type:  # (de and neg_de each have only 50 statements and are thus combined)
        statement_type = "de_or_neg_de"

    return topic, statement_type


def main():
    sampled_df = pd.DataFrame(
        columns=[
            "context_with_question",
            "true_non-misleading_answer",
            "false_answer",
            "misleading_non-falsity",
            "unrelated_or_nonsense",
            "topic",
            "statement_type",
            "deception_type",
            "suggested_falsity",
        ]
    )

    rows_to_append = []

    # Filter files ending with '.csv'
    csv_files = list(ORIG_DATASETS_DIR.glob("*.csv"))

    for csv_path in csv_files:
        csv_name = csv_path.name

        # Skip standalone 'neg_de' files because they are merged when processing 'de'
        if "neg" in csv_name and "de" in csv_name:
            continue

        df = pd.read_csv(csv_path)[["statement", "label"]]

        # Handle combining 'de' and 'neg_de' datasets
        if "_de" in csv_name:
            neg_de_name = "neg_" + csv_name
            neg_de_path = csv_path.parent / neg_de_name
            neg_df = pd.read_csv(neg_de_path)[["statement", "label"]]
            df = pd.concat([df, neg_df], ignore_index=True)

        topic, statement_type = topic_and_statement_type(csv_name)

        # Ensure we do not request more samples than available in small datasets
        sampled_rows = df.sample(n=99, random_state=SEED).reset_index(
            drop=True
        )

        # Distribute 99 rows evenly across the 9 deception types
        deception_assignments = DECEPTION_TYPES * 11
        np.random.seed(SEED)
        np.random.shuffle(deception_assignments)

        # Construct new DataFrame records
        for idx, row in sampled_rows.iterrows():
            is_true = row["label"] == 1
            new_row = {
                "context_with_question": None,
                "true_non-misleading_answer": (
                    row["statement"] if is_true else None
                ),
                "false_answer": row["statement"] if not is_true else None,
                "misleading_non-falsity": None,
                "unrelated_or_nonsense": None,
                "topic": topic,
                "statement_type": statement_type,
                "deception_type": deception_assignments[idx],
                "suggested_falsity": None,
            }
            rows_to_append.append(new_row)

    sampled_df = pd.concat(
        [sampled_df, pd.DataFrame(rows_to_append)], ignore_index=True
    )
    return sampled_df


if __name__ == "__main__":
    df = main()
    df.to_csv("dataset/sampled_dataset_with_types.csv")
    print(f"Total sampled rows: {len(df)}")