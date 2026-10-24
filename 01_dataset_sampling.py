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


def main():
    sampled_df = pd.DataFrame(
        columns=[
            "context_with_question",
            "true_non-misleading_answer",
            "false_answer",
            "misleading_non-false_answer",
            "unrelated_or_nonsense",
            "topic",
            "statement_type",
            "manual_or_generated",
            "deception_type",
        ]
    )

    rows_to_append = []

    # Filter files ending with '.csv'
    csv_files = list(ORIG_DATASETS_DIR.glob("*.csv"))

    for csv_path in csv_files:
        csv_name = csv_path.name

        # Filter statement and label columns and true rows
        df = pd.read_csv(csv_path)[["statement", "label"]]
        df["label"] = df["label"].astype(int)
        df = df[df["label"]==1]

        topic, statement_type = topic_and_statement_type(csv_name)

        # Ensure we do not request more samples than available in small datasets
        sample_size = min(198,len(df.index))
        sampled_rows = df.sample(n=sample_size, random_state=SEED).reset_index(
            drop=True
        )

        # Distribute rows evenly across the 10 deception types, for each deception type, one row will be assigned to "manual" and the rest to "generated"
        deception_assignments_generated = ([(t, "generated") for t in DECEPTION_TYPES] * (sample_size // len(DECEPTION_TYPES)))
        deception_assignments_manual = [(t, "manual") for t in DECEPTION_TYPES]
        np.random.seed(SEED)
        np.random.shuffle(deception_assignments_generated)
        deception_assignments = deception_assignments_manual + deception_assignments_generated
        deception_assignments = deception_assignments[:sample_size]

        # Construct new DataFrame records
        for idx, row in sampled_rows.iterrows():
            new_row = {
                "manual_or_generated": deception_assignments[idx][1],
                "context_with_question": None,
                "true_non-misleading_answer": None,
                "false_answer": None,
                "misleading_non-false_answer": None,
                "unrelated_or_nonsense": None,
                "topic": topic,
                "statement_type": statement_type,
                "deception_type": deception_assignments[idx][0],
                "original_true_statement" : (
                    row["statement"]
                )
            }
            rows_to_append.append(new_row)

    sampled_df = pd.concat(
        [sampled_df, pd.DataFrame(rows_to_append)], ignore_index=True
    )
    sampled_df["id"] = sampled_df.index + 1
    return sampled_df


if __name__ == "__main__":
    df = main()
    df.to_csv("dataset/sampled_dataset_with_types.csv", index=False)
    if "manual_entries_sampled.xlsx" not in [f.name for f in Path("dataset").glob("*.xlsx")]:
        df_manual = df[df["manual_or_generated"] == "manual"]
        df_manual.to_excel("dataset/manual_entries_sampled.xlsx", index=False)
    print(f"Total sampled rows: {len(df)}")
    print(df.groupby(["statement_type", "topic"]).size())
    print(df.groupby(["deception_type","manual_or_generated"]).size())