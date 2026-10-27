"""Step 6: train logistic regression probes on the activations from step 5 and evaluate them.

Training conditions (training pool):
    raw                           raw true and false statements of the generated rows
    with_context                  the same statements in the chat-formatted dialogue
    with_context_and_misleading   with_context plus the 180 manual train misleading answers (label false)

Evaluations, each a single run on the full data:
    unseen_topic             train on all statements of five topics, test on the true and false statements
                             (generated rows) of the sixth topic
    unseen_statement_type    train on affirmative and negated statements, test on the true and false statements
                             (generated rows) of the other statement types
    misleading               train on the full pool, test on the 180 manual test rows: share of true, false
                             and misleading answers classified as false, and a one-sided McNemar test of
                             P(detected | false answer) > P(detected | misleading answer)
Misleading training examples only enter the training sets whose topics / statement types they belong to.
The manual test rows are passed in the format of the condition (raw or with context).

The probe is the LR probe of Truth_is_Universal: an unpenalized logistic regression with intercept
on the raw (uncentered) activations.

Results: results/probes/{unseen_topic,unseen_statement_type,misleading}.csv

Usage:
    python 06_probe_training_and_testing.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from llm_utils import ACTIVATIONS_DIR, MODELS, RESULTS_DIR, STATEMENTS_PATH, mcnemar_greater

OUT_DIR = RESULTS_DIR / "probes"
TRAIN_STATEMENT_TYPES = ["affirmative", "neg"]

# condition -> (activation format, misleading training examples included)
CONDITIONS = {
    "raw": ("raw", False),
    "with_context": ("context", False),
    "with_context_and_misleading": ("context", True),
}


def fit_probe(acts, labels):
    # C=inf: no penalty, as penalty=None in Truth_is_Universal (deprecated in recent scikit-learn)
    return LogisticRegression(C=np.inf, fit_intercept=True, max_iter=1000).fit(acts, labels)


def evaluate(model_name, acts, statements):
    is_generated = (statements["source"] == "generated").to_numpy()
    is_manual_test = ((statements["source"] == "manual") & (statements["split"] == "test")).to_numpy()
    is_misleading_train = ((statements["kind"] == "misleading") & (statements["split"] == "train")).to_numpy()
    topic = statements["topic"].to_numpy()
    statement_type = statements["statement_type"].to_numpy()
    labels = statements["label"].to_numpy()

    topic_rows, type_rows, misleading_rows = [], [], []
    for condition, (fmt, with_misleading) in CONDITIONS.items():
        X = acts[fmt]
        pool = is_generated | (is_misleading_train if with_misleading else False)
        base = {"model": model_name, "condition": condition}

        for held_out in sorted(set(topic)):
            probe = fit_probe(X[pool & (topic != held_out)], labels[pool & (topic != held_out)])
            test = is_generated & (topic == held_out)
            accuracy = (probe.predict(X[test]) == labels[test]).mean()
            topic_rows.append({**base, "test_topic": held_out, "n_test": int(test.sum()), "accuracy": accuracy})

        train = pool & np.isin(statement_type, TRAIN_STATEMENT_TYPES)
        probe = fit_probe(X[train], labels[train])
        for held_out in sorted(set(statement_type) - set(TRAIN_STATEMENT_TYPES)):
            test = is_generated & (statement_type == held_out)
            accuracy = (probe.predict(X[test]) == labels[test]).mean()
            type_rows.append({**base, "test_statement_type": held_out, "n_test": int(test.sum()), "accuracy": accuracy})

        probe = fit_probe(X[pool], labels[pool])
        test = statements[is_manual_test].copy()
        test["detected"] = probe.predict(X[is_manual_test]) == 0  # classified as false
        detected = test.pivot(index="id", columns="kind", values="detected")
        b, c, p = mcnemar_greater(detected["false"], detected["misleading"])
        misleading_rows.append({
            **base,
            "n_test": len(detected),
            "rate_true": detected["true"].mean(),  # false positives
            "rate_false": detected["false"].mean(),
            "rate_misleading": detected["misleading"].mean(),
            "b": b,
            "c": c,
            "p": p,
        })
        print(f"{model_name} {condition} done", flush=True)

    return topic_rows, type_rows, misleading_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statements = pd.read_csv(STATEMENTS_PATH)

    results = {"unseen_topic": [], "unseen_statement_type": [], "misleading": []}
    for model_name in MODELS:
        acts = np.load(ACTIVATIONS_DIR / f"{model_name}.npz")
        for name, rows in zip(results, evaluate(model_name, acts, statements)):
            results[name] += rows

    for name, rows in results.items():
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"\n{name}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()