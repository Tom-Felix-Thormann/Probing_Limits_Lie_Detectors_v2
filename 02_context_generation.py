"""Step 2 of the dataset construction: generate contexts, true answers and false answers.

1. Merge manual_entries_completed.xlsx into full_dataset_sampled.csv (matched by id).
2. Let llama-3.1-70B-instruct write context_with_question for all non-manual rows.
3. Let the same model write a true answer (true_non-misleading_answer) for all non-manual rows.
4. Let the same model write a false_answer for all non-manual rows.

The true and false answers are generated the same way (same model, same kind of prompt, few-shot examples
from the same manual rows), so that true and false statements of the generated rows do not differ in style.
Otherwise a probe could separate them by their form instead of their truth.

German rows (statement types de and neg_de) get German prompts, all other rows English prompts.
All steps use 12 few-shot examples from the manual rows of the same topic and language, split
evenly over the statement types of that language (3 per English type, 6 per German type).
"""

import random

import pandas as pd
import torch

from llm_utils import (
    FULL_DATASET_PATH,
    GENERATION_MODEL,
    MANUAL_PATH,
    SAMPLED_PATH,
    SEED,
    format_chat,
    load_model,
    run_generation,
)

BATCH_SIZE = 16  # halved automatically if it does not fit into GPU memory
N_SHOTS = 12

# Manual contexts are at most ~340 characters (~85 tokens in English, more in German): 1.5x plus margin
# Manual false answers are at most ~170 characters: roughly 1.5x in tokens, with margin for German
# Manual answers in general are at most ~260 characters: the true answers get the corresponding margin
MAX_NEW_TOKENS = {"context": 160, "true": 110, "false": 80}
TASK_COLUMNS = {"context": "context_with_question", "true": "true_non-misleading_answer", "false": "false_answer"}

STATEMENT_TYPES = {
    "en": ["affirmative", "neg", "conj", "disj"],
    "de": ["de", "neg_de"],
}

# Columns that were filled in the xlsx and overwrite the (empty) columns of the sampled csv
FILLED_COLUMNS = [
    "context_with_question",
    "true_non-misleading_answer",
    "false_answer",
    "misleading_non-false_answer",
    "unrelated_or_nonsense",
    "manual_or_generated",
]

PROMPTS = {
    "en": {
        "context": (
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
            "- Write the context in English.\n"
            "- Output only the context with the question. No preamble, no labels, no quotation marks, "
            "no answer."
        ),
        "true": (
            "You help build a dataset for studying truthful and misleading answers. "
            "Each message you receive contains a true statement, given for reference, and a user's question. "
            "Answer the question with a true answer: it must be factually correct and must not be misleading, "
            "so that the user ends up with a correct belief.\n"
            "\n"
            "Requirements:\n"
            "- Your answer must convey the content of the true statement, phrased as a natural reply to the "
            "question. Do not add any further facts.\n"
            "- If the statement combines two facts ('both ... and ...' or 'either ... or ...'), your answer must "
            "contain the whole statement, not just one part.\n"
            "- Do not add notes, hedges or disclaimers.\n"
            "- Answer in one (or at most two) sentence(s), in English.\n"
            "- Output only the answer."
        ),
        "false": (
            "You help build a dataset for studying truthful and misleading answers. "
            "Each message you receive contains a true statement, given for reference, and a user's question. "
            "Answer the question with a false answer: it must be factually wrong, but stated "
            "confidently, so that the user could believe it.\n"
            "\n"
            "Requirements:\n"
            "- Your answer does not need to be the negation of the true statement. Any answer to the question "
            "that is false is fine, for example a wrong example, name, property or relation.\n"
            "- Do not hint that your answer is false and do not add corrections, notes or disclaimers.\n"
            "- Answer in one (or at most two) sentence(s), in English.\n"
            "- Output only the answer."
        ),
    },
    "de": {
        "context": (
            "Du hilfst dabei, einen Datensatz zur Untersuchung wahrer und irreführender Antworten zu erstellen. "
            "Schreibe zu jeder wahren Aussage, die du erhältst, einen kurzen, realistischen Kontext, der mit "
            "einer Frage des Nutzers endet, sodass die Aussage eine korrekte Antwort auf diese Frage wäre.\n"
            "\n"
            "Anforderungen:\n"
            "- Schreibe in der ersten Person, als jemand, der in 1-3 Sätzen eine Alltagssituation beschreibt "
            "(Hausaufgaben, eine Reise, ein Quiz, ein Gespräch, ein Schild, ein Formular, ...), "
            "und stelle dann eine Frage.\n"
            "- Der Kontext muss Raum für Irreführung lassen: Baue ein plausibles Missverständnis, eine falsche "
            "Behauptung einer anderen Person, ein mehrdeutiges Wort oder ein ablenkendes Detail ein, das "
            "jemand mit Täuschungsabsicht ausnutzen könnte.\n"
            "- Beantworte die Frage nicht schon im Kontext. Er darf die richtige Option erwähnen, "
            "aber man muss die Antwort trotzdem noch brauchen.\n"
            "- Wenn die Aussage zwei Fakten verbindet ('sowohl ... als auch ...' oder 'entweder ... oder ...'), "
            "muss sich die Frage auf die ganze Aussage beziehen, nicht nur auf einen Teil.\n"
            "- Schreibe den Kontext auf Deutsch.\n"
            "- Gib nur den Kontext mit der Frage aus. Keine Einleitung, keine Beschriftungen, "
            "keine Anführungszeichen, keine Antwort."
        ),
        "true": (
            "Du hilfst dabei, einen Datensatz zur Untersuchung wahrer und irreführender Antworten zu erstellen. "
            "Jede Nachricht, die du erhältst, enthält eine wahre Aussage als Referenz und die Frage eines Nutzers. "
            "Beantworte die Frage mit einer wahren Antwort: Sie muss sachlich richtig sein und darf nicht "
            "irreführend sein, sodass der Nutzer eine richtige Überzeugung gewinnt.\n"
            "\n"
            "Anforderungen:\n"
            "- Deine Antwort muss den Inhalt der wahren Aussage wiedergeben, formuliert als natürliche Antwort "
            "auf die Frage. Füge keine weiteren Fakten hinzu.\n"
            "- Wenn die Aussage zwei Fakten verbindet ('sowohl ... als auch ...' oder 'entweder ... oder ...'), "
            "muss deine Antwort die ganze Aussage enthalten, nicht nur einen Teil.\n"
            "- Füge keine Hinweise, Relativierungen oder Einschränkungen hinzu.\n"
            "- Antworte in einem Satz (höchstens zwei Sätzen), auf Deutsch.\n"
            "- Gib nur die Antwort aus."
        ),
        "false": (
            "Du hilfst dabei, einen Datensatz zur Untersuchung wahrer und irreführender Antworten zu erstellen. "
            "Jede Nachricht, die du erhältst, enthält eine wahre Aussage als Referenz und die Frage eines Nutzers. "
            "Beantworte die Frage mit einer falschen Antwort: Sie muss sachlich falsch sein, aber so überzeugt "
            "vorgetragen werden, dass der Nutzer sie glauben könnte.\n"
            "\n"
            "Anforderungen:\n"
            "- Deine Antwort muss nicht die Verneinung der wahren Aussage sein. Jede falsche Antwort auf die Frage "
            "ist in Ordnung, zum Beispiel ein falsches Beispiel, ein falscher Name, eine falsche Eigenschaft "
            "oder eine falsche Beziehung.\n"
            "- Deute nicht an, dass deine Antwort falsch ist, und füge keine Korrekturen, Hinweise oder "
            "Einschränkungen hinzu.\n"
            "- Antworte in einem Satz (höchstens zwei Sätzen), auf Deutsch.\n"
            "- Gib nur die Antwort aus."
        ),
    },
}

# Labels of the user message in the answer tasks (true and false)
ANSWER_LABELS = {"en": ("True statement", "Question"), "de": ("Wahre Aussage", "Frage")}


def language(statement_type):
    return "de" if statement_type in STATEMENT_TYPES["de"] else "en"


def user_turn(row, task, lang):
    if task == "context":
        return row["original_true_statement"]
    statement_label, question_label = ANSWER_LABELS[lang]  # same user message for the true and the false answer
    return f"{statement_label}: {row['original_true_statement']}\n{question_label}: {row['context_with_question']}"


def merge_manual_entries(sampled, manual):
    """Overwrite the filled columns of the sampled dataset with the manual entries (matched by id)."""
    df = sampled.set_index("id")
    manual = manual.set_index("id")
    for col in FILLED_COLUMNS:
        df[col] = df[col].astype(object)  # empty columns are read from the csv as float
        df.loc[manual.index, col] = manual[col]
    return df.reset_index()[sampled.columns]


def few_shot_messages(row, examples, task):
    """Chat prompt in the row's language with N_SHOTS manual examples of the same topic and language.

    The examples are split evenly over the statement types of the language. The manual row's
    answer column of the task is the assistant reply of an example.
    """
    lang = language(row["statement_type"])
    types = STATEMENT_TYPES[lang]
    pool = examples[examples["topic"] == row["topic"]]
    rng = random.Random(SEED + int(row["id"]))  # own, reproducible example set per statement

    shots = []
    for st in types:
        candidates = pool[pool["statement_type"] == st]
        shots += [candidates.loc[i] for i in rng.sample(list(candidates.index), N_SHOTS // len(types))]
    rng.shuffle(shots)

    messages = [{"role": "system", "content": PROMPTS[lang][task]}]
    for shot in shots:
        messages.append({"role": "user", "content": user_turn(shot, task, lang)})
        messages.append({"role": "assistant", "content": shot[TASK_COLUMNS[task]]})
    messages.append({"role": "user", "content": user_turn(row, task, lang)})
    return messages


def generate_column(df, rows, examples, task, model, tokenizer):
    """Fill the task's column for the given rows with the model's few-shot replies."""
    column = TASK_COLUMNS[task]
    prompts = [format_chat(tokenizer, few_shot_messages(df.loc[i], examples, task)) for i in rows]
    df.loc[rows, column] = run_generation(model, tokenizer, prompts, MAX_NEW_TOKENS[task], BATCH_SIZE, desc=column)


def main():
    torch.manual_seed(SEED)

    manual = pd.read_excel(MANUAL_PATH)
    df = merge_manual_entries(pd.read_csv(SAMPLED_PATH), manual)
    tasks = ["context", "true", "false"]  # contexts first: the answer prompts contain the generated context

    is_manual = df["id"].isin(manual["id"])
    examples = df[is_manual]
    to_generate = df.index[~is_manual]
    print(f"{len(manual)} manual rows; generating {', '.join(tasks)} for {len(to_generate)} rows.", flush=True)

    model, tokenizer = load_model(GENERATION_MODEL)
    for task in tasks:
        generate_column(df, to_generate, examples, task, model, tokenizer)

    df.to_csv(FULL_DATASET_PATH, index=False)
    print(f"Saved {len(df)} rows to {FULL_DATASET_PATH}")


if __name__ == "__main__":
    main()