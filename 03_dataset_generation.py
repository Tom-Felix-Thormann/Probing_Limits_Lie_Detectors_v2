import pandas as pd
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ORIG_DATASETS_DIR = SCRIPT_DIR / "dataset" / "original_datasets"

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
    "overstatement": ("Deception through overstatement arises when S exaggerates an aspect"
                        "of something featured in the proposition about which she intends to deceive H."),
    "obfuscation": ("S says something that S believes is true but believes that H will not be able to fully understand it and"
                    "will therefore be deceived, as might happen when S intentionally uses technical jargon or euphemisms."),
    "half_truth": "Half-truths involve revealing only part of the truth, while concealing another part that involves the nexus of deception.",
    "augmentation": "In augmentation, S adds something gratuitous to the truth, thereby deceiving H.",
}

