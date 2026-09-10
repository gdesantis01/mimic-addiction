"""
=============================================================================
THIS IS THE ONLY FILE YOU NEED TO EDIT.
=============================================================================

Concrete implementation for the ClinicalT5-base classifier + spaCy NER cascade.
Set the two paths in `Pipeline.__init__` and you should be able to run.

Before the first full run, execute:

    python check_adapter.py

which loads both models, prints the labels your NER actually emits, and runs a
handful of sentences through the cascade. It catches the two failure modes that
would otherwise silently corrupt every number downstream: a label-name mismatch
between your model and `config.ADDICTION_LABEL`, and character offsets that do
not line up with the text you passed in.

Two rules that matter for the results to be interpretable:

1.  `classify()` and `tag()` must be INDEPENDENT. Do not make `tag()` call
    `classify()` internally. The runner applies the cascade logic itself,
    because we need to measure each stage separately as well as end-to-end.

2.  `tag()` must return CHARACTER offsets relative to the string it was given,
    not token indices.
"""

from typing import List, Tuple

import torch
import label_normalization as label_norm

import config

Span = Tuple[int, int, str]


# -----------------------------------------------------------------------------
# EDIT THESE TWO PATHS
# -----------------------------------------------------------------------------

# Directory containing the fine-tuned ClinicalT5 classifier: the output of
# save_pretrained(), i.e. config.json + model weights + tokenizer files.
# Use the BALANCED fine-tuning checkpoint (750+750).
CLASSIFIER_DIR = "LLM_model"
LLM_FILENAME = "ClinicalT5-base_model_tte.pt"
LLM_MODEL_NAME = "hossboll/clinical-t5"
# Directory containing the trained spaCy pipeline. If you trained with
# `spacy train`, this is the `model-best` directory.
NER_DIR = "NER_model/model-best"

# Which logit index means "contains PHI". Check the `id2label` field in your
# classifier's config.json. Almost always 1, but verify rather than assume.
POSITIVE_CLASS_INDEX = 1

# Max token length for the classifier. Match what you used at training time.
MAX_LENGTH = 256


class Pipeline:
    """ClinicalT5-base classifier gating a spaCy NER tagger."""

    def __init__(self, classifier_dir: str = None, ner_dir: str = None):
        from transformers import AutoTokenizer, T5ForSequenceClassification
        import spacy

        classifier_dir = classifier_dir or CLASSIFIER_DIR
        ner_dir = ner_dir or NER_DIR

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEBUG] Chosen device: {self.device}")

        #self.tokenizer = AutoTokenizer.from_pretrained(classifier_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.clf = T5ForSequenceClassification.from_pretrained(LLM_MODEL_NAME, num_labels=2)

        state_dict = torch.load(classifier_dir + "/" + LLM_FILENAME, map_location=self.device)
        self.clf.load_state_dict(state_dict)

        self.clf.eval()
        self.clf.to(self.device)

        import spacy

        if torch.cuda.is_available():
            spacy.require_gpu()
            print("[DEBUG] spaCy device: GPU")
        else:
            spacy.require_cpu()
            print("[DEBUG] spaCy device: CPU")

        self.ner = spacy.load(ner_dir)

        # Labels your NER actually emits. Printed by check_adapter.py.
        self.ner_labels = set(self.ner.get_pipe("ner").labels)

    # ------------------------------------------------------------------ block 1

    @torch.no_grad()
    def classify(self, texts: List[str]) -> List[int]:
        """Sentence-level binary PHI classification. Returns 0/1 per input."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        logits = self.clf(**enc).logits
        # T5ForSequenceClassification returns (batch, num_labels).
        preds = logits.argmax(dim=-1)
        #return (preds == POSITIVE_CLASS_INDEX).long().tolist()
        return (preds != 2).long().tolist() #per avere tutto vero


    # ------------------------------------------------------------------ block 2

    def tag(self, texts: List[str]) -> List[List[Span]]:
        # """Span-level entity tagging. Character offsets relative to each input."""
        # out = []
        # for doc in self.ner.pipe(texts, batch_size=len(texts)):
        #     out.append([(e.start_char, e.end_char, e.label_) for e in doc.ents])
        # return out
        out = label_norm.tag(self, texts=texts)
        return out



# -----------------------------------------------------------------------------
# Tokenizer used for sentence splitting and for BIO alignment.
#
# This MUST match the tokenizer your NER model uses, otherwise gold and
# predicted BIO tags are aligned over different token sequences and every
# span-level metric is wrong. Loading your actual spaCy pipeline and reusing
# its tokenizer is the safe option; the blank fallback is only correct if your
# model uses spaCy's default English tokenizer unmodified.
# -----------------------------------------------------------------------------

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    import spacy

    try:
        nlp = spacy.load(NER_DIR, exclude=["ner"])
        if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    except Exception:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    nlp.max_length = 5_000_000
    _TOKENIZER = nlp
    return nlp
