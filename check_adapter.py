"""
Run this before `prepare_data.py`. It takes seconds and catches the failures
that would otherwise corrupt every downstream number without raising an error.

    python check_adapter.py
"""

import sys

import config
from pipeline_adapter import Pipeline, get_tokenizer, POSITIVE_CLASS_INDEX

PROBES = [
    "Social history: the patient reports drinking approximately six beers daily.",
    "Denies tobacco, alcohol, or illicit drug use.",
    "The patient has a history of hypertension and type 2 diabetes.",
    "Vital signs remained stable throughout the hospitalization.",
    "SH: +ETOH, +tob 1ppd x 20yrs, denies IVDU.",
]

ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  FAIL  {msg}")


def warn(msg):
    print(f"  WARN  {msg}")


print("loading models...")
pipe = Pipeline()
nlp = get_tokenizer()
print(f"  device: {pipe.device}")

# --- 1. label names -----------------------------------------------------------
print("\n[1] NER labels emitted by your model:")
print(f"  {sorted(pipe.ner_labels)}")

bio_addiction_labels = [ f"B-{config.ADDICTION_LABEL}", f"I-{config.ADDICTION_LABEL}"]

bio_labels = [
    f"{prefix}-{category}"
    for category in config.PHI_CATEGORIES
    for prefix in ("B", "I")
]

if not set(bio_addiction_labels).issubset(pipe.ner_labels):
    fail(f"At least one of ADDICTION_LABELS = '{config.ADDICTION_LABEL}' is NOT among them. "
         f"Experiment 1 would score zero for every metric. Fix config.py.")
else:
    print(f"  OK    '{config.ADDICTION_LABEL}' present")

missing = [c for c in bio_labels if c not in pipe.ner_labels]
if missing:
    warn(f"config.PHI_CATEGORIES lists labels your model does not emit: {missing}. "
         f"They will show as zero in Experiment 2. Remove them from config.py "
         f"if that is not intended.")

extra = [l for l in pipe.ner_labels if l not in bio_labels]
if extra:
    warn(f"your model emits labels absent from config.PHI_CATEGORIES: {extra}. "
         f"They will be silently dropped from the feasibility table.")

# --- 2. classifier ------------------------------------------------------------
print(f"\n[2] classifier (positive class index = {POSITIVE_CLASS_INDEX}):")
preds = pipe.classify(PROBES)
if len(preds) != len(PROBES):
    fail(f"classify() returned {len(preds)} results for {len(PROBES)} inputs")
for p, t in zip(preds, PROBES):
    print(f"  {p}  {t[:68]}")
if len(set(preds)) == 1:
    warn("classifier returned the same label for every probe. If that label is "
         "1, check POSITIVE_CLASS_INDEX against id2label in config.json.")

# --- 3. tagger and offset correctness ----------------------------------------
print("\n[3] tagger (verifying character offsets resolve to real substrings):")
spans = pipe.tag(PROBES)
if len(spans) != len(PROBES):
    fail(f"tag() returned {len(spans)} results for {len(PROBES)} inputs")

for text, sp in zip(PROBES, spans):
    if not sp:
        print(f"  (none)  {text[:60]}")
        continue
    for (s, e, lab) in sp:
        if not (0 <= s < e <= len(text)):
            fail(f"offset ({s},{e}) out of range for a {len(text)}-char string. "
                 f"tag() must return offsets relative to the input sentence.")
            continue
        print(f"  [{lab}] '{text[s:e]}'")

# --- 4. tokenizer consistency -------------------------------------------------
print("\n[4] tokenizer alignment check:")
probe = PROBES[0]
ner_toks = [t.text for t in pipe.ner.tokenizer(probe)]
aln_toks = [t.text for t in nlp.tokenizer(probe)]
if ner_toks != aln_toks:
    fail("get_tokenizer() and your NER model tokenize differently. Gold and "
         "predicted BIO tags would be aligned over different sequences and "
         "every span-level metric would be wrong.")
    print(f"    NER : {ner_toks}")
    print(f"    ALGN: {aln_toks}")
else:
    print(f"  OK    identical tokenization ({len(ner_toks)} tokens)")

# --- 5. sentence splitting ----------------------------------------------------
print("\n[5] sentence splitting on a telegraphic clinical fragment:")
frag = ("Social History:\nMarried, lives with wife.\n"
        "Tob: 1ppd x 30 yrs, quit 2 yrs ago.\nEtOH: social.\nIllicits: denies.")
sents = [s.text for s in nlp(frag).sents]
for s in sents:
    print(f"  | {s!r}")
print("  ^ if these are wrong, gold spans will straddle boundaries and be "
      "dropped. The count is reported by prepare_data.py; watch it.")

print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED - fix before running"))
sys.exit(0 if ok else 1)
