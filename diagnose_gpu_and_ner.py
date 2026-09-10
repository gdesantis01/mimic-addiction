import torch
import cupy
import spacy
import thinc
import config
from pipeline_adapter import NER_DIR, Pipeline

print("=== GPU DIAGNOSTIC ===")
print("Python:", __import__("sys").executable)
print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("Torch available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Torch GPU:", torch.cuda.get_device_name(0))

print("CuPy:", cupy.__version__)
print("CuPy devices:", cupy.cuda.runtime.getDeviceCount())

print("spaCy:", spacy.__version__)
print("Thinc:", thinc.__version__)

try:
    spacy.require_gpu()
    print("spaCy require_gpu: OK")
except Exception as e:
    print("spaCy require_gpu: FAILED")
    print(type(e).__name__, e)

print("======================")


# Sentences in the style the model was TRAINED on (synthetic, well-formed).
# If it fires on these but not on MIMIC, the problem is domain shift.
# If it fires on neither, the problem is the model or the wiring.
SYNTHETIC_STYLE = [
    "The patient reports a history of alcohol abuse and heavy smoking.",
    "The patient was diagnosed with type 2 diabetes mellitus in 2018.",
    "Genetic testing revealed a BRCA1 mutation.",
    "The patient has a certified disability of 75 percent.",
    "The couple underwent in vitro fertilization treatment.",
    "The radiology report confirms the presence of a pulmonary nodule.",
]

# Sentences in MIMIC style: telegraphic, abbreviated, list-formatted.
MIMIC_STYLE = [
    "Social History: Tob: 1ppd x 30yrs. EtOH: social. Illicits: denies.",
    "PMH: HTN, DM2, CAD s/p CABG.",
    "Pt reports drinking approximately 6 beers daily.",
    "___ year old male with h/o polysubstance abuse presenting with AMS.",
    "Denies tobacco, alcohol, or illicit drug use.",
]

print("=" * 70)
print("[1] pipeline structure")
print("=" * 70)

nlp = spacy.load(NER_DIR)
print(f"  path       : {NER_DIR}")
print(f"  lang       : {nlp.lang}")
print(f"  pipe_names : {nlp.pipe_names}")

if "ner" not in nlp.pipe_names:
    print("\n  >>> CAUSE FOUND: there is no 'ner' component in this pipeline.")
    print("      You loaded a model directory that does not contain a trained")
    print("      NER, or the component is named something else. Check")
    print("      config.cfg in the model directory for the component name.")
    raise SystemExit(1)

ner = nlp.get_pipe("ner")
labels = list(ner.labels)
print(f"  ner labels : {labels}")

if not labels:
    print("\n  >>> CAUSE FOUND: the NER component has no labels. The model was")
    print("      saved before training, or you loaded model-last from a run")
    print("      that never completed an epoch. Check for model-best.")
    raise SystemExit(1)

configured = set(config.PHI_CATEGORIES_NER)
emitted = set(labels)
if not (emitted & configured):
    print("\n  >>> CAUSE FOUND: label-name mismatch.")
    print(f"      model emits    : {sorted(emitted)}")
    print(f"      config expects : {sorted(configured)}")
    print("      The pipeline is detecting entities, but every prediction is")
    print("      being discarded downstream because the names do not match.")
    print("      Fix config.PHI_CATEGORIES and config.ADDICTION_LABEL to use")
    print("      the names on the left, then re-run. No re-training needed.")
    print("      (This is what check_adapter.py checks first.)")

print()
print("=" * 70)
print("[2] raw spaCy output, bypassing the adapter entirely")
print("=" * 70)

for label, probes in (("TRAINING-STYLE", SYNTHETIC_STYLE), ("MIMIC-STYLE", MIMIC_STYLE)):
    print(f"\n  --- {label} ---")
    total = 0
    for doc in nlp.pipe(probes):
        ents = [(e.text, e.label_) for e in doc.ents]
        total += len(ents)
        print(f"    {doc.text[:58]:<60} -> {ents}")
    print(f"    total entities: {total}")
    if total == 0 and label == "TRAINING-STYLE":
        print("\n    >>> The model produces nothing even on text resembling its")
        print("        own training distribution. This is not domain shift.")
        print("        Verify you loaded the correct checkpoint: the directory")
        print("        should be `model-best`, not `model-last` from an aborted")
        print("        run, and not the base model before fine-tuning.")

print()
print("=" * 70)
print("[3] same probes through the adapter")
print("=" * 70)

pipe = Pipeline()
for label, probes in (("TRAINING-STYLE", SYNTHETIC_STYLE), ("MIMIC-STYLE", MIMIC_STYLE)):
    print(f"\n  --- {label} ---")
    spans = pipe.tag(probes)
    for t, sp in zip(probes, spans):
        print(f"    {t[:58]:<60} -> {sp}")

print()
print("=" * 70)
print("[4] classifier behaviour")
print("=" * 70)
print("  Experiment 1 flagged 20,372 of 26,465 sentences (77%) while achieving")
print("  0.62 recall on ADDICTION-bearing ones. Flagging 77% of sentences at")
print("  random would yield ~0.77 recall, so ADDICTION sentences were forwarded")
print("  LESS often than average sentences. Checking whether the classifier is")
print("  near-degenerate on this corpus:\n")

mixed = SYNTHETIC_STYLE + MIMIC_STYLE + [
    "Vital signs remained stable overnight.",
    "Discharge instructions were reviewed with the patient.",
    "Follow up in two weeks.",
    "The wound was clean, dry, and intact.",
]
preds = pipe.classify(mixed)
rate = sum(preds) / len(preds)
for p, t in zip(preds, mixed):
    print(f"    {p}  {t[:64]}")
print(f"\n  positive rate on these probes: {rate:.0%}")
if rate > 0.9:
    print("  >>> the classifier fires on nearly everything, including clearly")
    print("      non-sensitive administrative sentences. Its gate is not")
    print("      discriminating on this corpus and the 77% flag rate in")
    print("      Experiment 1 is not an artifact of the category mismatch.")
