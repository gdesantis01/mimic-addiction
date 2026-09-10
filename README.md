# Evaluating a synthetically trained GDPR-grounded PHI detector on real clinical notes

Evaluation harness and aggregate results for *What Transfers and What Does Not*
(Clinical-MIRF @ CIKM 2026).

The detector under evaluation is a two-stage cascade — a ClinicalT5-base
sentence classifier gating a spaCy NER tagger — fine-tuned on a synthetic
corpus annotated to a GDPR Art. 9 taxonomy and described in the companion
paper. This repository does not contain model weights. It contains the code
that applies that detector to MIMIC without adaptation, and the aggregate
numbers that came out.

## What is and is not here

`results/` holds aggregate figures only: counts, rates, metrics, timings. No
clinical text, no patient, admission, or note identifiers. Safe to publish
under the PhysioNet Data Use Agreement.

`work/` is produced locally by `prepare_data.py` and **does** contain MIMIC
text. It is gitignored and must never be committed.

Two files the harness writes into `results/` also contain MIMIC text —
`exp1_ner_vs_gold.csv` (full sentences) and `exp1b_out_of_scope_preds.csv`
(spans with context windows). Run `sanitize_results.py` to replace them with
text-free equivalents, then delete the originals. **If either was ever
committed, removing it from the working tree does not remove it from git
history**; rewrite the history before making the repository public.

## Data

All MIMIC resources require PhysioNet credentialed access and completed
human-subjects training. MIMIC-SBDH is distributed on GitHub as annotations
only and requires MIMIC-III for the underlying text.

| Resource | Version | Used for |
|---|---|---|
| [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) | 1.4 | note text for the MIMIC-SBDH annotations |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) | 2.2 | unlabeled corpus for the feasibility analysis |
| [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) | 2.2 | `diagnoses_icd` for the ICD comparison |
| [MIMIC-SBDH](https://github.com/hibaahsan/MIMIC-SBDH) | — | gold substance-use annotations |

We use the matched MIMIC-IV / MIMIC-IV-Note v2.2 pair rather than MIMIC-IV
v3.1: the note module has not been re-released beyond v2.2, so a v3.1 cohort
would include admissions with no corresponding narrative.

Expected layout under `DATA_ROOT`:

```
MIMIC-SBDH/
  MIMIC-SBDH.csv
  MIMIC-SBDH-keywords.csv
physionet.org/files/
  mimiciii/1.4/NOTEEVENTS.csv.gz
  mimic-iv-note/2.2/note/discharge.csv.gz
  mimiciv/2.2/hosp/diagnoses_icd.csv.gz
```

Leave the `.gz` archives compressed. The scripts read them directly, and
decompressing on Windows in text mode corrupts the multi-line quoted `text`
field in ways that surface much later as parse errors.

## Running

```bash
pip install -r requirements.txt

# 1. config.py       -> set DATA_ROOT
# 2. pipeline_adapter.py -> point CLASSIFIER_DIR and NER_DIR at your models

python check_adapter.py     # do not skip this
python prepare_data.py
python run_experiments.py
python evaluate_scoped.py
python sanitize_results.py  # before committing anything under results/
```

Start with `N_SBDH_NOTES = 200` and `N_MIMIC4_NOTES = 200` for a smoke test.

### check_adapter.py is not optional

It takes seconds and catches four failures that otherwise corrupt every
downstream number without raising an error: a mismatch between the labels your
NER emits and those in `config.py`; a `POSITIVE_CLASS_INDEX` that disagrees with
the classifier's `id2label`; character offsets that do not resolve to real
substrings; and a tokenizer used for BIO alignment that differs from the one
the NER model uses. Each of these produces plausible-looking metrics rather
than a crash.

## Scripts

| Script                   | Purpose |
|--------------------------|---|
| `config.py`              | paths, sample sizes, category names, ICD code ranges |
| `pipeline_adapter.py`    | the only file requiring your code: model loading and inference |
| `check_adapter.py`       | pre-flight checks (see above) |
| `prepare_data.py`        | builds evaluation sets; projects gold spans onto sentences |
| `run_experiments.py`     | experiments 1–3, writes `results/` |
| `scope_sections.py`      | section-boundary tracking through each note |
| `evaluate_scoped.py`     | scope-restricted evaluation and out-of-scope export |
| `label_normalization.py` | recovers spans from a BIO-as-entity-type NER (see below) |
| `sanitize_results.py`    | strips MIMIC text from results before publication |
| `diagnose_discharge.py`  | verifies a MIMIC download against its published checksum |
| `diagnose_gpu_and_ner.py` | isolates why the NER returns nothing |
| `diagnose_offsets.py`     | Diagnoses whether the mismatch is caused by an offset misalignment |

## Experiments

**1 — detection accuracy on `ADDICTION`.** The pipeline is applied with no
target-domain fine-tuning. Ground truth is the character-offset keyword spans
in `MIMIC-SBDH-keywords.csv`, restricted to `behavior_alcohol`,
`behavior_tobacco`, and `behavior_drug`, projected onto sentences and converted
to BIO tags over spaCy tokenization. Notes covered by MIMIC-SBDH carrying no
addiction keyword are true negatives, since annotators reviewed them.

Because MIMIC-SBDH annotates the social history section and is silent
elsewhere, results are reported unrestricted, restricted to social-history
regions, and restricted further to affirmative gold mentions. Predictions
outside the annotated region are exported for manual adjudication rather than
scored. The classification stage is evaluated only as a gate: its recall is
interpretable, its precision against an `ADDICTION`-only reference is not.

**2 — feasibility at scale.** Flagging rate per category, throughput, and
projected full-corpus runtime over unlabeled MIMIC-IV-Note. Descriptive, not an
accuracy benchmark: no ground truth exists for these categories.

**3 — cross-modal consistency.** Admissions whose notes were flagged for
`ADDICTION` against admissions carrying substance-use ICD codes. ICD coding is
not a gold standard for the presence of a textual mention, so this measures
agreement between two imperfect views rather than validating either.

## Known issues in the evaluated model

Two defects in the detector are visible in these results and are reported in
the paper rather than worked around silently.

The NER was trained with BIO tags supplied to spaCy as entity **types**, so it
learned `B-ADDICTION` and `I-ADDICTION` as unrelated classes rather than as the
internal encoding of one. It emits single-token fragments where complete spans
are expected. `label_normalization.py` strips the prefixes and merges adjacent
same-category spans, which recovers usable spans for evaluation but cannot
recover boundaries the model never learned. Set `NORMALIZE_BIO_LABELS = False`
after retraining with bare category labels.

The classification gate forwards 77% of sentences while achieving lower recall
on gold-bearing sentences than that rate would imply by chance. Bypassing it
yields marginally more true positives. The gate is reported as an ablation.

## Output files

| File | Contents |
|---|---|
| `exp1_addiction_metrics.csv` | precision/recall/F1 per stage and BIO tag |
| `exp1_confusion.csv` | TP/FP/FN/TN and false positive rate |
| `exp1_class_balance.csv` | positive rate of the evaluation set |
| `exp1b_scope_breakdown.csv` | gold and predictions by document region |
| `exp1b_scoped_metrics.csv` | metrics under each scope condition |
| `exp1b_adjudication_summary.csv` | adjudication category counts (text-free) |
| `exp1b_span_length_hist.csv` | span length distribution (text-free) |
| `exp1_sentence_stats.csv` | per-sentence outcome flags (text-free) |
| `exp2_feasibility.csv` | flagging proportions per category |
| `exp2_throughput.csv` | sentences/sec and projected runtime |
| `exp3_icd_agreement.csv` | text flags against ICD codes |
| `run_manifest.json` | seed, versions, sizes, timings |

## Citation

```bibtex
@inproceedings{vergallo2026transfers,
  author    = {Vergallo, Roberto and De Santis, Gabriele and
               Vetrani, Claudia and Mainetti, Luca},
  title     = {What Transfers and What Does Not: Evaluating a Synthetically
               Trained {GDPR}-Grounded {PHI} Detector on Real Clinical Notes},
  booktitle = {Clinical-MIRF @ CIKM},
  year      = {2026}
}
```

## Acknowledgments

Project IN-DEEP, no. F/350283/05/X60, CUP B69J24002350005. MIMIC data accessed
under PhysioNet credentialed access; we thank the MIT Laboratory for
Computational Physiology and the authors of MIMIC-SBDH.
