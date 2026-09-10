"""
Scope-restricted evaluation.

WHY THIS EXISTS
---------------
The unrestricted evaluation compares two nearly disjoint sets. MIMIC-SBDH
annotates the Social History section; the NER fires on problem lists, discharge
diagnoses, and hospital-course narrative. Their intersection is close to empty
by construction, so the resulting F1 measures annotation scope rather than
model quality.

This script restores a fair comparison in two ways:

  A. RESTRICTED EVALUATION. Score only sentences inside a Social History
     section, where the MIMIC-SBDH annotation is exhaustive and a missed gold
     span is a genuine miss.

  B. OUT-OF-SCOPE INVENTORY. Export every prediction outside those sections to
     a CSV for manual adjudication. These are not false positives in any
     meaningful sense; they are detections the reference does not cover.

Additionally reports the restricted evaluation with and without negated gold
mentions ("denies alcohol use", "never smoked"), since a model trained on
affirmative synthetic assertions has no reason to treat those as entities and
including them conflates two distinct failure modes.

    python evaluate_scoped.py

Outputs into RESULTS_DIR:
    exp1b_scoped_metrics.csv       metrics under each scope condition
    exp1b_scope_breakdown.csv      how sentences and gold spans distribute
    exp1b_out_of_scope_preds.csv   predictions to adjudicate manually
"""

import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config
from pipeline_adapter import get_tokenizer
from run_experiments import prf, spans_to_bio, token_counts

RESULTS_DIR = Path("./results")


def log(msg):
    print(f"[scoped] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

from scope_sections import apply_scope   # section-boundary tracking

NEGATION = re.compile(
    r"\b(denie[sd]|denies|never|none|non-?smoker|no\s+h/?o|negative|"
    r"quit|former|ex-smoker|abstinen)", re.I)


def is_negated(text):
    return bool(NEGATION.search(text))


# ---------------------------------------------------------------------------

def load_rows():
    """
    Reads the per-sentence diagnostic CSV if present, otherwise reconstructs
    from work/sbdh_sentences.jsonl plus a fresh inference pass.
    """
    diag = RESULTS_DIR / "exp1_ner_vs_gold.csv"
    if diag.exists():
        log(f"reading {diag.name}")
        df = pd.read_csv(diag)
        rows = []
        for r in df.itertuples():
            rows.append({
                "row_id": getattr(r, "row_id", None),
                "sent_idx": getattr(r, "sent_idx", 0),
                "text": str(r.text),
                "gold": ast.literal_eval(r.gold_spans) if isinstance(r.gold_spans, str) else [],
                "pred": ast.literal_eval(r.ner_spans) if isinstance(r.ner_spans, str) else [],
            })
        return rows

    log("diagnostic CSV not found; running inference from work/sbdh_sentences.jsonl")
    from pipeline_adapter import Pipeline
    pipe = Pipeline()
    src = [json.loads(l) for l in open(config.WORK_DIR / "sbdh_sentences.jsonl")]
    texts = [r["text"] for r in src]
    preds = []
    B = config.BATCH_SIZE
    for i in range(0, len(texts), B):
        batch = texts[i:i + B]
        for spans in pipe.tag(batch):
            preds.append([(s, e) for (s, e, lab) in spans
                          if lab == config.ADDICTION_LABEL])
    return [{"row_id": r.get("row_id"), "sent_idx": r.get("sent_idx", 0),
             "text": r["text"], "gold": r["gold_spans"], "pred": p}
            for r, p in zip(src, preds)]


def normalize_pred(p):
    """Predictions may be (s,e) or (s,e,label); reduce to (s,e)."""
    out = []
    for item in p:
        if len(item) >= 2:
            out.append((int(item[0]), int(item[1])))
    return out


def score(rows, nlp, label):
    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in rows:
        g_tags, _ = spans_to_bio(r["text"], r["gold"], nlp, label)
        p_tags, _ = spans_to_bio(r["text"], normalize_pred(r["pred"]), nlp, label)
        for tag, c in token_counts(g_tags, p_tags).items():
            for k in ("tp", "fp", "fn"):
                agg[tag][k] += c[k]
    tp = sum(c["tp"] for c in agg.values())
    fp = sum(c["fp"] for c in agg.values())
    fn = sum(c["fn"] for c in agg.values())
    p, r_, f = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r_, "f1": f}, agg


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    nlp = get_tokenizer()
    rows = load_rows()
    log(f"{len(rows)} sentences")

    apply_scope(rows)          # sets r["in_scope"] with section tracking
    for r in rows:
        r["negated"] = is_negated(r["text"])
        r["has_gold"] = bool(r["gold"])
        r["has_pred"] = bool(r["pred"])

    # ---- breakdown -------------------------------------------------------
    def n(pred_fn):
        return sum(1 for r in rows if pred_fn(r))

    breakdown = [
        {"partition": "all sentences", "sentences": len(rows),
         "with_gold": n(lambda r: r["has_gold"]),
         "with_pred": n(lambda r: r["has_pred"])},
        {"partition": "in social-history scope",
         "sentences": n(lambda r: r["in_scope"]),
         "with_gold": n(lambda r: r["in_scope"] and r["has_gold"]),
         "with_pred": n(lambda r: r["in_scope"] and r["has_pred"])},
        {"partition": "out of scope",
         "sentences": n(lambda r: not r["in_scope"]),
         "with_gold": n(lambda r: not r["in_scope"] and r["has_gold"]),
         "with_pred": n(lambda r: not r["in_scope"] and r["has_pred"])},
        {"partition": "in scope, affirmative gold",
         "sentences": n(lambda r: r["in_scope"] and not r["negated"]),
         "with_gold": n(lambda r: r["in_scope"] and not r["negated"] and r["has_gold"]),
         "with_pred": n(lambda r: r["in_scope"] and not r["negated"] and r["has_pred"])},
        {"partition": "in scope, negated gold",
         "sentences": n(lambda r: r["in_scope"] and r["negated"]),
         "with_gold": n(lambda r: r["in_scope"] and r["negated"] and r["has_gold"]),
         "with_pred": n(lambda r: r["in_scope"] and r["negated"] and r["has_pred"])},
    ]
    pd.DataFrame(breakdown).to_csv(
        RESULTS_DIR / "exp1b_scope_breakdown.csv", index=False)
    log("wrote exp1b_scope_breakdown.csv")
    for b in breakdown:
        log(f"  {b['partition']:<34} sent={b['sentences']:>6}  "
            f"gold={b['with_gold']:>4}  pred={b['with_pred']:>4}")

    # ---- scored conditions ----------------------------------------------
    conditions = [
        ("unrestricted", rows),
        ("social-history scope", [r for r in rows if r["in_scope"]]),
        ("social-history scope, affirmative only",
         [r for r in rows if r["in_scope"] and not r["negated"]]),
    ]

    out = []
    for name, subset in conditions:
        if not subset:
            log(f"  {name}: empty partition, skipped")
            continue
        m, _ = score(subset, nlp, config.ADDICTION_LABEL)
        m["condition"] = name
        m["sentences"] = len(subset)
        out.append(m)
        log(f"  {name:<40} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})")

    cols = ["condition", "sentences", "tp", "fp", "fn",
            "precision", "recall", "f1"]
    pd.DataFrame(out)[cols].to_csv(
        RESULTS_DIR / "exp1b_scoped_metrics.csv", index=False)
    log("wrote exp1b_scoped_metrics.csv")

    # ---- out-of-scope predictions for manual adjudication ---------------
    oos = []
    for r in rows:
        if r["in_scope"] or not r["has_pred"]:
            continue
        for (s, e) in normalize_pred(r["pred"]):
            oos.append({
                "predicted_text": r["text"][s:e],
                "context": r["text"][max(0, s - 60):e + 60].replace("\n", " "),
                "correct_addiction_mention": "",   # fill in: yes / no
            })
    pd.DataFrame(oos).to_csv(
        RESULTS_DIR / "exp1b_out_of_scope_preds.csv", index=False)
    log(f"wrote exp1b_out_of_scope_preds.csv ({len(oos)} predictions to adjudicate)")
    log("")
    log("NEXT: open exp1b_out_of_scope_preds.csv and fill the")
    log("`correct_addiction_mention` column with yes/no. Two annotators and a")
    log("Cohen's kappa would be better than one, and the file is small enough")
    log("that this is a short task.")
    log("")
    log("NOTE: this file contains short excerpts of MIMIC text for adjudication.")
    log("It must NOT be published. Report only the adjudicated counts.")


if __name__ == "__main__":
    main()
