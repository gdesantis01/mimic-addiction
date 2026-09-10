"""
Step 2 of 2. Runs the pipeline over the prepared data and writes result CSVs.

Everything written to RESULTS_DIR is aggregate and contains no MIMIC text and
no MIMIC identifiers. Those files are the ones you publish.

Outputs (into results/):

    exp1_addiction_metrics.csv      per-stage, per-tag precision/recall/F1
    exp1_confusion.csv              raw TP/FP/FN/TN counts behind those metrics
    exp1_class_balance.csv          positive/negative rates in the test set
    exp2_feasibility.csv            flagged-sentence proportions per category
    exp2_throughput.csv             sentences/sec and projected corpus runtime
    exp3_icd_agreement.csv          pipeline flags vs. substance-use ICD codes
    run_manifest.json               versions, seed, sizes, timestamps

Run:  python run_experiments.py
"""

import json
import sys
import platform

import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd

import config
from pipeline_adapter import Pipeline, get_tokenizer  # noqa


def log(msg):
    print(f"[run] {msg}", flush=True)


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------------------------------------------------------------------------
# BIO conversion and metrics
# ---------------------------------------------------------------------------

def spans_to_bio(text, spans, nlp, label):
    """
    Convert character spans to per-token BIO tags over spaCy tokenization.

    Spans that do not align to token boundaries are snapped outward using
    spaCy's `expand` alignment mode; the count of such cases is returned so
    it can be reported rather than hidden.
    """
    doc = nlp.tokenizer(text)
    tags = ["O"] * len(doc)
    misaligned = 0
    for s, e in spans:
        span = doc.char_span(s, e, alignment_mode="expand")
        if span is None:
            misaligned += 1
            continue
        for k, tok in enumerate(span):
            tags[tok.i] = ("B-" if k == 0 else "I-") + label
    return tags, misaligned


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def token_counts(gold_tags, pred_tags):
    """Per-tag TP/FP/FN over aligned token tag sequences."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for g, p in zip(gold_tags, pred_tags):
        if g == p:
            if g != "O":
                counts[g]["tp"] += 1
        else:
            if g != "O":
                counts[g]["fn"] += 1
            if p != "O":
                counts[p]["fp"] += 1
    return counts


# ---------------------------------------------------------------------------
# Experiment 1: quantitative ADDICTION evaluation on MIMIC-SBDH
# ---------------------------------------------------------------------------

def spans_to_text(text, spans):
    """Convert character spans into the actual text they cover."""
    return [
        text[start:end]
        for start, end in spans
    ]


def mark_spans(text, spans):
    """Return text with each span surrounded by [ ]."""
    if not spans:
        return text

    spans = sorted(spans, key=lambda x: x[0])

    parts = []
    pos = 0

    for start, end in spans:
        # Skip overlapping spans
        if start < pos:
            continue

        parts.append(text[pos:start])
        parts.append(f"[{text[start:end]}]")
        pos = end

    parts.append(text[pos:])

    return "".join(parts)

# Experiment 1: quantitative ADDICTION evaluation on MIMIC-SBDH
# (LLM gate + cascaded NER, with full statistics/CSV outputs)
# ---------------------------------------------------------------------------

def experiment_1(pipe, nlp, verbose=True):
    log("=== experiment 1: ADDICTION on MIMIC-SBDH (zero target-domain exposure) ===")

    rows = [
        json.loads(l)
        for l in open(config.WORK_DIR / "sbdh_sentences.jsonl", encoding="utf-8")
    ]
    log(f"{len(rows)} sentences")

    texts = [r["text"] for r in rows]
    gold_sent = [1 if r["gold_spans"] else 0 for r in rows]

    n_pos = sum(gold_sent)
    log(f"class balance: {n_pos} positive / {len(rows) - n_pos} negative "
        f"({100 * n_pos / len(rows):.2f}% positive)")

    # ---- Block 1: classification (LLM gate) -------------------------------
    preds_sent = []
    t0 = time.time()
    for batch in batched(texts, config.BATCH_SIZE):
        preds_sent.extend(pipe.classify(batch))
    t_clf = time.time() - t0
    log(f"classification done in {t_clf:.1f}s")

    tp = sum(1 for g, p in zip(gold_sent, preds_sent) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold_sent, preds_sent) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold_sent, preds_sent) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gold_sent, preds_sent) if g == 0 and p == 0)
    clf_p, clf_r, clf_f = prf(tp, fp, fn)
    clf_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    log(f"LLM: gate recall={clf_r:.4f} (this is the interpretable figure)")
    log(f"     raw P={clf_p:.4f} F1={clf_f:.4f} FPR={clf_fpr:.4f} "
        f"-- see caveat below")
    log("     CAVEAT: the classifier fires on ANY PHI category, while this "
        "ground truth covers ADDICTION only. Sentences correctly flagged for "
        "CLIN_COND, MED_REP etc. are counted as false positives here. Precision, "
        "F1 and FPR against this reference are therefore LOWER BOUNDS and are "
        "not measures of classifier quality. Recall IS interpretable: it is the "
        "proportion of ADDICTION-bearing sentences the gate forwards to the NER, "
        "which is what determines the cascade's ceiling.")

    # ---- Block 2: tagging, cascaded ----------------------------------------
    # Only sentences the classifier accepted are passed to the NER. Sentences
    # it rejected get an empty prediction, which is what the deployed system
    # would do.
    gate = [i for i, p in enumerate(preds_sent) if p == 1]
    gate_set = set(gate)
    log(f"cascade: {len(gate)}/{len(rows)} sentences forwarded to NER")

    pred_spans = [[] for _ in rows]
    t0 = time.time()
    for batch_idx in batched(gate, config.BATCH_SIZE):
        out = pipe.tag([texts[i] for i in batch_idx])
        for i, spans in zip(batch_idx, out):
            pred_spans[i] = [
                (s, e) for (s, e, lab) in spans if lab == config.ADDICTION_LABEL
            ]
            # BUG FIX: this used to print `gate[i]`, but `i` here is already
            # the row index (an element taken from `gate`), not a position
            # inside `gate`. Indexing `gate` with it was wrong (silently
            # printed the wrong row, or raised IndexError once i >= len(gate)).
            if pred_spans[i] and verbose:
                print(f"{i}: {pred_spans[i]}")
    t_ner = time.time() - t0
    log(f"tagging done in {t_ner:.1f}s")

    # ---- print predictions vs gold (from experiment_1_check_positives) ----
    if verbose:
        for r, pspans in zip(rows, pred_spans):
            print(f'row_id: {r["row_id"]}\tsent_idx: {r["sent_idx"]}')
            print(r["text"])
            print(f'NER:        {pspans}')
            print(f'gold_spans: {r["gold_spans"]}')
            print()

    # ---- BIO alignment and token-level metrics (incl. TN) ------------------
    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # TN is kept separately because it is not associated with a BIO label.
    global_tn = 0

    mis_gold = mis_pred = 0

    for r, pspans in zip(rows, pred_spans):
        g_tags, m1 = spans_to_bio(r["text"], r["gold_spans"], nlp, config.ADDICTION_LABEL)
        p_tags, m2 = spans_to_bio(r["text"], pspans, nlp, config.ADDICTION_LABEL)

        mis_gold += m1
        mis_pred += m2

        # Same logic used later for the per-sentence CSV.
        sent_counts = token_counts(g_tags, p_tags)

        for tag, c in sent_counts.items():
            for k in ("tp", "fp", "fn"):
                agg[tag][k] += c[k]

        # Token-level TN: gold = O AND prediction = O
        global_tn += sum(
            1 for g, p in zip(g_tags, p_tags) if g == "O" and p == "O"
        )

    if mis_gold or mis_pred:
        log(
            f"note: {mis_gold} gold and {mis_pred} predicted spans could not be "
            f"aligned to token boundaries and were skipped"
        )

    # ---- assemble metrics ---------------------------------------------------
    g_tp = sum(c["tp"] for c in agg.values())
    g_fp = sum(c["fp"] for c in agg.values())
    g_fn = sum(c["fn"] for c in agg.values())
    g_tn = global_tn

    gp, gr, gf = prf(g_tp, g_fp, g_fn)

    out_rows = [{
        "stage": "LLM", "tag": "gate recall (ADDICTION)",
        "precision": "", "recall": clf_r, "f1": "",
        "interpretable": "yes",
        "note": "proportion of ADDICTION-bearing sentences forwarded to NER",
    }, {
        "stage": "LLM", "tag": "sentence-level (raw)",
        "precision": clf_p, "recall": clf_r, "f1": clf_f,
        "interpretable": "recall only",
        "note": ("classifier detects ANY PHI category; reference covers ADDICTION "
                 "only, so precision/F1 are lower bounds, not quality measures"),
    }, {
        "stage": "NER", "tag": "global",
        "precision": gp, "recall": gr, "f1": gf,
        "interpretable": "yes",
        "note": "token-level metrics over all sentences; cascade means "
                "non-forwarded sentences are scored as empty predictions",
    }]

    for tag in sorted(agg):
        c = agg[tag]
        p, r_, f = prf(c["tp"], c["fp"], c["fn"])
        out_rows.append({
            "stage": "NER", "tag": tag,
            "precision": p, "recall": r_, "f1": f,
            "interpretable": "yes", "note": "",
        })

    pd.DataFrame(out_rows).to_csv(
        config.RESULTS_DIR / "exp1_addiction_metrics.csv", index=False
    )

    # ---- confusion matrix (incl. NER TN, from check_positives) -------------
    conf = [{
        "stage": "LLM", "tag": "sentence-level",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "fpr": clf_fpr,
        "note": "fp inflated by non-ADDICTION PHI categories",
    }, {
        "stage": "NER", "tag": "global",
        "tp": g_tp, "fp": g_fp, "fn": g_fn, "tn": g_tn,
        "fpr": (g_fp / (g_fp + g_tn) if (g_fp + g_tn) > 0 else 0.0),
        "note": "",
    }]

    for tag in sorted(agg):
        c = agg[tag]
        conf.append({
            "stage": "NER", "tag": tag,
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": "", "fpr": "",
            "note": "",
        })

    pd.DataFrame(conf).to_csv(
        config.RESULTS_DIR / "exp1_confusion.csv", index=False
    )

    # ---- class balance -------------------------------------------------------
    pd.DataFrame([{
        "n_sentences": len(rows),
        "n_positive": n_pos,
        "n_negative": len(rows) - n_pos,
        "positive_rate": n_pos / len(rows) if rows else 0.0,
        "n_forwarded_to_ner": len(gate),
    }]).to_csv(config.RESULTS_DIR / "exp1_class_balance.csv", index=False)

    # ---- visual comparison CSV (from experiment_1_check_positives) --------
    comparison_rows = []

    for idx, (r, pspans) in enumerate(zip(rows, pred_spans)):

        g_tags, _ = spans_to_bio(r["text"], r["gold_spans"], nlp, config.ADDICTION_LABEL)
        p_tags, _ = spans_to_bio(r["text"], pspans, nlp, config.ADDICTION_LABEL)

        # IMPORTANT: use exactly the same token-level logic as the confusion matrix.
        sent_counts = token_counts(g_tags, p_tags)

        sent_tp = sum(c["tp"] for c in sent_counts.values())
        sent_fp = sum(c["fp"] for c in sent_counts.values())
        sent_fn = sum(c["fn"] for c in sent_counts.values())
        sent_tn = sum(1 for g, p in zip(g_tags, p_tags) if g == "O" and p == "O")

        # Determine sentence status.
        status = []
        if sent_tp > 0:
            status.append("TP")
        if sent_fp > 0:
            status.append("FP")
        if sent_fn > 0:
            status.append("FN")
        if not status:
            status.append("TN")
        status = "+".join(status)

        comparison_rows.append({
            "row_id": r["row_id"],
            "sent_idx": r["sent_idx"],
            "text": r["text"],

            # LLM gate info (new: not present in check_positives, since that
            # function never ran the classifier).
            "llm_pred": preds_sent[idx],
            "forwarded_to_ner": idx in gate_set,

            # Original spans
            "gold_spans": str(r["gold_spans"]),
            "ner_spans": str(pspans),

            # Actual text corresponding to spans
            "gold_words": str(spans_to_text(r["text"], r["gold_spans"])),
            "ner_words": str(spans_to_text(r["text"], pspans)),

            # Text with spans highlighted
            "gold_tagged_text": mark_spans(r["text"], r["gold_spans"]),
            "ner_tagged_text": mark_spans(r["text"], pspans),

            # Token-level counts
            "tp": sent_tp,
            "fp": sent_fp,
            "fn": sent_fn,
            "tn": sent_tn,

            # Sentence-level status
            "status": status,
        })

    pd.DataFrame(comparison_rows).to_csv(
        config.RESULTS_DIR / "exp1_ner_vs_gold.csv", index=False
    )

    log("wrote exp1_*.csv")

    return {
        "n_sentences": len(rows),
        "positive_rate": n_pos / len(rows) if rows else 0.0,
        "t_classify_s": t_clf,
        "t_tag_s": t_ner,
    }

def experiment_1_no_llm(pipe, nlp):
    # ---- load all sentences ----------------------------------------------
    rows = [
        json.loads(l)
        for l in open(
            config.WORK_DIR / "sbdh_sentences.jsonl",
            encoding="utf-8"
        )
    ]

    log(f"{len(rows)} sentences")

    texts = [r["text"] for r in rows]

    # ---- NER --------------------------------------------------------------
    # All sentences are forwarded directly to the NER.
    log(f"forwarding {len(rows)}/{len(rows)} sentences to NER")

    pred_spans = [[] for _ in rows]

    t0 = time.time()

    for batch_idx in batched(range(len(rows)), config.BATCH_SIZE):
        out = pipe.tag([texts[i] for i in batch_idx])

        for i, spans in zip(batch_idx, out):
            pred_spans[i] = [
                (s, e)
                for (s, e, lab) in spans
                if lab == config.ADDICTION_LABEL
            ]

    t_ner = time.time() - t0
    log(f"tagging done in {t_ner:.1f}s")

    # ---- print predictions vs gold ----------------------------------------
    for r, pspans in zip(rows, pred_spans):
        print(f'row_id: {r["row_id"]}\tsent_idx: {r["sent_idx"]}')
        print(r["text"])
        print(f'NER:        {pspans}')
        print(f'gold_spans: {r["gold_spans"]}')
        print()

    # ---- BIO alignment and token-level metrics ---------------------------
    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # TN is kept separately because it is not associated with a BIO label.
    global_tn = 0

    mis_gold = mis_pred = 0

    for r, pspans in zip(rows, pred_spans):

        g_tags, m1 = spans_to_bio(
            r["text"],
            r["gold_spans"],
            nlp,
            config.ADDICTION_LABEL
        )

        p_tags, m2 = spans_to_bio(
            r["text"],
            pspans,
            nlp,
            config.ADDICTION_LABEL
        )

        mis_gold += m1
        mis_pred += m2

        # Same logic used later for the per-sentence CSV.
        sent_counts = token_counts(g_tags, p_tags)

        # Aggregate TP / FP / FN by BIO tag.
        for tag, c in sent_counts.items():
            for k in ("tp", "fp", "fn"):
                agg[tag][k] += c[k]

        # Token-level TN:
        # gold = O AND prediction = O
        sent_tn = sum(
            1
            for g, p in zip(g_tags, p_tags)
            if g == "O" and p == "O"
        )

        global_tn += sent_tn

    if mis_gold or mis_pred:
        log(
            f"note: {mis_gold} gold and {mis_pred} predicted spans could not be "
            f"aligned to token boundaries and were skipped"
        )

    # ---- NER metrics ------------------------------------------------------
    g_tp = sum(c["tp"] for c in agg.values())
    g_fp = sum(c["fp"] for c in agg.values())
    g_fn = sum(c["fn"] for c in agg.values())
    g_tn = global_tn

    gp, gr, gf = prf(g_tp, g_fp, g_fn)

    out_rows = [{
        "stage": "NER",
        "tag": "global",
        "precision": gp,
        "recall": gr,
        "f1": gf,
        "interpretable": "yes",
        "note": "token-level metrics over all sentences",
    }]

    for tag in sorted(agg):
        c = agg[tag]
        p, r_, f = prf(c["tp"], c["fp"], c["fn"])

        out_rows.append({
            "stage": "NER",
            "tag": tag,
            "precision": p,
            "recall": r_,
            "f1": f,
            "interpretable": "yes",
            "note": "",
        })

    pd.DataFrame(out_rows).to_csv(
        config.RESULTS_DIR / "exp1_addiction_metrics.csv",
        index=False
    )

    # ---- confusion matrix -------------------------------------------------
    conf = [{
        "stage": "NER",
        "tag": "global",
        "tp": g_tp,
        "fp": g_fp,
        "fn": g_fn,
        "tn": g_tn,
        "fpr": (
            g_fp / (g_fp + g_tn)
            if (g_fp + g_tn) > 0
            else 0.0
        ),
    }]

    for tag in sorted(agg):
        c = agg[tag]

        conf.append({
            "stage": "NER",
            "tag": tag,
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "tn": "",
            "fpr": "",
        })

    pd.DataFrame(conf).to_csv(
        config.RESULTS_DIR / "exp1_confusion.csv",
        index=False
    )

    # ---- class balance ----------------------------------------------------
    n_positive = sum(
        bool(r["gold_spans"])
        for r in rows
    )

    n_negative = len(rows) - n_positive

    pd.DataFrame([{
        "n_sentences": len(rows),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate": (
            n_positive / len(rows)
            if rows
            else 0.0
        ),
        "n_forwarded_to_ner": len(rows),
    }]).to_csv(
        config.RESULTS_DIR / "exp1_class_balance.csv",
        index=False
    )

    # ---- visual comparison CSV -------------------------------------------
    comparison_rows = []

    for r, pspans in zip(rows, pred_spans):

        # Gold BIO tags
        g_tags, _ = spans_to_bio(
            r["text"],
            r["gold_spans"],
            nlp,
            config.ADDICTION_LABEL
        )

        # Predicted BIO tags
        p_tags, _ = spans_to_bio(
            r["text"],
            pspans,
            nlp,
            config.ADDICTION_LABEL
        )

        # IMPORTANT:
        # Use exactly the same token-level logic as the confusion matrix.
        sent_counts = token_counts(g_tags, p_tags)

        sent_tp = sum(
            c["tp"]
            for c in sent_counts.values()
        )

        sent_fp = sum(
            c["fp"]
            for c in sent_counts.values()
        )

        sent_fn = sum(
            c["fn"]
            for c in sent_counts.values()
        )

        sent_tn = sum(
            1
            for g, p in zip(g_tags, p_tags)
            if g == "O" and p == "O"
        )

        # Determine sentence status.
        status = []

        if sent_tp > 0:
            status.append("TP")

        if sent_fp > 0:
            status.append("FP")

        if sent_fn > 0:
            status.append("FN")

        # If there are no TP/FP/FN, the sentence is completely TN.
        if not status:
            status.append("TN")

        status = "+".join(status)

        comparison_rows.append({
            "row_id": r["row_id"],
            "sent_idx": r["sent_idx"],
            "text": r["text"],

            # Original spans
            "gold_spans": str(r["gold_spans"]),
            "ner_spans": str(pspans),

            # Actual text corresponding to spans
            "gold_words": str(
                spans_to_text(
                    r["text"],
                    r["gold_spans"]
                )
            ),

            "ner_words": str(
                spans_to_text(
                    r["text"],
                    pspans
                )
            ),

            # Text with spans highlighted
            "gold_tagged_text": mark_spans(
                r["text"],
                r["gold_spans"]
            ),

            "ner_tagged_text": mark_spans(
                r["text"],
                pspans
            ),

            # Token-level counts
            "tp": sent_tp,
            "fp": sent_fp,
            "fn": sent_fn,
            "tn": sent_tn,

            # Sentence-level status
            "status": status,
        })

    pd.DataFrame(comparison_rows).to_csv(
        config.RESULTS_DIR / "exp1_ner_vs_gold.csv",
        index=False
    )

    log("wrote exp1_*.csv")

    return {
        "n_sentences": len(rows),
        "positive_rate": (
            n_positive / len(rows)
            if rows
            else 0.0
        ),
        "t_tag_s": t_ner,
    }

# ---------------------------------------------------------------------------
# Experiment 2: feasibility on MIMIC-IV-Note
# ---------------------------------------------------------------------------

def experiment_2(pipe):
    log("=== experiment 2: feasibility on MIMIC-IV-Note ===")

    rows = [json.loads(l) for l in open(config.WORK_DIR / "mimic4_sentences.jsonl")]
    texts = [r["text"] for r in rows]
    n_notes = len({r["note_id"] for r in rows})
    log(f"{len(rows)} sentences from {n_notes} notes")

    t0 = time.time()
    preds = []
    for batch in batched(texts, config.BATCH_SIZE):
        preds.extend(pipe.classify(batch))
    t_clf = time.time() - t0

    gate = [i for i, p in enumerate(preds) if p == 1]
    log(f"{len(gate)}/{len(rows)} sentences flagged by the classifier")

    t0 = time.time()
    cat_sent = Counter()      # sentences containing >=1 span of that category
    cat_span = Counter()      # total spans of that category
    note_flags = defaultdict(set)
    for batch_idx in batched(gate, config.BATCH_SIZE):
        out = pipe.tag([texts[i] for i in batch_idx])
        for i, spans in zip(batch_idx, out):
            cats = {lab for (_, _, lab) in spans}
            for lab in cats:
                cat_sent[lab] += 1
                note_flags[rows[i]["note_id"]].add(lab)
            for (_, _, lab) in spans:
                cat_span[lab] += 1
    t_ner = time.time() - t0

    # Side file consumed by experiment 3, so that the ICD check does not
    # require re-running inference. Contains note_ids only, stays in work/.
    with open(config.WORK_DIR / "mimic4_addiction_notes.txt", "w") as f:
        for nid, cats in note_flags.items():
            if config.ADDICTION_LABEL in cats:
                f.write(f"{nid}\n")

    out_rows = []
    for cat in config.PHI_CATEGORIES:
        n_notes_with = sum(1 for c in note_flags.values() if cat in c)
        out_rows.append({
            "category": cat,
            "sentences_flagged": cat_sent.get(cat, 0),
            "pct_of_sentences": 100 * cat_sent.get(cat, 0) / len(rows),
            "total_spans": cat_span.get(cat, 0),
            "notes_with_category": n_notes_with,
            "pct_of_notes": 100 * n_notes_with / n_notes,
        })
    pd.DataFrame(out_rows).to_csv(
        config.RESULTS_DIR / "exp2_feasibility.csv", index=False)

    total_s = t_clf + t_ner
    sps = len(rows) / total_s if total_s else 0
    sent_per_note = len(rows) / n_notes
    # Full MIMIC-IV-Note v2.2 discharge corpus size.
    FULL_CORPUS_NOTES = 331_794
    proj_h = (FULL_CORPUS_NOTES * sent_per_note / sps / 3600) if sps else 0

    pd.DataFrame([{
        "n_notes": n_notes,
        "n_sentences": len(rows),
        "sentences_per_note": sent_per_note,
        "classify_seconds": t_clf,
        "tag_seconds": t_ner,
        "total_seconds": total_s,
        "sentences_per_second": sps,
        "projected_hours_full_corpus": proj_h,
        "full_corpus_notes": FULL_CORPUS_NOTES,
        "hardware": "FILL IN: GPU/CPU model",
    }]).to_csv(config.RESULTS_DIR / "exp2_throughput.csv", index=False)

    log(f"throughput {sps:.1f} sent/s; projected {proj_h:.1f} h for full corpus")
    log("wrote exp2_*.csv  (remember to fill in the `hardware` field)")
    return {"n_notes": n_notes, "n_sentences": len(rows),
            "sentences_per_second": sps}


# ---------------------------------------------------------------------------
# Experiment 3: ICD consistency check (supports the fusion framing)
# ---------------------------------------------------------------------------

def experiment_3():
    log("=== experiment 3: ADDICTION flags vs. substance-use ICD codes ===")

    idx = pd.read_csv(config.WORK_DIR / "mimic4_note_index.csv")
    hadm_ids = set(idx["hadm_id"].dropna().astype(int))
    if not hadm_ids:
        log("no hadm_ids available; skipping")
        return None

    flagged_hadm = set()
    rows = [json.loads(l) for l in open(config.WORK_DIR / "mimic4_sentences.jsonl")]
    note_to_hadm = dict(zip(idx["note_id"], idx["hadm_id"]))

    # Recompute which notes carried an ADDICTION flag. Written by experiment_2
    # into this side file so we do not have to re-run inference.
    flag_file = config.WORK_DIR / "mimic4_addiction_notes.txt"
    if not flag_file.exists():
        log("mimic4_addiction_notes.txt not found; run experiment 2 first")
        return None
    for line in open(flag_file):
        nid = line.strip()
        h = note_to_hadm.get(nid)
        if pd.notna(h):
            flagged_hadm.add(int(h))

    icd_hadm = set()
    for chunk in pd.read_csv(config.MIMIC4_DIAGNOSES, chunksize=200000,
                             dtype={"icd_code": "string"}):
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        for r in chunk.itertuples():
            code = (r.icd_code or "").strip().upper()
            prefixes = (config.ICD9_SUBSTANCE_PREFIXES if r.icd_version == 9
                        else config.ICD10_SUBSTANCE_PREFIXES)
            if any(code.startswith(p) for p in prefixes):
                icd_hadm.add(int(r.hadm_id))

    both = len(flagged_hadm & icd_hadm)
    only_text = len(flagged_hadm - icd_hadm)
    only_icd = len(icd_hadm - flagged_hadm)
    neither = len(hadm_ids) - both - only_text - only_icd

    pd.DataFrame([{
        "n_admissions": len(hadm_ids),
        "flagged_by_pipeline": len(flagged_hadm),
        "coded_in_icd": len(icd_hadm),
        "both": both,
        "text_only": only_text,
        "icd_only": only_icd,
        "neither": neither,
        "jaccard": both / max(len(flagged_hadm | icd_hadm), 1),
    }]).to_csv(config.RESULTS_DIR / "exp3_icd_agreement.csv", index=False)

    log(f"both={both} text_only={only_text} icd_only={only_icd} neither={neither}")
    log("NOTE: disagreement is expected and is itself the finding. Text-only "
        "cases are substance mentions never coded structurally, which is "
        "precisely the content a fusion pipeline would miss if it relied on "
        "structured fields alone.")
    log("wrote exp3_icd_agreement.csv")
    return {"both": both, "text_only": only_text, "icd_only": only_icd}


# ---------------------------------------------------------------------------

def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nlp = get_tokenizer()
    pipe = Pipeline()

    m1 = experiment_1(pipe, nlp)
    # Use the following function if you want to try the only-NER performance
    # m1 = experiment_1_no_llm(pipe, nlp)
    m2 = experiment_2(pipe)
    m3 = experiment_3()

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mimic_iv_note_version": "2.2",
        "mimic_iv_version": "2.2",
        "mimic_iii_version": "1.4",
        "addiction_sbdh_categories": config.ADDICTION_SBDH_CATEGORIES,
        "experiment_1": m1,
        "experiment_2": m2,
        "experiment_3": m3,
    }
    with open(config.RESULTS_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log("all done. results/ is safe to publish; work/ is not.")


if __name__ == "__main__":
    main()
