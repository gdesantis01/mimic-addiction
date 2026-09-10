"""
Step 1 of 2. Builds the evaluation datasets.

Outputs (into WORK_DIR, which contains MIMIC text and must NEVER be published):

    work/sbdh_sentences.jsonl
        One row per sentence from the MIMIC-SBDH-annotated MIMIC-III notes,
        with gold ADDICTION character spans projected onto the sentence.

    work/mimic4_sentences.jsonl
        One row per sentence from a random sample of MIMIC-IV-Note discharge
        summaries, unlabeled.

    work/mimic4_note_index.csv
        note_id -> subject_id, hadm_id for the sampled notes, used later for
        the ICD consistency check.

Run:  python prepare_data.py
"""

import csv
import json
import random
import sys
import time
import os

import pandas as pd

import config
from pipeline_adapter import get_tokenizer

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = maxInt // 2


def log(msg):
    print(f"[prepare] {msg}", flush=True)


def _resolve(path):
    """
    Prefer the .gz original over a decompressed copy.

    pandas reads .csv.gz transparently, and skipping the decompression step
    removes a whole class of corruption (notably Windows text-mode extraction
    mangling the CRLF inside the multi-line quoted `text` field).
    """
    from pathlib import Path
    p = Path(path)
    gz = p.with_suffix(p.suffix + ".gz")
    if gz.exists():
        return gz
    if p.exists():
        return p
    raise FileNotFoundError(f"neither {p} nor {gz} exists")


# ---------------------------------------------------------------------------
# MIMIC-SBDH: gold ADDICTION spans
# ---------------------------------------------------------------------------

def load_gold_spans():
    """
    Returns {row_id: [(start_char, end_char), ...]} for addiction keywords,
    and the set of all row_ids covered by MIMIC-SBDH annotation.

    Notes present in MIMIC-SBDH.csv but with no addiction keyword row are
    TRUE NEGATIVES: the annotators reviewed them and found no such mention.
    Notes absent from MIMIC-SBDH.csv were never annotated and are excluded.
    """
    status = pd.read_csv(config.SBDH_STATUS_CSV)
    annotated_ids = set(status["row_id"].astype(int))
    log(f"MIMIC-SBDH covers {len(annotated_ids)} annotated notes")

    kw = pd.read_csv(config.SBDH_KEYWORDS_CSV)
    kw = kw[kw["sbdh"].isin(config.ADDICTION_SBDH_CATEGORIES)]
    log(f"{len(kw)} addiction keyword spans "
        f"across {kw['row_id'].nunique()} notes")

    spans = {}
    for row_id, grp in kw.groupby("row_id"):
        spans[int(row_id)] = sorted(
            (int(r.start), int(r.end)) for r in grp.itertuples()
        )
    return spans, annotated_ids


def stream_mimic3_notes(wanted_row_ids):
    """Yield (row_id, text) for the requested ROW_IDs, reading in chunks."""
    found = 0
    for chunk in pd.read_csv(
        _resolve(config.MIMIC3_NOTEEVENTS),
        chunksize=20000,
        usecols=["ROW_ID", "TEXT"],
        dtype={"ROW_ID": "int64", "TEXT": "string"},
        engine="c",
    ):
        hit = chunk[chunk["ROW_ID"].isin(wanted_row_ids)]
        for r in hit.itertuples():
            found += 1
            yield int(r.ROW_ID), r.TEXT
        if found >= len(wanted_row_ids):
            break
    log(f"retrieved {found}/{len(wanted_row_ids)} notes from NOTEEVENTS")


def project_spans_to_sentence(spans, sent_start, sent_end):
    """
    Keep gold spans fully contained in [sent_start, sent_end) and re-base
    their offsets to the start of the sentence.

    Spans straddling a sentence boundary are dropped and counted, so the
    number is reportable rather than silently ignored.
    """
    kept, straddling = [], 0
    for s, e in spans:
        if s >= sent_start and e <= sent_end:
            kept.append((s - sent_start, e - sent_start))
        elif s < sent_end and e > sent_start:
            straddling += 1
    return kept, straddling


def build_sbdh_dataset(nlp):
    gold, annotated_ids = load_gold_spans()

    row_ids = sorted(annotated_ids)
    if config.N_SBDH_NOTES:
        rng = random.Random(config.SEED)
        row_ids = sorted(rng.sample(row_ids, min(config.N_SBDH_NOTES, len(row_ids))))
        log(f"subsampled to {len(row_ids)} notes (N_SBDH_NOTES set)")
    wanted = set(row_ids)

    out_path = config.WORK_DIR / "sbdh_sentences.jsonl"
    n_sent = n_pos = n_notes = total_straddling = 0

    with open(out_path, "w") as fout:
        elem = 0
        now = time.time()
        for row_id, text in stream_mimic3_notes(wanted):
            elem+=1

            print(f"[DEBUG] Done element {elem}/{len(wanted)} from start: ", time.time() - now, " s")
            if not isinstance(text, str):
                continue
            n_notes += 1
            note_spans = gold.get(row_id, [])
            doc = nlp(text)
            for sent in doc.sents:
                s_txt = sent.text
                if not s_txt.strip():
                    continue
                kept, straddling = project_spans_to_sentence(
                    note_spans, sent.start_char, sent.end_char
                )
                total_straddling += straddling
                n_sent += 1
                if kept:
                    n_pos += 1
                fout.write(json.dumps({
                    "row_id": row_id,
                    "sent_idx": sent.start_char,
                    "text": s_txt,
                    "gold_spans": kept,
                }) + "\n")

    log(f"wrote {out_path}: {n_notes} notes, {n_sent} sentences, "
        f"{n_pos} positive ({100*n_pos/max(n_sent,1):.2f}%)")
    log(f"gold spans dropped for straddling a sentence boundary: {total_straddling}")
    log("  ^ report this number in the paper; it bounds the loss introduced "
        "by sentence segmentation")


# ---------------------------------------------------------------------------
# MIMIC-IV-Note: unlabeled feasibility sample
# ---------------------------------------------------------------------------



def build_mimic4_dataset(nlp):
    """
    Single pass with reservoir sampling.

    The previous two-pass version counted rows first and then re-read the file,
    which doubles the I/O on a multi-GB CSV and fails twice as often on a
    damaged one. Reservoir sampling gives a uniform sample of N notes in one
    pass without knowing the total in advance.
    """
    src = _resolve(config.MIMIC4_DISCHARGE)
    log(f"reading {src.name} (single pass, reservoir sampling)")

    rng = random.Random(config.SEED)
    k = config.N_MIMIC4_NOTES
    reservoir = []   # list of (note_id, subject_id, hadm_id, text)
    seen = 0

    reader = pd.read_csv(
        src,
        chunksize=2000,
        usecols=["note_id", "subject_id", "hadm_id", "text"],
        engine="c",
    )

    try:
        for chunk in reader:
            for r in chunk.itertuples():
                if not isinstance(r.text, str):
                    continue
                seen += 1
                item = (r.note_id, r.subject_id, r.hadm_id, r.text)
                if len(reservoir) < k:
                    reservoir.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < k:
                        reservoir[j] = item
            if seen % 50000 < 2000:
                log(f"  {seen} notes scanned, reservoir {len(reservoir)}/{k}")
    except Exception as e:
        log(f"ERROR while reading {src.name} after {seen} notes: {type(e).__name__}: {e}")
        log("run `python diagnose_discharge.py` before assuming the data is fine.")
        if len(reservoir) < k:
            log(f"only {len(reservoir)} notes collected; aborting rather than "
                f"silently producing a smaller, non-uniform sample")
            raise
        log("reservoir was already full; continuing with a sample drawn from "
            "the readable prefix only. THIS IS NOT A UNIFORM SAMPLE of the "
            "corpus and must be reported as such if you keep it.")

    log(f"scanned {seen} notes, sampled {len(reservoir)}")

    out_path = config.WORK_DIR / "mimic4_sentences.jsonl"
    idx_path = config.WORK_DIR / "mimic4_note_index.csv"

    n_sent = 0
    with open(out_path, "w") as fout, open(idx_path, "w", newline="") as fidx:
        widx = csv.writer(fidx)
        widx.writerow(["note_id", "subject_id", "hadm_id"])
        for note_id, subject_id, hadm_id, text in reservoir:
            widx.writerow([note_id, subject_id, hadm_id])
            for sent in nlp(text).sents:
                if not sent.text.strip():
                    continue
                n_sent += 1
                fout.write(json.dumps({
                    "note_id": note_id,
                    "sent_idx": sent.start_char,
                    "text": sent.text,
                }) + "\n")

    log(f"wrote {out_path}: {len(reservoir)} notes, {n_sent} sentences")
    log(f"total notes scanned this run: {seen} "
        f"(expected 331794 for MIMIC-IV-Note v2.2 -- a lower number means the "
        f"file is truncated)")


def main():
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nlp = get_tokenizer()
    nlp.max_length = 5_000_000

    log("=== building MIMIC-SBDH evaluation set ===")
    build_sbdh_dataset(nlp)

    log("=== building MIMIC-IV-Note feasibility sample ===")
    build_mimic4_dataset(nlp)

    log("done. work/ contains MIMIC text: do NOT publish it.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")

    os.system("shutdown.exe /h")
