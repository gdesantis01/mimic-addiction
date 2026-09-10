"""
The NER predicts plausible spans but scores zero true positives. That means
gold spans and predictions are in different places: an offset alignment
problem, not a model problem.

This script resolves it by taking gold spans straight from
MIMIC-SBDH-keywords.csv and printing what text[start:end] actually returns.

    python diagnose_offsets.py

If the extracted strings are substance-use words, the offsets are correct and
the bug is downstream. If they are not, this script identifies the shift and
tells you which correction to apply.
"""

import sys

import pandas as pd

import config
from prepare_data import _resolve

csv_field_limit = sys.maxsize
try:
    import csv as _csv
    _csv.field_size_limit(csv_field_limit)
except Exception:
    pass

# Words we expect to find at a behavior_* keyword offset. Not exhaustive; used
# only to locate the true position when the offset is wrong.
EXPECTED = [
    "tobacco", "tob", "smok", "cigarette", "nicotine", "ppd", "pack",
    "alcohol", "etoh", "drink", "beer", "wine", "liquor", "audit",
    "drug", "cocaine", "heroin", "ivdu", "illicit", "substance", "marijuana",
    "denies", "quit", "social",
]

N_NOTES = 15


def load_gold():
    kw = pd.read_csv(config.SBDH_KEYWORDS_CSV)
    kw = kw[kw["sbdh"].isin(config.ADDICTION_SBDH_CATEGORIES)]
    spans = {}
    for row_id, grp in kw.groupby("row_id"):
        spans[int(row_id)] = sorted((int(r.start), int(r.end)) for r in grp.itertuples())
    return spans


def fetch_notes(wanted):
    got = {}
    for chunk in pd.read_csv(
        _resolve(config.MIMIC3_NOTEEVENTS),
        chunksize=20000,
        usecols=["ROW_ID", "TEXT"],
        dtype={"ROW_ID": "int64", "TEXT": "string"},
    ):
        hit = chunk[chunk["ROW_ID"].isin(wanted)]
        for r in hit.itertuples():
            got[int(r.ROW_ID)] = r.TEXT
        if len(got) >= len(wanted):
            break
    return got


def find_true_offset(text, s, e):
    """
    Search outward from the recorded offset for a plausible keyword.
    Returns (delta, matched_text) or (None, None).
    """
    width = e - s
    for radius in (0, 50, 200, 1000, 5000, 20000, len(text)):
        lo = max(0, s - radius)
        hi = min(len(text), e + radius)
        window = text[lo:hi].lower()
        for w in EXPECTED:
            k = window.find(w)
            while k != -1:
                cand = lo + k
                if abs(cand - s) <= radius:
                    return cand - s, text[cand:cand + width]
                k = window.find(w, k + 1)
    return None, None


gold = load_gold()
sample = sorted(gold)[:N_NOTES]
notes = fetch_notes(set(sample))
print(f"loaded {len(notes)} notes\n")

deltas = []
cr_counts = []

for row_id in sample:
    text = notes.get(row_id)
    if not isinstance(text, str):
        continue
    n_cr = text.count("\r")
    n_lf = text.count("\n")
    print(f"--- ROW_ID {row_id}  len={len(text)}  CR={n_cr}  LF={n_lf}")

    for (s, e) in gold[row_id][:4]:
        if e > len(text):
            print(f"    ({s},{e}) BEYOND END OF TEXT (len={len(text)})")
            continue
        got = text[s:e]
        ok = any(w in got.lower() for w in EXPECTED)
        mark = "OK  " if ok else "BAD "
        print(f"    {mark}({s},{e}) -> {got!r}")
        if not ok:
            delta, matched = find_true_offset(text, s, e)
            if delta is not None:
                print(f"          true text appears at delta {delta:+d}: {matched!r}")
                deltas.append(delta)
                # how many CR precede the corrected position
                cr_counts.append(text[:s + delta].count("\r"))
    print()

print("=" * 66)
if not deltas:
    print("No misalignment detected. Offsets resolve to plausible keywords, so")
    print("the bug is downstream: check project_spans_to_sentence() and the")
    print("sentence text stored in work/sbdh_sentences.jsonl.")
    raise SystemExit(0)

print(f"misaligned spans examined : {len(deltas)}")
print(f"delta range               : {min(deltas):+d} .. {max(deltas):+d}")

if all(d == deltas[0] for d in deltas):
    print(f"\nCONSTANT shift of {deltas[0]:+d} on every span.")
    print("Apply that constant correction when loading the keywords file.")
else:
    print("\nVARIABLE shift: it grows through each note, which is the signature")
    print("of a line-ending mismatch.")
    matches_cr = sum(1 for d, c in zip(deltas, cr_counts) if abs(abs(d) - c) <= 2)
    print(f"spans whose shift equals the preceding CR count: "
          f"{matches_cr}/{len(deltas)}")
    if matches_cr > len(deltas) * 0.7:
        print("\n>>> CAUSE CONFIRMED: your NOTEEVENTS text contains CRLF, while")
        print("    the MIMIC-SBDH offsets were computed on LF-only text. Every")
        print("    preceding line break shifts the offsets by one character.")
        print("\n    FIX: normalise the text on load, before any offset use:")
        print("         text = text.replace('\\r\\n', '\\n').replace('\\r', '\\n')")
        print("    Add this in prepare_data.stream_mimic3_notes() and re-run")
        print("    prepare_data.py. No re-training and no re-download needed.")
    else:
        print("\nShift is variable but does not track CR counts. Inspect the")
        print("printed cases above before applying any correction.")
