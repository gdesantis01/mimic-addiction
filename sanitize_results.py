"""
Strip MIMIC-derived text from result files so the repository can be published
under the PhysioNet Data Use Agreement.

    python sanitize_results.py

The DUA prohibits redistribution of MIMIC content. Two files produced by the
harness contain it and must not be published in their raw form:

    exp1_ner_vs_gold.csv          full sentences from the notes
    exp1b_out_of_scope_preds.csv  predicted spans plus context windows

This script writes aggregate-only replacements that preserve everything needed
to reproduce the reported numbers, and nothing that reproduces the corpus.

    exp1b_adjudication_summary.csv   category counts, no text
    exp1_sentence_stats.csv          per-sentence outcome flags, no text

Delete the originals afterwards. If they were ever committed, removing them
from the working tree is not enough: rewrite the history with git filter-repo
before making the repository public.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path("results")


def log(m):
    print(f"[sanitize] {m}", flush=True)


def read_any(path):
    """These files have been produced with both ',' and ';' separators."""
    for sep in (",", ";"):
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError(f"could not parse {path}")


def h(x, n=10):
    return hashlib.sha256(str(x).encode()).hexdigest()[:n]


# ---------------------------------------------------------------- adjudication

def derive_fragment(series):
    """
    Recompute the single-token-fragment flag from the span text.

    The adjudication file carries a `fragment` column, but the raw harness
    output does not, and the flag is cheap to rederive: a fragment is a span
    that is a single token drawn from the set of category-bearing words the
    model emits in isolation.
    """
    BARE = {
        "abuse", "overdose", "dependence", "dependent", "dependency",
        "withdrawal", "intoxication", "use", "user", "addiction", "detox",
        "toxicity", "deficiency", "cessation", "ingestion",
    }
    norm = series.astype(str).str.strip().str.strip(".:,;()+-?").str.lower()
    return norm.isin(BARE)


def sanitize_adjudication(src, dst):
    df = read_any(src)
    log(f"reading {src.name}: {len(df)} rows, columns {list(df.columns)}")

    # Which column carries the judgement? Prefer the human one.
    col = None
    for cand in ("adjudicated", "correct_addiction_mention", "suggested"):
        if cand in df.columns and df[cand].notna().any() and \
           df[cand].astype(str).str.strip().ne("").any():
            col = cand
            break

    # The fragment flag: use the stored column if present, else rederive it.
    if "fragment" in df.columns:
        frag = df["fragment"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    elif "predicted_text" in df.columns:
        frag = derive_fragment(df["predicted_text"])
        log("no `fragment` column; rederived from span text")
    else:
        frag = pd.Series([False] * len(df))
        log("no `fragment` column and no `predicted_text`; fragment stats omitted")

    if col is None:
        log(f"WARNING: {src.name} carries no judgement column with data.")
        log("         This looks like the raw harness output rather than the")
        log("         adjudicated file. Emitting counts only, no categories.")
        out = pd.DataFrame([{
            "category": "UNADJUDICATED",
            "n": len(df),
            "pct": 100.0,
            "fragmentary": int(frag.sum()),
            "pct_fragmentary": round(100 * frag.mean(), 2) if len(df) else 0.0,
        }])
    else:
        log(f"using `{col}` as the judgement column")
        rows = []
        for cat, grp in df.groupby(df[col].astype(str).str.strip()):
            m = frag[grp.index]
            rows.append({
                "category": cat,
                "n": len(grp),
                "pct": round(100 * len(grp) / len(df), 2),
                "fragmentary": int(m.sum()),
                "pct_fragmentary": round(100 * m.mean(), 2),
            })
        out = pd.DataFrame(rows).sort_values("n", ascending=False)
        out.loc[len(out)] = {
            "category": "TOTAL", "n": len(df), "pct": 100.0,
            "fragmentary": int(frag.sum()),
            "pct_fragmentary": round(100 * frag.mean(), 2),
        }

    out.to_csv(dst, index=False)
    log(f"wrote {dst.name}: {len(df)} predictions reduced to {len(out)} rows, no text")

    if "predicted_text" in df.columns:
        lens = df["predicted_text"].astype(str).str.split().str.len()
        hist = lens.value_counts().sort_index().reset_index()
        hist.columns = ["tokens_in_span", "n_predictions"]
        hp = dst.parent / "exp1b_span_length_hist.csv"
        hist.to_csv(hp, index=False)
        log(f"wrote {hp.name}: span length distribution, no text")


# ---------------------------------------------------------------- per-sentence

def sanitize_sentence_level(src, dst):
    df = read_any(src)
    keep = pd.DataFrame()

    if "row_id" in df.columns:
        # Note identifiers let a reader reconstruct which notes were used.
        # Hash them: joins within the file still work, the corpus does not leak.
        keep["note"] = df["row_id"].map(h)
    if "sent_idx" in df.columns:
        keep["sent_idx"] = df["sent_idx"]

    def n_spans(v):
        try:
            import ast
            return len(ast.literal_eval(str(v)))
        except Exception:
            return 0

    if "gold_spans" in df.columns:
        keep["n_gold"] = df["gold_spans"].map(n_spans)
    if "ner_spans" in df.columns:
        keep["n_pred"] = df["ner_spans"].map(n_spans)
    if "status" in df.columns:
        keep["status"] = df["status"]
    keep["n_tokens"] = df["text"].astype(str).str.split().str.len()

    keep.to_csv(dst, index=False)
    log(f"wrote {dst.name}: {len(keep)} rows, counts and flags only, no text")


def main():
    if not RESULTS.exists():
        sys.exit("run this from the repository root (results/ not found)")

    # Prefer a file carrying human judgements over the raw harness output.
    adj = None
    patterns = [
        "*adjudicat*.csv",              # anything named adjudication/adjudicated
        "exp1b_out_of_scope_preds.csv",  # raw output, last resort
    ]
    for pat in patterns:
        hits = sorted(RESULTS.glob(pat))
        # also look one level up, in case the adjudicated file was kept outside results/
        hits += sorted(RESULTS.parent.glob(pat))
        hits = [h for h in hits if h.name != "exp1b_adjudication_summary.csv"]
        if hits:
            adj = hits[0]
            if len(hits) > 1:
                log(f"several candidates found: {[h.name for h in hits]}")
                log(f"using {adj.name}; pass a different one explicitly if wrong")
            break
    if adj:
        sanitize_adjudication(adj, RESULTS / "exp1b_adjudication_summary.csv")
    else:
        log("no adjudication file found; skipping")

    ner = RESULTS / "exp1_ner_vs_gold.csv"
    if ner.exists():
        sanitize_sentence_level(ner, RESULTS / "exp1_sentence_stats.csv")
    else:
        log("exp1_ner_vs_gold.csv not found; skipping")

    print()
    log("NOW DELETE THE ORIGINALS:")
    for p in (RESULTS / "exp1_ner_vs_gold.csv",
              RESULTS / "exp1b_out_of_scope_preds.csv"):
        if p.exists():
            print(f"    git rm --cached {p}   &&   rm {p}")
    print()
    log("If either was ever committed, `git rm --cached` does not remove it from")
    log("history. Run git filter-repo (or delete and recreate the repository)")
    log("before making it public.")


if __name__ == "__main__":
    main()
