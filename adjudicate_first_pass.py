"""
First-pass adjudication of out-of-scope NER predictions.

Produces a suggested label for every prediction in exp1b_out_of_scope_preds.csv
so the human adjudication becomes a review task rather than a from-scratch one.

    python adjudicate_first_pass.py <in.csv> <out.csv>

THE OUTPUT IS A SUGGESTION, NOT A JUDGEMENT. Every row must be reviewed before
any number derived from it is reported. The `suggested` column is what this
script thinks; the `adjudicated` column is empty and is for the human.

Categories
  YES         a genuine substance-use / addiction mention
  NO_CLIN     clinical or device dependence misread as substance dependence
              (insulin dependent, ventilator dependency, PEEP dependent)
  NO_MED      a medication name or therapeutic agent, not a substance of abuse
  NO_SELFHARM medication overdose in a self-harm or poisoning context: sensitive
              health data, but not the ADDICTION category as defined here
  CHECK       cannot be decided from the span alone; the context column decides

Span completeness is scored separately, because a prediction can identify the
right region and still be a fragment: the model was trained with BIO tags as
entity types and emits single tokens where a full span is expected.
"""

import re
import sys

import pandas as pd

SUBSTANCE = (
    r"alcohol|etoh|ethanol|liquor|beer|wine|drink|"
    r"tobacco|cigarette|smok|nicotine|"
    r"cocaine|heroin|opium|opiate|opioid|narcotic|methadone|"
    r"marijuana|cannabis|thc|amphetamine|benzo|"
    r"polysubstance|substance|illicit|ivdu|ivda|drug"
)

# Dependence that is clinical or device-related, not substance-related.
CLINICAL_DEP = (
    r"insulin|vent(ilator)?|peep|o2|oxygen|dialysis|hd\b|preload|"
    r"dopamine|pressor|transfusion|steroid|noninsulin"
)

# Therapeutic agents that are not substances of abuse in these contexts.
MEDICATION = (
    r"allopurinol|propofol|toradol|colchicine|corticosteroid|"
    r"warfarin|heparin|insulin"
)

SELFHARM = r"suicid|self-?harm|intentional|attempt"
OD_MED = r"tylenol|acetaminophen|motrin|ibuprofen|depakote|clonidine|percocet"

# Bare tokens that carry no category on their own.
BARE = {
    "abuse", "overdose", "dependence", "dependent", "dependency", "withdrawal",
    "intoxication", "use", "user", "addiction", "detox", "toxicity",
    "deficiency", "cessation",
}


def strip(s):
    return str(s).strip().strip(".:,;()+-?").lower()


def classify(pred, ctx):
    p, c = strip(pred), str(ctx).lower()

    # Decide from the span itself where possible.
    if re.search(CLINICAL_DEP, p):
        return "NO_CLIN", "clinical/device dependence in the span"
    if re.search(MEDICATION, p) and not re.search(SUBSTANCE, p):
        return "NO_MED", "therapeutic agent in the span"
    if re.search(OD_MED, p):
        return ("NO_SELFHARM" if re.search(SELFHARM, c) else "NO_MED",
                "medication overdose")
    if re.search(SUBSTANCE, p):
        return "YES", "substance term in the span"

    # Bare token: fall back to the surrounding context.
    if p in BARE or len(p.split()) <= 1:
        if re.search(CLINICAL_DEP, c):
            return "NO_CLIN", "bare token; clinical dependence in context"
        if re.search(OD_MED, c):
            return ("NO_SELFHARM" if re.search(SELFHARM, c) else "NO_MED",
                    "bare token; medication overdose in context")
        if re.search(MEDICATION, c) and not re.search(SUBSTANCE, c):
            return "NO_MED", "bare token; therapeutic agent in context"
        if re.search(SUBSTANCE, c):
            return "YES", "bare token; substance term in context"
        return "CHECK", "bare token, no decisive context"

    if re.search(SUBSTANCE, c):
        return "YES", "substance term in context"
    return "CHECK", "undecidable from span or context"


def is_fragment(pred):
    """True if the span looks like a single-token fragment of a longer entity."""
    p = strip(pred)
    return p in BARE or (len(p.split()) == 1 and p in BARE)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/exp1b_out_of_scope_preds.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "results/exp1b_adjudication.csv"

    df = pd.read_csv(src)
    rows = []
    for r in df.itertuples():
        cat, why = classify(r.predicted_text, r.context)
        rows.append({
            "predicted_text": r.predicted_text,
            "context": r.context,
            "suggested": cat,
            "reason": why,
            "fragment": is_fragment(r.predicted_text),
            "adjudicated": "",       # <- the human fills this in
        })

    out = pd.DataFrame(rows)
    out.to_csv(dst, index=False)

    n = len(out)
    print(f"{n} predictions\n")
    print("suggested category:")
    for cat, k in out.suggested.value_counts().items():
        print(f"  {cat:<12} {k:>4}  ({100*k/n:.1f}%)")
    print(f"\nfragmentary spans: {out.fragment.sum()} ({100*out.fragment.mean():.1f}%)")
    print(f"\nrequiring manual decision (CHECK): {(out.suggested=='CHECK').sum()}")
    print(f"\nwrote {dst}")
    print("\nReview every row. The `suggested` column is a heuristic first pass,")
    print("not an adjudication, and the CHECK rows have no suggestion at all.")


if __name__ == "__main__":
    main()
