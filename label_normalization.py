"""
Patch for pipeline_adapter.Pipeline.tag()

Your NER was trained with BIO tags as spaCy ENTITY TYPES rather than as the
internal encoding spaCy derives itself. As a result it emits single-token
entities labelled 'B-GENETIC', 'I-GENETIC', ... instead of one span labelled
'GENETIC'.

The correct fix is to rebuild the training data with bare category labels and
retrain: spaCy will then handle BIO internally and learn multi-token spans as
spans. Until then, this normalization recovers usable spans from the malformed
output, so that the evaluation measures what the model actually learned rather
than an artifact of the label scheme.

WHAT IT DOES
  1. strips the B-/I- prefix, mapping 'B-GENETIC' and 'I-GENETIC' -> 'GENETIC'
  2. merges adjacent or touching spans of the same category into one span

WHAT IT CANNOT DO
  It cannot recover entity boundaries the model never learned. Two genuinely
  distinct adjacent mentions of the same category will be merged into one.
  Report this as a limitation if you publish results obtained this way, and say
  plainly that the underlying model was trained on a malformed label scheme.

Replace the `tag` method in pipeline_adapter.py with the version below, and set
NORMALIZE_BIO_LABELS = False once you retrain with correct labels.
"""

NORMALIZE_BIO_LABELS = True

# Max gap in characters between two same-category spans for them to be merged.
# 1 covers a single separating space; raise it only with a reason.
MERGE_MAX_GAP = 1


def _strip_bio(label: str) -> str:
    if len(label) > 2 and label[:2] in ("B-", "I-"):
        return label[2:]
    return label


def _merge_spans(spans, max_gap=MERGE_MAX_GAP):
    """Merge adjacent same-category spans. `spans` is [(start, end, label), ...]."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[2], x[0]))
    out = []
    cur_s, cur_e, cur_l = spans[0]
    for s, e, l in spans[1:]:
        if l == cur_l and s - cur_e <= max_gap:
            cur_e = max(cur_e, e)
        else:
            out.append((cur_s, cur_e, cur_l))
            cur_s, cur_e, cur_l = s, e, l
    out.append((cur_s, cur_e, cur_l))
    return sorted(out, key=lambda x: x[0])


# --- drop this into pipeline_adapter.Pipeline, replacing the existing tag() ---

def tag(self, texts):
    """Span-level entity tagging. Character offsets relative to each input."""
    out = []
    for doc in self.ner.pipe(texts, batch_size=len(texts)):
        spans = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
        if NORMALIZE_BIO_LABELS:
            spans = _merge_spans([(s, e, _strip_bio(l)) for (s, e, l) in spans])
        out.append(spans)
    return out


# --- self-test ---------------------------------------------------------------

if __name__ == "__main__":
    # Reproduces the exact BRCA1 case from your diagnose_ner.py output.
    raw = [(27, 32, "B-GENETIC"), (33, 41, "I-GENETIC")]
    text = "Genetic testing revealed a BRCA1 mutation."
    got = _merge_spans([(s, e, _strip_bio(l)) for (s, e, l) in raw])
    print(f"raw   : {raw}")
    print(f"merged: {got}")
    print(f"text  : {[text[s:e] for (s, e, _) in got]}")
    assert got == [(27, 41, "GENETIC")], got

    # Non-adjacent same-category spans must stay separate.
    raw2 = [(0, 5, "B-CLIN_COND"), (40, 48, "B-CLIN_COND")]
    got2 = _merge_spans([(s, e, _strip_bio(l)) for (s, e, l) in raw2])
    assert len(got2) == 2, got2

    # Different categories must never merge.
    raw3 = [(0, 5, "B-CLIN_COND"), (6, 10, "B-GENETIC")]
    got3 = _merge_spans([(s, e, _strip_bio(l)) for (s, e, l) in raw3])
    assert len(got3) == 2, got3

    print("self-test passed")
