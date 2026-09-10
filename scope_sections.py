"""
Section-boundary tracking, replacing the per-sentence marker match in
evaluate_scoped.py.

THE BUG THIS FIXES
------------------
The previous implementation marked a sentence as in-scope only if that sentence
itself contained a social-history marker. Sections span many sentences, so
every sentence after the header was misclassified as out-of-scope. On the full
run this put 6,381 of 9,269 gold-bearing sentences (69%) outside the scope the
reference actually covers, making the scope-restricted metrics meaningless.

THE FIX
-------
Walk each note's sentences in document order, maintaining a state flag: enter
the social-history region on a section header, leave it on the next section
header of any other kind. This requires sentences grouped by note and ordered
by position, which is why the functions below take a note-grouped structure
rather than a flat list.

    python scope_sections.py        # runs the self-test
"""

import re

# Headers that open a social-history region.
SH_OPEN = re.compile(
    r"^\s{0,4}(social\s+(history|hx)|SH)\s*:", re.I | re.M)

# The compact field block that serves the same function when no header is present.
SH_FIELD = re.compile(
    r"^\s{0,4}(tobacco|tob|smoking|cigarettes?|etoh|alcohol|illicits?|"
    r"drugs?|substance\s+(use|abuse)|ivdu|ivda)\s*:",
    re.I | re.M)

# Any other section header, which closes the region.
SECTION_OPEN = re.compile(
    r"^\s{0,4}("
    r"family\s+history|physical\s+exam(ination)?|medications?(\s+on\s+\w+)?|"
    r"allergies|past\s+medical\s+history|pmh|review\s+of\s+systems|ros|"
    r"pertinent\s+results|brief\s+hospital\s+course|hospital\s+course|"
    r"discharge\s+(diagnosis|medications?|disposition|condition|instructions)|"
    r"admission\s+date|chief\s+complaint|"
    r"history\s+of\s+present\s+illness|hpi|"
    r"major\s+surgical|service|attending|addendum|"
    r"followup\s+instructions|facility"
    r")\s*:", re.I | re.M)


def annotate_scope(sentences):
    """
    Assign in-scope flags to one note's sentences.

    Args:
        sentences: list of sentence strings, in document order.

    Returns:
        list of bool, same length.

    A sentence is in scope if the social-history region is open when it is
    reached. The region opens on a social-history header or a substance field
    label, and closes on any other section header. The sentence carrying the
    opening header is itself in scope; the sentence carrying a closing header
    is not.
    """
    flags = []
    open_region = False
    for s in sentences:
        opens = bool(SH_OPEN.search(s)) or bool(SH_FIELD.search(s))
        closes = bool(SECTION_OPEN.search(s))

        if opens:
            # An opening marker wins over a closing one in the same sentence:
            # "Family History: ... Social History: ..." enters the region.
            open_region = True
            flags.append(True)
            continue

        if closes:
            open_region = False
            flags.append(False)
            continue

        flags.append(open_region)
    return flags


def group_by_note(rows, note_key="row_id", order_key="sent_idx"):
    """
    Group flat sentence rows into per-note ordered lists.

    Returns a list of (note_id, [row, ...]) with rows sorted by position.
    Rows lacking the ordering key keep their input order.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[r.get(note_key)].append((r.get(order_key, i), i, r))
    out = []
    for note_id, items in buckets.items():
        items.sort(key=lambda x: (x[0], x[1]))
        out.append((note_id, [it[2] for it in items]))
    return out


def apply_scope(rows, note_key="row_id", order_key="sent_idx"):
    """Set rows[i]['in_scope'] in place. Returns the same list."""
    for _, note_rows in group_by_note(rows, note_key, order_key):
        flags = annotate_scope([r["text"] for r in note_rows])
        for r, f in zip(note_rows, flags):
            r["in_scope"] = f
    return rows


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    note = [
        "Admission Date: [**2133-5-13**] Discharge Date: [**2133-5-15**]",
        "Chief Complaint: shortness of breath",
        "History of Present Illness: 62 yo M with COPD presenting with dyspnea.",
        "Social History:",
        "90 tobacco pack yr history, lives alone, drinks beer and liquor",
        "[**1-24**] drinks per day, on disability for the last 10 years",
        "Denies illicit drug use.",
        "Family History: Non-contributory",
        "Physical Exam: Vitals stable, lungs with wheezes",
        "Brief Hospital Course: Alcohol withdrawal managed with CIWA protocol.",
    ]
    flags = annotate_scope(note)
    for s, f in zip(note, flags):
        print(f"  {'IN ' if f else 'OUT'}  {s[:66]}")

    assert flags == [False, False, False, True, True, True, True,
                     False, False, False], flags

    # Compact field-block form, no explicit header.
    note2 = [
        "Past Medical History: HTN, DM2",
        "Tobacco: None",
        "ETOH: None",
        "Illicits: None",
        "Family History: Non-contributory",
    ]
    flags2 = annotate_scope(note2)
    for s, f in zip(note2, flags2):
        print(f"  {'IN ' if f else 'OUT'}  {s[:66]}")
    assert flags2 == [False, True, True, True, False], flags2

    print("\nself-test passed")
