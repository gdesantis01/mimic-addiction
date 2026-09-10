"""
Configuration for the Clinical-MIRF MIMIC experiments.

Edit DATA_ROOT to point at the directory that contains `MIMIC-SBDH/` and
`physionet.org/`. Everything else is derived from it.
"""
import time
from pathlib import Path

# ---------------------------------------------------------------- paths

# The directory containing MIMIC-SBDH/ and physionet.org/
DATA_ROOT = Path("MIMIC")

MIMIC_SBDH_DIR = DATA_ROOT / "MIMIC-SBDH"
SBDH_STATUS_CSV = MIMIC_SBDH_DIR / "MIMIC-SBDH.csv"
SBDH_KEYWORDS_CSV = MIMIC_SBDH_DIR / "MIMIC-SBDH-keywords.csv"

PHYSIONET = DATA_ROOT / "physionet.org" / "files"

MIMIC3_NOTEEVENTS = PHYSIONET / "mimiciii" / "1.4" / "NOTEEVENTS.csv"
MIMIC4_DISCHARGE = PHYSIONET / "mimic-iv-note" / "2.2" / "note" / "discharge.csv"
MIMIC4_DIAGNOSES = PHYSIONET / "mimiciv" / "2.2" / "hosp" / "diagnoses_icd.csv"

# Working directory for intermediate artefacts. These contain MIMIC text and
# must NEVER be committed or published.
WORK_DIR = Path("./work")

# Directory for publishable result CSVs. These contain aggregate numbers only.
RESULTS_DIR = Path("./results_" , f"{time.time()}")

# ---------------------------------------------------------------- experiment

SEED = 42

# Experiment 2: how many MIMIC-IV-Note discharge summaries to process for the
# feasibility analysis. Start small (e.g. 200) to time a run, then scale up.
N_MIMIC4_NOTES = 2000 #1000

# Experiment 1: cap on how many MIMIC-SBDH notes to evaluate. None = all of them
# (MIMIC-SBDH covers 7,025 notes). Use a small number for a smoke test.
N_SBDH_NOTES = None #200 #None

# MIMIC-SBDH categories mapped onto our ADDICTION class.
ADDICTION_SBDH_CATEGORIES = ["behavior_alcohol", "behavior_tobacco", "behavior_drug"]

# The label your NER model emits for addiction spans. Change if yours differs.
ADDICTION_LABEL = "ADDICTION"

PHI_CATEGORIES = [
    "CLIN_COND",
    "MED_REP",
    "GENETIC",
    "DISABILITY",
    "FERTILITY",
    "ADDICTION",
]

# Batch size for calls into your pipeline.
BATCH_SIZE = 64

# ---------------------------------------------------------------- ICD codes

# Substance-use related ICD codes, stored dot-less as MIMIC does.
# Matching is by prefix against `icd_code`.
#
# ICD-9:  291 alcohol-induced mental disorders; 292 drug-induced;
#         303 alcohol dependence; 304 drug dependence; 305 alcohol/tobacco/drug abuse;
#         V1582 personal history of tobacco use.
# ICD-10: F10-F19 mental and behavioural disorders due to psychoactive substance use;
#         Z720 tobacco use; Z87891 personal history of nicotine dependence.
#
# NOTE: verify this list against your own reading of the coding manuals before
# publishing. It is a defensible starting point, not an authoritative mapping.
ICD9_SUBSTANCE_PREFIXES = ["291", "292", "303", "304", "305", "V1582"]
ICD10_SUBSTANCE_PREFIXES = [
    "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19",
    "Z720", "Z87891",
]
