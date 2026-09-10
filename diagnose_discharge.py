"""
Diagnose the discharge.csv parse failure before deciding to re-download.

    python diagnose_discharge.py

Checks, in order of decisiveness:
  1. SHA256 against PhysioNet's published checksum  (definitive)
  2. file size and line count
  3. whether the .gz original is present and readable
  4. whether the last line is complete
"""

import gzip
import hashlib
from pathlib import Path

import config


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            pct = 100 * done / size
            print(f"\r  hashing... {pct:5.1f}%", end="", flush=True)
    print()
    return h.hexdigest()


csv_path = Path(config.MIMIC4_DISCHARGE)
gz_path = csv_path.with_suffix(csv_path.suffix + ".gz")
sums_path = csv_path.parent / "SHA256SUMS_modified.txt"

print(f"target: {csv_path}")
print(f"exists: {csv_path.exists()}   size: "
      f"{csv_path.stat().st_size / 1e9:.2f} GB" if csv_path.exists() else "MISSING")
print(f".gz present: {gz_path.exists()}"
      + (f"   size: {gz_path.stat().st_size / 1e9:.2f} GB" if gz_path.exists() else ""))

# --- 1. checksum ------------------------------------------------------------
print("\n[1] checksum")
if sums_path.exists():
    published = {}
    for line in open(sums_path):
        parts = line.split()
        if len(parts) == 2:
            published[parts[1].lstrip("*")] = parts[0]
    print(f"  SHA256SUMS.txt lists: {list(published)}")

    for cand in (gz_path, csv_path):
        print("[DEBUG] cand: ", cand)
        print(f"[DEBUG] {cand.exists()}")
        print(f"[DEBUG] {cand.name in published}")
        if cand.exists() and cand.name in published:
            print(f"  hashing {cand.name} (this takes a few minutes)")
            actual = sha256(cand)
            if actual == published[cand.name]:
                print(f"  OK    {cand.name} matches the published checksum")
            else:
                print(f"  FAIL  {cand.name} does NOT match")
                print(f"        expected {published[cand.name]}")
                print(f"        actual   {actual}")
                print("        -> the file is corrupt or incomplete; re-download it")
else:
    print(f"  SHA256SUMS.txt not found at {sums_path}")
    print("  download it from the same PhysioNet directory to run this check")

# --- 2. line count and tail -------------------------------------------------
print("\n[2] structure")


def tail_and_count(opener, path, label):
    n = 0
    last = b""
    with opener(path, "rb") as f:
        for line in f:
            n += 1
            last = line
    print(f"  {label}: {n} physical lines")
    #print(f"  last line ends with newline: {last.endswith((b'\\n', b'\\r\\n'))}")
    print("  last line ends with newline:", last.endswith((b'\n', b'\r\n')))
    print(f"  last 120 bytes: {last[-120:]!r}")
    return n


if csv_path.exists():
    try:
        tail_and_count(open, csv_path, "decompressed csv")
    except Exception as e:
        print(f"  error reading csv: {e}")

if gz_path.exists():
    print("\n[3] .gz integrity")
    try:
        n = tail_and_count(gzip.open, gz_path, "gzip stream")
        print("  OK    gzip stream decompresses cleanly to the end")
        print("  -> read directly from the .gz and skip the decompressed copy")
    except EOFError:
        print("  FAIL  gzip stream is truncated; re-download")
    except Exception as e:
        print(f"  FAIL  gzip error: {e}")
else:
    print("\n[3] .gz not present -- keep it next time; reading it directly "
          "avoids any decompression step that could corrupt the file")

print("\ninterpretation:")
print("  checksum mismatch            -> re-download")
print("  checksum OK but csv fails    -> decompression corrupted it "
      "(Windows text-mode); read the .gz directly")
print("  .gz absent and csv truncated -> re-download the .gz, do not decompress")
