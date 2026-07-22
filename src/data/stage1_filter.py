#!/usr/bin/env python3
"""
Stage 1: Quick Filter
100K → ~40K, CPU only, pure rule-based

Rules:
1. Text length < 800 chars → discard
2. Text length > 50,000 chars → long_queue
3. nemocurator_scores < 1.5 → discard
4. Noise density > 0.15 per 100 chars → discard
5. Math content density < 0.5 per 500 chars → discard
"""

import json
import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_FILE  = Path("/home/ubuntu/Midterm_Project/nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl")
OUTPUT_DIR  = Path("/home/ubuntu/Midterm_Project/stage1_output")
OUTPUT_DIR.mkdir(exist_ok=True)

NORMAL_FILE  = OUTPUT_DIR / "stage1_normal.jsonl"
LONG_FILE    = OUTPUT_DIR / "stage1_long.jsonl"
DISCARD_FILE = OUTPUT_DIR / "stage1_discarded.jsonl"   # 可选，用于调试

# ── Thresholds ─────────────────────────────────────────────────────────────
MIN_LEN             = 800
MAX_LEN             = 50_000
MIN_NEMO_SCORE      = 1.5
MAX_NOISE_DENSITY   = 0.15   # noise matches per 100 chars
MIN_MATH_DENSITY    = 0.5    # math indicators per 500 chars

# ── Noise patterns ─────────────────────────────────────────────────────────
NOISE_PATTERNS = re.compile(
    r"\[INS:[^\]]*\]"                   # CMS insert markers
    r"|\[DEL:[^\]]*\]"                  # CMS delete markers
    r"|\b(cookie|advertisement|navbar|menu|sidebar|footer|header)\b"  # web chrome
    r"|\d{1,2}:\d{2}\s*(am|pm|AM|PM)"  # timestamps
    r"|\bhttps?://\S+"                  # bare URLs
    r"|\b(click here|read more|subscribe|sign up|log in|register)\b"  # CTA phrases
    r"|\b(views?|likes?|shares?|comments?|followers?)\s*:\s*\d+"      # social stats
    , re.IGNORECASE
)

# ── Math content indicators ─────────────────────────────────────────────────
MATH_PATTERNS = re.compile(
    r"\$[^$]+\$"                        # inline LaTeX $...$
    r"|\$\$[\s\S]+?\$\$"               # display LaTeX $$...$$
    r"|\\[a-zA-Z]+"                     # LaTeX commands \frac \int etc.
    r"|\b(equation|theorem|proof|lemma|corollary|formula|integral|"
    r"derivative|matrix|vector|polynomial|function|variable|coefficient|"
    r"calculate|compute|solve|simplify|evaluate|differentiate|integrate)\b"
    r"|\b\d+\s*[\+\-\*\/\^=]\s*\d+"   # arithmetic expressions
    r"|\b(sin|cos|tan|log|ln|sqrt|lim|sum|prod)\b"  # math functions
    , re.IGNORECASE
)


def noise_density(text: str) -> float:
    """Returns noise matches per 100 characters."""
    if not text:
        return 0.0
    matches = len(NOISE_PATTERNS.findall(text))
    return matches / (len(text) / 100)


def math_density(text: str) -> float:
    """Returns math indicator matches per 500 characters."""
    if not text:
        return 0.0
    matches = len(MATH_PATTERNS.findall(text))
    return matches / (len(text) / 500)


def classify(record: dict) -> str:
    """
    Returns one of: 'normal', 'long', 'discard'
    Also attaches filter_reason to record for discarded items.
    """
    text  = record.get("text", "")
    meta  = record.get("metadata", {})
    nemo  = meta.get("nemocurator_scores", 0.0)
    tlen  = len(text)

    # Rule 1 & 2: length
    if tlen < MIN_LEN:
        record["filter_reason"] = f"too_short({tlen})"
        return "discard"

    # Rule 3: quality score
    if nemo < MIN_NEMO_SCORE:
        record["filter_reason"] = f"low_nemo({nemo:.2f})"
        return "discard"

    # Rule 4: noise density (check before expensive math density)
    nd = noise_density(text)
    if nd > MAX_NOISE_DENSITY:
        record["filter_reason"] = f"noise_dense({nd:.3f})"
        return "discard"

    # Rule 5: math density
    md = math_density(text)
    if md < MIN_MATH_DENSITY:
        record["filter_reason"] = f"low_math({md:.3f})"
        return "discard"

    # Rule 2: long queue (after quality checks pass)
    if tlen > MAX_LEN:
        return "long"

    return "normal"


def main():
    counters = {"total": 0, "normal": 0, "long": 0, "discard": 0}
    discard_reasons: dict[str, int] = {}

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as fin,
        open(NORMAL_FILE,  "w", encoding="utf-8") as f_normal,
        open(LONG_FILE,    "w", encoding="utf-8") as f_long,
        open(DISCARD_FILE, "w", encoding="utf-8") as f_discard,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counters["discard"] += 1
                continue

            counters["total"] += 1
            label = classify(record)
            counters[label] += 1

            if label == "normal":
                f_normal.write(json.dumps(record, ensure_ascii=False) + "\n")
            elif label == "long":
                f_long.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                reason = record.get("filter_reason", "unknown").split("(")[0]
                discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
                f_discard.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Progress
            if counters["total"] % 10_000 == 0:
                print(f"  Processed {counters['total']:,} / 100,000 ...", flush=True)

    # ── Summary ────────────────────────────────────────────────────────────
    total   = counters["total"]
    normal  = counters["normal"]
    long_q  = counters["long"]
    discard = counters["discard"]

    print("\n" + "=" * 50)
    print("Stage 1 Filter — Summary")
    print("=" * 50)
    print(f"Total input  : {total:>8,}")
    print(f"Normal queue : {normal:>8,}  ({normal/total*100:.1f}%)")
    print(f"Long queue   : {long_q:>8,}  ({long_q/total*100:.1f}%)")
    print(f"Discarded    : {discard:>8,}  ({discard/total*100:.1f}%)")
    print(f"Retained     : {normal+long_q:>8,}  ({(normal+long_q)/total*100:.1f}%)")
    print()
    print("Discard breakdown:")
    for reason, cnt in sorted(discard_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:<25} {cnt:>7,}  ({cnt/total*100:.1f}%)")
    print()
    print(f"Output files:")
    print(f"  Normal : {NORMAL_FILE}")
    print(f"  Long   : {LONG_FILE}")
    print(f"  Debug  : {DISCARD_FILE}")


if __name__ == "__main__":
    main()
