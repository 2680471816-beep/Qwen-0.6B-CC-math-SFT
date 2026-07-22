#!/usr/bin/env python3
"""
Stage 2: Text Cleaning
Input : stage1_normal.jsonl + stage1_long_chunks.jsonl  (~91,331 records)
Output: stage2_clean.jsonl  (~88K expected)

Operations (in order):
  1. Delete navigation / UI chrome lines
  2. Delete CMS edit markers  ([INS:...:INS] keep content; [DEL:...:DEL] drop)
  3. Delete forum user-metadata lines (Joined/Posts/Thanks/…)
  4. Delete wrong-answer paragraphs (明确标注的错误解法段落)
  5. Normalize LaTeX delimiters  \\(…\\) → $…$  and  \\[…\\] → $$…$$
  6. Compress excess blank lines (3+ → 2)
  7. Post-clean length check (< 600 chars → discard)
"""

import json
import re
from pathlib import Path

# ── I/O paths ─────────────────────────────────────────────────────────────────
INPUT_FILES = [
    Path("/home/ubuntu/Midterm_Project/stage1_output/stage1_normal.jsonl"),
    Path("/home/ubuntu/Midterm_Project/stage1_output/stage1_long_chunks.jsonl"),
]
OUTPUT_DIR  = Path("/home/ubuntu/Midterm_Project/stage2_output")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE  = OUTPUT_DIR / "stage2_clean.jsonl"
DISCARD_FILE = OUTPUT_DIR / "stage2_discarded.jsonl"

MIN_POST_CLEAN_LEN = 600   # chars after cleaning


# ══════════════════════════════════════════════════════════════════════════════
# Rule 1 — Navigation / UI chrome  (line-level deletion)
# ══════════════════════════════════════════════════════════════════════════════
# Match entire lines that are purely navigation / social / page-control noise.
# We are conservative: only drop lines whose ENTIRE content is nav noise.
_NAV_LINE = re.compile(
    r'^[\-\*\s]*('
    r'Facebook|Twitter|LinkedIn|Pinterest|Instagram|YouTube'          # social icons
    r'|Share [Tt]his( [Cc]ourse)?'                                    # share buttons
    r'|Tweet'                                                          # tweet button
    r'|Home\s*[>»|/\\]'                                               # breadcrumb start
    r'|Skip to (content|main|navigation)'                             # a11y skip links
    r'|You are here\s*:'                                              # breadcrumb label
    r'|Page\s+\d+\s+of\s+\d+'                                        # pagination
    r'|Print\s*(this)?\s*(page|article)?'                             # print button
    r'|Subscribe\s*(now|today|free)?'                                 # subscribe CTA
    r'|Newsletter\s*Sign[\s\-]?[Uu]p'                                # newsletter
    r'|Follow [Uu]s\s*(on)?'                                         # follow button
    r'|Copyright\s*©?\s*\d{4}'                                       # copyright line
    r'|All [Rr]ights [Rr]eserved'                                    # rights line
    r'|Privacy [Pp]olicy|Terms of [Ss]ervice|Cookie [Pp]olicy'       # legal links
    r')[^\n]*$',
    re.MULTILINE
)

def remove_nav_lines(text: str) -> str:
    return _NAV_LINE.sub("", text)


# ══════════════════════════════════════════════════════════════════════════════
# Rule 2 — CMS edit markers
#   [INS: content :INS]  →  keep "content"
#   [DEL: content :DEL]  →  delete entirely
# ══════════════════════════════════════════════════════════════════════════════
_INS = re.compile(r'\[INS:\s*(.*?)\s*:INS\]', re.S)
_DEL = re.compile(r'\[DEL:.*?:DEL\]',          re.S)

# Fallback: bare [INS: or [DEL: without closing tag (malformed)
_INS_BARE = re.compile(r'\[INS:\s*')
_DEL_BARE = re.compile(r'\[DEL:[^\]]*\]')

def remove_cms_markers(text: str) -> str:
    text = _DEL.sub("", text)          # delete [DEL:...:DEL] entirely
    text = _INS.sub(r'\1', text)       # unwrap [INS:...:INS]
    text = _DEL_BARE.sub("", text)     # malformed [DEL:..] fallback
    text = _INS_BARE.sub("", text)     # malformed [INS: fallback
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Rule 3 — Forum user-metadata lines
#   Lines like:  "Joined: February 2009"  "Posts: 81"  "Thanks: 3,243"
#   Strategy: delete only isolated metadata lines, not surrounding reply content
# ══════════════════════════════════════════════════════════════════════════════
_FORUM_META_LINE = re.compile(
    r'^[ \t]*(Joined|Posts?|Thanks|Reputation|Location|Member since'
    r'|Likes? received|Trophy points?|Gender|Online\s+Status'
    r'|Last [Ss]een|Registered|Messages?)\s*:[ \t]*.*$',
    re.MULTILINE | re.IGNORECASE
)

def remove_forum_meta(text: str) -> str:
    return _FORUM_META_LINE.sub("", text)


# ══════════════════════════════════════════════════════════════════════════════
# Rule 4 — Wrong-answer paragraphs
#   Only delete paragraphs that contain an EXPLICIT wrong-answer label:
#     - Markdown header:  ## Incorrect Attempts / ### Wrong Solutions
#     - Bold label:       **WRONG**: ...  /  **Incorrect**: ...
#     - List item label:  1. WRONG: ...  /  - Incorrect answer: ...
#   We delete the entire paragraph (text between blank lines) containing the label.
#   Contextual uses like "where did I go wrong?" are NOT deleted.
# ══════════════════════════════════════════════════════════════════════════════
_WRONG_SECTION_HEADER = re.compile(
    r'^#{1,4}\s+(Incorrect [Aa]ttempts?|Wrong [Aa]nswers?|'
    r'[Ee]rroneous [Ss]olutions?|[Ff]alse [Ss]olutions?|'
    r'[Cc]ommon [Mm]istakes?)\s*$',
    re.MULTILINE
)
_WRONG_EXPLICIT_LABEL = re.compile(
    r'(?:^|\n)[ \t]*(?:\d+\.|[-*])\s+\*{0,2}WRONG\*{0,2}\s*:'  # "1. **WRONG**:"
    r'|(?:^|\n)[ \t]*\*{1,2}(?:WRONG|Incorrect|Wrong answer)\*{1,2}\s*:',  # "**WRONG**:"
    re.MULTILINE
)

def _delete_section_after(text: str, header_pat: re.Pattern) -> str:
    """Delete from a matched section header until the next same-level header."""
    parts = []
    last = 0
    for m in header_pat.finditer(text):
        # keep everything before this header
        parts.append(text[last:m.start()])
        # find the next heading of same or higher level to stop deletion
        level = len(re.match(r'#+', m.group()).group())
        stop_pat = re.compile(r'^#{1,' + str(level) + r'}\s', re.MULTILINE)
        next_h = stop_pat.search(text, m.end())
        last = next_h.start() if next_h else len(text)
    parts.append(text[last:])
    return "".join(parts)

def _delete_paragraphs_with(text: str, label_pat: re.Pattern) -> str:
    """Delete individual paragraphs (separated by blank lines) containing the label."""
    paragraphs = re.split(r'(\n{2,})', text)   # keep separators
    result = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        if label_pat.search(para):
            # skip this paragraph (and its following separator if any)
            i += 1
            if i < len(paragraphs) and re.fullmatch(r'\n{2,}', paragraphs[i]):
                i += 1
        else:
            result.append(para)
            i += 1
    return "".join(result)

def remove_wrong_answers(text: str) -> str:
    text = _delete_section_after(text, _WRONG_SECTION_HEADER)
    text = _delete_paragraphs_with(text, _WRONG_EXPLICIT_LABEL)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Rule 5 — Normalize LaTeX delimiters
#   \( ... \)  →  $ ... $
#   \[ ... \]  →  $$ ... $$
# ══════════════════════════════════════════════════════════════════════════════
# Non-greedy match; handle nested content safely by matching the paired delimiter.
_LATEX_INLINE  = re.compile(r'\\\((.+?)\\\)', re.S)
_LATEX_DISPLAY = re.compile(r'\\\[(.+?)\\\]', re.S)

def normalize_latex(text: str) -> str:
    text = _LATEX_DISPLAY.sub(r'$$\1$$', text)   # display first (longer match)
    text = _LATEX_INLINE.sub(r'$\1$',   text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Rule 6 — Compress excess blank lines  (3+ consecutive newlines → 2)
# ══════════════════════════════════════════════════════════════════════════════
_EXCESS_BLANK = re.compile(r'\n{3,}')

def compress_blank_lines(text: str) -> str:
    return _EXCESS_BLANK.sub('\n\n', text)


# ══════════════════════════════════════════════════════════════════════════════
# Master clean function
# ══════════════════════════════════════════════════════════════════════════════
def clean(text: str) -> str:
    text = remove_cms_markers(text)      # Rule 2 first (may expose nav lines)
    text = remove_nav_lines(text)        # Rule 1
    text = remove_forum_meta(text)       # Rule 3
    text = remove_wrong_answers(text)    # Rule 4
    text = normalize_latex(text)         # Rule 5
    text = compress_blank_lines(text)    # Rule 6
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    counters = {"total": 0, "kept": 0, "discarded": 0}
    # Track how much each rule actually removed
    rule_hits = {f"rule{i}": 0 for i in range(1, 7)}

    with (
        open(OUTPUT_FILE,  "w", encoding="utf-8") as f_out,
        open(DISCARD_FILE, "w", encoding="utf-8") as f_dis,
    ):
        for input_path in INPUT_FILES:
            print(f"Processing {input_path.name} ...")
            for line in open(input_path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                counters["total"] += 1
                original = record["text"]

                # Apply each rule and track hits
                t = original
                t1 = remove_cms_markers(t);        rule_hits["rule2"] += (t1 != t); t = t1
                t1 = remove_nav_lines(t);           rule_hits["rule1"] += (t1 != t); t = t1
                t1 = remove_forum_meta(t);          rule_hits["rule3"] += (t1 != t); t = t1
                t1 = remove_wrong_answers(t);       rule_hits["rule4"] += (t1 != t); t = t1
                t1 = normalize_latex(t);            rule_hits["rule5"] += (t1 != t); t = t1
                t1 = compress_blank_lines(t);       rule_hits["rule6"] += (t1 != t); t = t1
                cleaned = t.strip()

                # Rule 7: post-clean length check
                if len(cleaned) < MIN_POST_CLEAN_LEN:
                    record["filter_reason"] = f"post_clean_too_short({len(cleaned)})"
                    counters["discarded"] += 1
                    f_dis.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                record["text"] = cleaned
                counters["kept"] += 1
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if counters["total"] % 10_000 == 0:
                    print(f"  Processed {counters['total']:,} ...", flush=True)

    total    = counters["total"]
    kept     = counters["kept"]
    discard  = counters["discarded"]

    print(f"\n{'='*52}")
    print("Stage 2 Clean — Summary")
    print(f"{'='*52}")
    print(f"Total input    : {total:>8,}")
    print(f"Kept           : {kept:>8,}  ({kept/total*100:.1f}%)")
    print(f"Discarded      : {discard:>8,}  ({discard/total*100:.1f}%)")
    print()
    print("Rule activation (docs modified):")
    rule_desc = {
        "rule1": "Nav/UI chrome removal",
        "rule2": "CMS marker removal",
        "rule3": "Forum metadata removal",
        "rule4": "Wrong-answer paragraph removal",
        "rule5": "LaTeX delimiter normalization",
        "rule6": "Blank line compression",
    }
    for k, desc in rule_desc.items():
        n = rule_hits[k]
        print(f"  {desc:<38} {n:>7,}  ({n/total*100:.1f}%)")
    print(f"\nOutput : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
