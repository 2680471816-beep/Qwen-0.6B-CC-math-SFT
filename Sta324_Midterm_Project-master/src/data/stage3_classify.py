#!/usr/bin/env python3
"""
Stage 3: Content Classification
Input : stage2_output/stage2_clean.jsonl  (91,331 records)
Output: stage3_output/stage3_{A,B,C,D,discard}.jsonl

Strategy (hybrid):
  Pass-1 — Rule-based (CPU, instant): catches high-confidence E/B/D cases
  Pass-2 — LLM batch inference (GPU): classifies remaining ambiguous records
  Pass-3 — Rule-based fallback: any LLM output of 'F'/'?' → re-examine and
            reclassify as C (article/tutorial) unless discard evidence exists

Category mapping:
  A → stage3_A.jsonl   (PROBLEM_SOLUTION)
  B → stage3_B.jsonl   (FORUM_QA)
  C → stage3_C.jsonl   (ARTICLE_TUTORIAL)
  D → stage3_D.jsonl   (TEXTBOOK)
  E → stage3_discard.jsonl  (CALCULATOR_TOOL  — discard)
  F → stage3_discard.jsonl  (OTHER            — discard)
"""

import json
import re
import time
import torch
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Paths ───────────────────────────────────────────────────────────────────
INPUT_FILE  = Path("/home/ubuntu/Midterm_Project/stage2_output/stage2_clean.jsonl")
OUTPUT_DIR  = Path("/home/ubuntu/Midterm_Project/stage3_output")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH  = "/home/ubuntu/models/Qwen3-0.6B"

OUT_FILES = {
    "A": OUTPUT_DIR / "stage3_A.jsonl",
    "B": OUTPUT_DIR / "stage3_B.jsonl",
    "C": OUTPUT_DIR / "stage3_C.jsonl",
    "D": OUTPUT_DIR / "stage3_D.jsonl",
    "discard": OUTPUT_DIR / "stage3_discard.jsonl",
}

# ── LLM config ──────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
SNIPPET_LEN  = 600    # chars fed to model
MAX_IN_TOKS  = 900    # max input tokens (after padding)
MAX_NEW_TOKS = 4      # we only need 1 letter


# ══════════════════════════════════════════════════════════════════════════════
# PASS-1 : Rule-based pre-classifier
# ══════════════════════════════════════════════════════════════════════════════

# ── E: Calculator / tool pages ───────────────────────────────────────────────
# These pages have extremely high density of repetitive arithmetic lines
_CALC_LINE = re.compile(
    r'^\s*[\d,]+\s*[\+\-\×\*\/÷=]\s*[\d,]+\s*=\s*[\d,\.]+\s*$', re.MULTILINE)
_UNIT_TABLE = re.compile(
    r'(\d+\.?\d*\s+(inch|cm|kg|lb|mile|km|°[CF]|gallon|liter|foot|feet|yard|'
    r'ounce|pound|meter|gram)\s*=\s*\d+\.?\d*\s+\w+\s*\n){3,}', re.I)

def is_calculator_page(text: str) -> bool:
    calc_lines = _CALC_LINE.findall(text)
    if len(calc_lines) >= 8:
        return True
    if _UNIT_TABLE.search(text):
        return True
    # High ratio of lines that are pure number-arithmetic
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False
    num_lines = sum(1 for l in lines if re.match(r'^[\d\s\+\-\*\/=\.,\(\)]+$', l))
    return len(lines) >= 10 and num_lines / len(lines) > 0.5

# ── B: Forum / QA pages ──────────────────────────────────────────────────────
_FORUM_STRONG = re.compile(
    r'(Posted by\s+\w|Reply\s*#\d|Quote\s*:|^\s*(OP|bump)\b'
    r'|Thread starter\s*:|Thread:\s+\w'
    r'|<\s*blockquote|wrote\s*:\s*\n'
    r'|#\d+\s+\d{1,2}[-/]\w{2,3}[-/]\d{2,4}'  # post numbering with date
    r'|\bMember\b.*\bPosts\b)',                  # leftover forum sidebar
    re.I | re.MULTILINE
)

def is_forum(text: str) -> bool:
    hits = _FORUM_STRONG.findall(text)
    return len(hits) >= 2

# ── D: Textbook pages ────────────────────────────────────────────────────────
_TEXTBOOK_NUMBERED_SECTION = re.compile(
    r'^\s*\d+\.\d+[\.\d]*\s+\w', re.MULTILINE)       # "1.1 Introduction", "2.3.1 ..."
_TEXTBOOK_FORMAL_BLOCK = re.compile(
    r'^(Definition|Theorem|Lemma|Corollary|Proposition|Proof|Exercise|Remark)'
    r'\s+\d+[\.\d]*', re.MULTILINE | re.IGNORECASE)   # "Theorem 2.1", "Definition 3"

def is_textbook(text: str) -> bool:
    numbered = _TEXTBOOK_NUMBERED_SECTION.findall(text)
    formal   = _TEXTBOOK_FORMAL_BLOCK.findall(text)
    # Require BOTH numbered sections AND formal blocks
    return len(numbered) >= 2 and len(formal) >= 2

# ── A: Problem+Solution (high-confidence only) ───────────────────────────────
# Only fire rule if there's an explicit structural label (header/bold)
_PROB_HEADER = re.compile(
    r'^#{1,4}\s*(Problem|Question|Exercise|Task|Challenge)\b'
    r'|^\*{1,2}(Problem|Question|Exercise)\*{1,2}\s*:',
    re.MULTILINE | re.IGNORECASE
)
_SOL_HEADER = re.compile(
    r'^#{1,4}\s*(Solution|Answer|Worked [Ss]olution|Step[- ]by[- ]Step)\b'
    r'|^\*{1,2}(Solution|Answer)\*{1,2}\s*:',
    re.MULTILINE | re.IGNORECASE
)

def is_problem_solution(text: str) -> bool:
    return bool(_PROB_HEADER.search(text)) and bool(_SOL_HEADER.search(text))

# ── Master rule classifier ────────────────────────────────────────────────────
def rule_classify(text: str):
    """
    Returns (label, confidence) or (None, None) if uncertain.
    confidence: 'high' means skip LLM; None means send to LLM.
    """
    if is_calculator_page(text):
        return "E", "high"
    if is_forum(text):
        return "B", "high"
    if is_textbook(text):
        return "D", "high"
    if is_problem_solution(text):
        return "A", "high"
    return None, None   # uncertain — send to LLM


# ══════════════════════════════════════════════════════════════════════════════
# PASS-2 : LLM batch classifier
# ══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_TEMPLATE = """\
Classify each math document into one category.
Categories: A=problem+solution  B=forum/discussion  C=tutorial/article  D=textbook  E=calculator-tool  F=other

Text: "## Problem 1\\nSolve the equation x^2 - 5x + 6 = 0.\\n## Solution\\nFactoring gives (x-2)(x-3)=0, so x=2 or x=3.\\n## Answer: x=2 or x=3"
Category: A

Text: "Thread: Help with derivatives\\nPosted by student99: I don't understand the chain rule.\\nReply #1 by mathpro: The chain rule states d/dx[f(g(x))] = f'(g(x))·g'(x). For example..."
Category: B

Text: "# Introduction to Limits\\n\\nA limit describes the value a function approaches as the input approaches some value.\\n\\n## Definition\\nWe say lim(x→a) f(x) = L if...\\n\\n## Example 1\\nFind lim(x→2)(x^2+1). Substituting x=2 gives 5."
Category: C

Text: "## Definition 2.1 (Vector Space)\\nA vector space over field F is a set V...\\n## Theorem 2.3\\nEvery vector space has a basis.\\n**Proof.** By Zorn's lemma...\\n## Exercises 2\\n1. Show that R^n is a vector space."
Category: D

Text: "1 foot = 12 inches\\n2 feet = 24 inches\\n3 feet = 36 inches\\n4 feet = 48 inches\\n5 feet = 60 inches"
Category: E

Text: "{snippet}"
Category:\
"""

def build_prompt(text: str) -> str:
    snippet = text[:SNIPPET_LEN].replace('"', "'").replace('\\n', ' ')
    return FEW_SHOT_TEMPLATE.replace("{snippet}", snippet)

def llm_classify_batch(model, tok, texts: list[str]) -> list[str]:
    prompts = [build_prompt(t) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True,
               truncation=True, max_length=MAX_IN_TOKS)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    labels = []
    for i, row in enumerate(out):
        new_toks = row[enc["input_ids"].shape[1]:]
        decoded  = tok.decode(new_toks, skip_special_tokens=True).strip()
        m = re.search(r'[A-F]', decoded)
        labels.append(m.group() if m else "F")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# PASS-3 : Post-LLM fallback
# Convert residual "F" labels using lightweight heuristics
# ══════════════════════════════════════════════════════════════════════════════
_WORD_PROBLEM = re.compile(
    r'\b(find|calculate|compute|solve|evaluate|determine|how many|how much|'
    r'what is|prove that|show that|if .{5,50} then|given that)\b',
    re.I
)
_DEFINITION_BLOCK = re.compile(
    r'\b(definition|theorem|lemma|corollary|proposition|proof|remark|note)\b',
    re.I
)

def fallback_classify(text: str, llm_label: str) -> str:
    """Reclassify 'F' outputs; keep A/B/C/D/E as-is."""
    if llm_label != "F":
        return llm_label
    # Re-run fast rules (they may now catch things LLM missed)
    rule_label, _ = rule_classify(text)
    if rule_label:
        return rule_label
    # Heuristic: if word-problem phrases present → A
    if _WORD_PROBLEM.search(text[:800]) and len(text) < 5000:
        return "A"
    # Heuristic: if definition/theorem language → C
    if _DEFINITION_BLOCK.search(text):
        return "C"
    # Default unclassifiable short/odd texts → discard (keep as F)
    return "F"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"Model on: {next(model.parameters()).device}\n")

    # Open all output files
    out_handles = {k: open(v, "w", encoding="utf-8") for k, v in OUT_FILES.items()}

    def write_record(record: dict, label: str):
        record["stage3_label"] = label
        key = label if label in ("A","B","C","D") else "discard"
        out_handles[key].write(json.dumps(record, ensure_ascii=False) + "\n")

    counters      = Counter()
    rule_counts   = Counter()
    llm_queue_ids = []     # indices for LLM batch
    llm_queue_recs = []
    all_records   = []

    # ── Pass 1: Load + rule classify ──────────────────────────────────────
    print("Pass 1: Rule-based classification...")
    t0 = time.time()
    with open(INPUT_FILE, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            all_records.append(record)

            label, conf = rule_classify(record["text"])
            if label is not None:
                rule_counts[label] += 1
                write_record(record, label)
                counters[label] += 1
            else:
                llm_queue_ids.append(len(all_records) - 1)
                llm_queue_recs.append(record)

    print(f"  Total loaded : {len(all_records):,}")
    print(f"  Rule-handled : {sum(rule_counts.values()):,}  {dict(rule_counts)}")
    print(f"  To LLM queue : {len(llm_queue_recs):,}")
    print(f"  Time: {time.time()-t0:.1f}s\n")

    # ── Pass 2: LLM batch classify ────────────────────────────────────────
    print("Pass 2: LLM batch classification...")
    t0 = time.time()
    llm_labels = []
    n = len(llm_queue_recs)
    for start in range(0, n, BATCH_SIZE):
        batch = llm_queue_recs[start:start + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        batch_labels = llm_classify_batch(model, tok, texts)
        llm_labels.extend(batch_labels)

        done = min(start + BATCH_SIZE, n)
        if done % 5000 < BATCH_SIZE or done == n:
            elapsed = time.time() - t0
            rate    = done / elapsed
            eta     = (n - done) / rate if rate > 0 else 0
            print(f"  LLM: {done:>6,}/{n:,}  "
                  f"speed={rate:.1f} rec/s  ETA={eta/60:.1f}min", flush=True)

    print(f"  LLM done in {time.time()-t0:.1f}s\n")

    # ── Pass 3: Fallback + write LLM results ─────────────────────────────
    print("Pass 3: Fallback reclassification + write...")
    llm_counter = Counter()
    for record, raw_label in zip(llm_queue_recs, llm_labels):
        final_label = fallback_classify(record["text"], raw_label)
        llm_counter[final_label] += 1
        write_record(record, final_label)
        counters[final_label] += 1

    for h in out_handles.values():
        h.close()

    # ── Summary ───────────────────────────────────────────────────────────
    total = sum(counters.values())
    label_desc = {
        "A": "PROBLEM_SOLUTION",
        "B": "FORUM_QA",
        "C": "ARTICLE_TUTORIAL",
        "D": "TEXTBOOK",
        "E": "CALCULATOR_TOOL (discarded)",
        "F": "OTHER (discarded)",
    }
    print(f"\n{'='*56}")
    print("Stage 3 Classification — Summary")
    print(f"{'='*56}")
    print(f"Total input  : {total:>8,}")
    print()
    print(f"{'Label':<6} {'Category':<30} {'Count':>7}  {'%':>5}")
    print("-"*56)
    for lbl in "ABCDEF":
        n = counters[lbl]
        if n:
            print(f"  {lbl}    {label_desc[lbl]:<30} {n:>7,}  {n/total*100:>5.1f}%")
    kept    = counters["A"] + counters["B"] + counters["C"] + counters["D"]
    discard = counters["E"] + counters["F"]
    print("-"*56)
    print(f"  Kept for Stage 4 : {kept:>7,}  ({kept/total*100:.1f}%)")
    print(f"  Discarded        : {discard:>7,}  ({discard/total*100:.1f}%)")
    print()
    print("Rule vs LLM breakdown:")
    print(f"  Rule-classified : {sum(rule_counts.values()):>7,}")
    print(f"  LLM-classified  : {sum(llm_counter.values()):>7,}  {dict(llm_counter)}")
    print()
    for lbl, path in OUT_FILES.items():
        size = sum(1 for _ in open(path))
        print(f"  {path.name:<35} {size:>7,} records")


if __name__ == "__main__":
    main()
