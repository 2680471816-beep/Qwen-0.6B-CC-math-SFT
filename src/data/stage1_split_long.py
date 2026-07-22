#!/usr/bin/env python3
"""
Long Document Splitter — Stage 1 supplement
Strategy: semantic splitting on Markdown headings
  1. Parse document into sections by H1/H2/H3/H4 headings
  2. Merge adjacent tiny sections (< MIN_CHUNK) upward
  3. Split oversized sections (> MAX_CHUNK) at paragraph / blank-line boundaries
  4. Re-apply math density check; drop chunk if fails
  5. Write chunks as new JSONL records with provenance metadata
"""

import json
import re
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE  = Path("/home/ubuntu/Midterm_Project/stage1_output/stage1_long.jsonl")
OUTPUT_FILE = Path("/home/ubuntu/Midterm_Project/stage1_output/stage1_long_chunks.jsonl")

MIN_CHUNK   = 800       # chars — same as Stage 1 lower bound
TARGET_CHUNK = 8_000    # chars — ideal chunk size
MAX_CHUNK   = 15_000    # chars — hard upper limit before force-split

MIN_MATH_DENSITY = 0.3  # slightly relaxed for chunks (vs 0.5 for whole docs)

# ── Math density check ───────────────────────────────────────────────────────
MATH_PAT = re.compile(
    r"\$[^$]+\$|\$\$[\s\S]+?\$\$|\\[a-zA-Z]+"
    r"|\b(equation|theorem|proof|lemma|corollary|formula|integral|derivative|"
    r"matrix|vector|polynomial|function|variable|coefficient|calculate|compute|"
    r"solve|simplify|evaluate|differentiate|integrate)\b"
    r"|\b\d+\s*[\+\-\*\/\^=]\s*\d+"
    r"|\b(sin|cos|tan|log|ln|sqrt|lim|sum|prod)\b", re.I
)

def math_density(text: str) -> float:
    if not text:
        return 0.0
    return len(MATH_PAT.findall(text)) / (len(text) / 500)


# ── Heading parser ───────────────────────────────────────────────────────────
HEADING_RE = re.compile(r'^(#{1,4})\s+(.+)', re.MULTILINE)

def split_by_headings(text: str) -> list[tuple[str, str]]:
    """
    Returns list of (heading_title, section_body) pairs.
    The first pair may have an empty heading if text starts before any heading.
    """
    sections = []
    matches = list(HEADING_RE.finditer(text))

    if not matches:
        return [("", text)]

    # Text before the first heading
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(0)
        start   = m.start()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body    = text[start:end].strip()
        sections.append((heading, body))

    return sections


# ── Paragraph-level force splitter ───────────────────────────────────────────
def force_split_by_paragraphs(text: str, max_size: int) -> list[str]:
    """
    When a single section exceeds max_size, split on blank lines,
    accumulating paragraphs until we approach max_size.
    """
    paragraphs = re.split(r'\n{2,}', text)
    chunks, current = [], []
    current_len = 0

    for para in paragraphs:
        plen = len(para) + 2  # +2 for the blank line
        if current_len + plen > max_size and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += plen

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


# ── Main splitter ─────────────────────────────────────────────────────────────
def split_document(record: dict) -> list[dict]:
    """
    Returns a list of chunk records derived from one long document.
    Each chunk inherits the original record's metadata plus chunk provenance.
    """
    text = record["text"]
    doc_id = record["id"]
    meta   = record["metadata"]

    # Step 1: split by headings
    raw_sections = split_by_headings(text)

    # Step 2: merge tiny sections into previous chunk
    merged: list[str] = []
    for _, body in raw_sections:
        if not body:
            continue
        if merged and len(body) < MIN_CHUNK:
            merged[-1] += "\n\n" + body
        else:
            merged.append(body)

    # Step 3: force-split any still-oversized sections
    final_pieces: list[str] = []
    for piece in merged:
        if len(piece) > MAX_CHUNK:
            final_pieces.extend(force_split_by_paragraphs(piece, TARGET_CHUNK))
        else:
            final_pieces.append(piece)

    # Step 4: build output records
    chunks = []
    for idx, piece in enumerate(final_pieces):
        if len(piece) < MIN_CHUNK:
            continue  # still too short after merging — discard
        if math_density(piece) < MIN_MATH_DENSITY:
            continue  # non-math section (table of contents, nav, etc.) — discard

        chunk_record = {
            "id": f"{doc_id}__chunk{idx:03d}",
            "text": piece,
            "metadata": {
                **meta,
                "source_doc_id": doc_id,
                "chunk_index":   idx,
                "chunk_total":   len(final_pieces),
                "chunk_chars":   len(piece),
            }
        }
        chunks.append(chunk_record)

    return chunks


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    records = [json.loads(l) for l in INPUT_FILE.open(encoding="utf-8") if l.strip()]
    print(f"Input: {len(records)} long documents")

    all_chunks = []
    skipped_docs = 0

    for r in records:
        chunks = split_document(r)
        if chunks:
            all_chunks.extend(chunks)
        else:
            skipped_docs += 1

    with OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for chunk in all_chunks:
            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── Stats ────────────────────────────────────────────────────────────────
    chunk_lens = [len(c["text"]) for c in all_chunks]
    avg_len = sum(chunk_lens) // len(chunk_lens) if chunk_lens else 0

    print(f"\n{'='*50}")
    print("Long Document Split — Summary")
    print(f"{'='*50}")
    print(f"Input docs          : {len(records):>6}")
    print(f"Docs with 0 chunks  : {skipped_docs:>6}  (all sections non-math)")
    print(f"Total chunks out    : {len(all_chunks):>6}")
    print(f"Avg chunks/doc      : {len(all_chunks)/len(records):.1f}")
    print(f"Avg chunk length    : {avg_len:>6,} chars")
    if chunk_lens:
        chunk_lens_s = sorted(chunk_lens)
        print(f"Chunk len p10/p50/p90: "
              f"{chunk_lens_s[len(chunk_lens_s)//10]:,} / "
              f"{chunk_lens_s[len(chunk_lens_s)//2]:,} / "
              f"{chunk_lens_s[int(len(chunk_lens_s)*0.9)]:,}")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
