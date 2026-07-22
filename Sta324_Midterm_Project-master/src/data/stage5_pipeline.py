#!/usr/bin/env python3
"""
Stage 5 Pipeline: Full rewash + reasoning-SFT generation.

Reads stage3 outputs, uses local vLLM (Qwen3-0.6B) to generate
high-quality reasoning-SFT data with <think>...</think> and \\boxed{}.

Usage:
    # Start vLLM first (in a separate terminal or tmux):
    #   python -m vllm.entrypoints.openai.api_server \\
    #       --model /home/ubuntu/models/Qwen3-0.6B-Base \\
    #       --served-model-name Qwen3-0.6B \\
    #       --port 8888 --max-model-len 9012 \\
    #       --gpu-memory-utilization 0.5 --dtype bfloat16

    # Small sample test (100 docs each):
    python src/data/stage5_pipeline.py --input A --limit 100
    python src/data/stage5_pipeline.py --input C --limit 100

    # Full run (A/B/D files → Route A):
    python src/data/stage5_pipeline.py --input A --workers 8
    python src/data/stage5_pipeline.py --input B --workers 8
    python src/data/stage5_pipeline.py --input D --workers 8

    # Full run (C file → Route B):
    python src/data/stage5_pipeline.py --input C --workers 8

    # Run all routes then merge:
    python src/data/stage5_pipeline.py --input all --workers 8

    # Resume interrupted run:
    python src/data/stage5_pipeline.py --input A --resume

    # Only merge/dedup existing route outputs:
    python src/data/stage5_pipeline.py --merge
"""

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent
STAGE3_DIR = BASE_DIR / "stage3_output"
OUT_DIR = BASE_DIR / "outputs" / "full_pipeline"

STAGE3_FILES: dict[str, Path] = {
    "A": STAGE3_DIR / "stage3_A.jsonl",
    "B": STAGE3_DIR / "stage3_B.jsonl",
    "C": STAGE3_DIR / "stage3_C.jsonl",
    "D": STAGE3_DIR / "stage3_D.jsonl",
}

# ── Global config (overridden by CLI args in main) ────────────────────────────
VLLM_BASE_URL = "http://localhost:8888/v1"
VLLM_MODEL = "Qwen3-0.6B"
TRAIN_SYSTEM = ""  # Must be empty string per Qwen3 recommendation

# Lazy-initialized OpenAI client
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")
    return _client


# ── Core LLM call ─────────────────────────────────────────────────────────────
def chat(prompt: str, max_tokens: int = 240, temp: float = 0.0) -> str:
    """Call vLLM with thinking disabled; strip residual <think> tags."""
    try:
        response = get_client().chat.completions.create(
            model=VLLM_MODEL,
            temperature=temp,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {
                    "role": "system",
                    "content": "只输出任务要求的内容，不要输出角色前缀、模板或解释性外壳。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return f"[CHAT_ERROR: {exc}]"
    # Strip any residual <think>...</think> block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("⚇", "").replace("⚼", "").strip()


# ── Text cleaning helpers ─────────────────────────────────────────────────────
def d_clean_text(text: str) -> str:
    """Light cleaning: normalize whitespace, strip URLs."""
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def d_title(text: str) -> str:
    """Extract first Markdown H1 title from document text."""
    match = re.search(r"^#\s+(.+)", text or "", re.MULTILINE)
    return d_clean_text(match.group(1)) if match else ""


def d_normalize(text: str) -> str:
    """Normalize text for dedup comparison (lowercase, strip math/punct)."""
    text = d_clean_text(text).lower()
    text = re.sub(r"\\boxed\{[^{}]*\}", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def d_tokens(text: str) -> set[str]:
    return set(d_normalize(text).split())


def d_jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def extract_boxed_answer(text: str) -> str:
    found = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    return found[-1] if found else ""


# ── Question extraction ───────────────────────────────────────────────────────
def d_clean_question(text: str, fallback: str = "") -> str:
    """Normalize a candidate question string."""
    raw = d_clean_text(text)
    matches = re.findall(r"[^?？\n]{8,220}[?？]", raw)
    question = matches[0].strip() if matches else raw.split("\n")[0].strip()
    question = re.sub(r"^[\-\d\.\)\s]+", "", question)
    question = re.sub(
        r"^(question|instruction)\s*[:：]\s*", "", question, flags=re.I
    )
    question = question.strip(" \"'`")
    if not question:
        question = fallback or "What is the main mathematical question in this source unit?"
    if question and question[-1] not in "?？":
        question = question.rstrip("。.:：") + "?"
    return question[:220]


def d_extract_explicit_question(source_unit: str, title: str = "") -> str:
    """Try to extract an explicit question from structured markdown."""
    text = source_unit or ""
    patterns = [
        r"### [^\n]*\n\n([^\n?]{8,220}\?)",
        r"\*\*Question:?\*\*\s*([^\n?]{8,220}\?)",
        r"\*\*Problem:?\*\*\s*([^\n?]{8,220}\?)",
        r"## Problem Statement.*?\n([^\n?]{8,220}\?)",
        r"^#\s+([^\n?]{8,220}\?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.S | re.MULTILINE)
        if match:
            return d_clean_question(match.group(1))
    # Fallback: scan lines for a standalone question
    # Skip lines that look like answers (contain numbers/formulas that are the result)
    _answer_like = re.compile(
        r"(=\s*[\d\.\-]+\s*$"              # ends with "= 42" or "= 3.14"
        r"|\\frac\{[^}]+\}\{[^}]+\}\s*$"  # ends with a fraction
        r"|^\$.*\$\s*$"                    # entire line is a LaTeX expression
        r"|is\s+\$[^$]+\$[\w\s\./,]*$"    # "is $X$ units" (broader than original)
        r"|cm\?$|kg\?$|m\?$|m/s\?$|s\?$|N\?$|J\?$"  # unit measurements
        r"|\\boxed\{"                       # line already contains the answer
        r"|is\s+-?[\d,]+(?:\.\d+)?\s*[.!]?\s*$"      # "is 133,320."
        r"|\*\*[\d][^*\n]{0,40}\*\*\s*[.!]?\s*$"     # ends with **value**
        r")",
        re.I,
    )
    for line in d_clean_text(text).split("\n"):
        line = line.strip()
        if line.endswith("?") and 12 <= len(line) <= 220:
            if not line.startswith(("**", "###", "##", "#")) and "Tags" not in line:
                if not _answer_like.search(line):
                    return d_clean_question(line)
    if title and title.strip().endswith("?"):
        return d_clean_question(title)
    return ""


def d_extract_problem_solution_unit(text: str, max_chars: int = 1400) -> str:
    """Extract a self-contained problem+solution block from a document."""
    patterns = [
        r"(## Problem Statement.*?## Solution.*?)(?=\n## |\Z)",
        r"(## Interview Question.*?## Answer.*?)(?=\n## |\Z)",
        r"(### Problem.*?## Solution.*?)(?=\n## |\Z)",
        r"(\*\*Question:\*\*.*?(?:\*\*Answer:\*\*|## Solution).*?)(?=\n## |\Z)",
        r"(## Example.*?## Solution.*?)(?=\n## |\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.S)
        if match:
            return d_clean_text(match.group(1)[:max_chars])
    cleaned = d_clean_text((text or "")[:max_chars])
    has_structure = any(
        tag in (text or "") for tag in ["Problem", "Question", "Solution", "Answer"]
    )
    if has_structure and "?" in cleaned:
        return cleaned
    return ""


# ── Concept doc detection (for C class) ──────────────────────────────────────
CONCEPT_TITLE_HINTS = [
    "lesson",
    "understanding",
    "how to",
    "guide",
    "ratio",
    "quart",
    "fraction",
    "equation",
    "graph",
    "trigonometric",
    "probability",
    "notation",
    "surface area",
    "distance between",
    "foi",
    "mixed numbers",
]

CONCEPT_BODY_HINTS = [
    "objective",
    "big idea",
    "definition",
    "properties",
    "vocabulary",
    "examples",
    "how to",
    "understanding",
    "lesson",
]


def d_looks_like_concept_doc(title: str, text: str) -> bool:
    """Return True if the document is a concept/tutorial article (→ Route B)."""
    title_l = (title or "").lower()
    head = (text or "")[:900].lower()
    positive = any(hint in title_l for hint in CONCEPT_TITLE_HINTS) or any(
        hint in head for hint in CONCEPT_BODY_HINTS
    )
    negative = (
        (
            title_l.endswith("?")
            and "how to" not in title_l
            and "what is" not in title_l
            and len(title_l) > 40
        )
        or "problem statement" in head
        or (
            "## solution" in head
            and "objective" not in head
            and "big idea" not in head
        )
    )
    return positive and not negative


def d_extract_concept_excerpt(text: str, max_chars: int = 900) -> str:
    """Extract key concept blocks (Objective, Definition, Example, etc.)."""
    blocks = [
        blk.strip()
        for blk in re.split(r"\n\s*\n", d_clean_text(text))
        if blk.strip()
    ]
    kept: list[str] = []
    markers = [
        "objective",
        "big idea",
        "definition",
        "example",
        "vocabulary",
        "properties",
        "strategy",
    ]
    for blk in blocks:
        if any(m in blk.lower() for m in markers):
            kept.append(blk)
        if len("\n\n".join(kept)) >= max_chars:
            break
    if not kept:
        kept = blocks[:3]
    return d_clean_text("\n\n".join(kept))[:max_chars]


# ── Answer / reasoning cleaning ───────────────────────────────────────────────
def d_clean_answer(text: str) -> str:
    """Clean LLM answer: strip leakage, normalize boxed format.

    If no \\boxed{} is found, attempt to rescue by extracting the last
    standalone numeric/expression result from the text.
    """
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("⚇", "").replace("⚼", "")
    # Truncate at next-question markers (model output bleed-through)
    text = re.sub(r"\n##\s+Q\b.*", "", text, flags=re.S)
    text = re.sub(r"\n#{1,3}\s+(Question|Problem|Exercise)\b.*", "", text, flags=re.S | re.I)
    # Remove prompt-leaked role lines
    text = re.sub(
        r"(?im)^(assistant|user|system|instruction|request|source unit|source excerpt|answer)\s*[:：].*$",
        "",
        text,
    )
    text = re.sub(r"(?i)(actual session|prompt:|response:)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Normalize \boxed{}: keep only last occurrence, append cleanly
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if matches:
        final_answer = matches[-1].strip()
        body = re.sub(r"\\boxed\{[^{}]*\}", "", text).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if body and body[-1] not in "。.!?":
            body += "。"
        return (body + "\n\n\\boxed{" + final_answer + "}").strip()
    # Rescue: try to extract the last math result from the final lines
    rescue = _rescue_boxed(text)
    if rescue:
        if text and text[-1] not in "。.!?":
            text += "。"
        return text + "\n\n\\boxed{" + rescue + "}"
    return text


def _rescue_boxed(text: str) -> str:
    """Try to extract a final answer from text that is missing \\boxed{}.

    Tries multiple patterns in order of specificity.
    Returns the extracted value string, or empty string on failure.
    """
    stripped = text.strip()
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    last_lines = lines[-6:]
    combined = " ".join(last_lines)
    last_line = last_lines[-1] if last_lines else ""

    # 1. Display math $$ ... $$ at end of text (last occurrence)
    m = re.search(r"\$\$\s*([\s\S]+?)\s*\$\$\s*[.。]?\s*$", stripped)
    if m:
        val = re.sub(r"\s+", " ", m.group(1)).strip()
        if val and len(val) < 120 and "\\begin{array}" not in val and "\\begin{align" not in val:
            return val

    # 2. "= <value>" at end of last sentence (e.g. "x = 5.")
    m = re.search(r"=\s*\$?([^$=\n]{1,60}?)\$?\s*[.。]?\s*$", combined)
    if m:
        val = m.group(1).strip().rstrip(".。 ")
        if val and len(val) < 50 and not val.startswith("="):
            return val

    # 3. Plain "is/are <number>" at end (e.g. "The sum is 133,320.")
    m = re.search(
        r"(?:is|are)\s+(-?[\d,]+(?:\.\d+)?(?:\s*/\s*[\d,]+)?)\s*[.。]?\s*$",
        combined, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".。,")
        if val and len(val) < 40:
            return val

    # 4. Bold **value** at end (e.g. "is **9.8 m/s**.")
    m = re.search(r"\*\*([^*\n]{1,60})\*\*\s*[.。]?\s*$", combined)
    if m:
        val = m.group(1).strip().rstrip(".。")
        if val and len(val) < 60:
            return val

    # 5. Last inline math $expr$ on its own line or at end of sentence
    m = re.search(r"\$([^$\n]{1,80})\$\s*[.。,]?\s*$", combined)
    if m:
        val = m.group(1).strip()
        if val and len(val) < 80 and len(val) > 1:
            return val

    # 6. "answer/result/value is/= <anything>"
    m = re.search(
        r"(?:answer|result|value)\s+(?:is|=)\s+\$?([^$\n.。]{1,50})",
        combined, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".。")
        if val and len(val) < 50:
            return val

    # 7. Last standalone number (integer or simple decimal/fraction) on last line
    m = re.search(r"(?<![a-zA-Z])(-?[\d,]+(?:\.\d+)?(?:\s*/\s*[\d,]+)?)\s*[.。]?\s*$", last_line)
    if m:
        val = m.group(1).strip().rstrip(".。,")
        if val and len(val) < 30 and len(val) > 0:
            return val

    return ""


def _add_boxed_via_llm(question: str, answer: str) -> str:
    """Targeted retry: ask the model to identify the final answer and wrap it in \\boxed{}.

    Only called when the first answer generation produced substantive text
    but no \\boxed{}. Returns the rewritten answer text.
    """
    prompt = (
        "下面是一道数学题和一个不含 \\boxed{} 的解答。\n"
        "请把最终答案用 \\boxed{} 格式补充到解答末尾（单独一行）。\n"
        "只输出完整解答（含末尾的 \\boxed{} 行），不要输出任何额外说明。\n\n"
        f"题目：{question}\n\n"
        f"解答：\n{answer}"
    )
    return chat(prompt, max_tokens=700, temp=0.0)


def d_clean_reasoning(text: str) -> str:
    """Clean reasoning text: strip think tags, leakage, boxed refs."""
    text = (text or "").strip()
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(
        r"(?im)^(assistant|user|system|instruction|request|answer|source unit|source excerpt)\s*[:：].*$",
        "",
        text,
    )
    text = re.sub(r"(?i)(actual session|prompt:|response:)", "", text)
    text = re.sub(r"\\boxed\{[^{}]*\}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def d_with_reasoning(think_text: str, answer: str) -> str:
    """Wrap reasoning in <think> block and prepend to answer."""
    return f"<think>{think_text}</think>\n{answer}"


# ── Quality control ───────────────────────────────────────────────────────────
def d_rule_flags(question: str, answer: str) -> list[str]:
    """Layer 1: zero-GPU rule checks on answer quality."""
    flags: list[str] = []
    answer_l = answer.lower()
    if "\\boxed{" not in answer:
        flags.append("missing_boxed")
    if any(
        token in answer_l
        for token in [
            "instruction",
            "actual session",
            "request:",
            "source unit",
            "source excerpt",
            "assistant:",
            "user:",
        ]
    ):
        flags.append("prompt_leak")
    if len(answer) < 50:
        flags.append("too_short")
    if answer.count("\n") < 1 and answer.count("。") + answer.count(".") < 1:
        flags.append("thin_reasoning")
    q_norm = d_normalize(question)
    a_norm = d_normalize(answer)
    # Only flag as copied if the answer has virtually no content beyond repeating the question.
    # If the answer contains \boxed{} it has a definite final answer, so it's legitimate even
    # if it restates the problem (common when the question is itself a factual statement like
    # "The result is X?" → answer: "The result is X.\n\n\boxed{X}").
    if q_norm and q_norm in a_norm and "\\boxed{" not in answer and len(a_norm) < len(q_norm) + 80:
        flags.append("question_copied")
    return flags


def d_reasoning_flags(text: str) -> list[str]:
    """Layer 3: checks on generated reasoning chain quality."""
    text_l = text.lower()
    flags: list[str] = []
    if len(text) < 180:
        flags.append("reasoning_too_short")
    if any(
        token in text_l
        for token in [
            "instruction",
            "actual session",
            "request:",
            "source unit",
            "source excerpt",
            "assistant:",
            "user:",
        ]
    ):
        flags.append("reasoning_prompt_leak")
    if text.count("\n") < 2 and text.count("。") + text.count(".") < 4:
        flags.append("reasoning_too_thin")
    if len(text) > 7000:
        flags.append("reasoning_too_long")
    return flags


def d_parse_quality(text: str) -> dict:
    """Parse LLM quality judge output into structured scores."""
    raw = d_clean_text(text)

    def grab_int(name: str, default: int = 1) -> int:
        match = re.search(rf"{name}\s*=\s*([1-5])", raw, re.I)
        return int(match.group(1)) if match else default

    keep_match = re.search(r"keep\s*=\s*(yes|no)", raw, re.I)
    reason_match = re.search(r"reason\s*=\s*(.+)", raw, re.I)
    return {
        "groundedness": grab_int("groundedness"),
        "reasoning": grab_int("reasoning"),
        "pedagogy": grab_int("pedagogy"),
        "format": grab_int("format"),
        "keep": bool(keep_match and keep_match.group(1).lower() == "yes"),
        "reason": reason_match.group(1).strip() if reason_match else raw[:200],
    }


def d_parse_instruction_list(text: str, limit: int = 2) -> list[str]:
    """Parse a numbered/bulleted list of questions from LLM output."""
    rows: list[str] = []
    for line in d_clean_text(text).split("\n"):
        line = re.sub(r"^[\-\d\.\)\s]+", "", line).strip()
        if len(line) < 10:
            continue
        if line[-1] not in "?？":
            line = line.rstrip("。.:：") + "?"
        if line not in rows:
            rows.append(line[:220])
        if len(rows) >= limit:
            break
    return rows


# ── Prompt functions ──────────────────────────────────────────────────────────
def _join(*lines: str) -> str:
    return "\n".join(lines)


def prompt_d_backtranslate_question(title: str, source_unit: str) -> str:
    """LLM #1 (Route A): recover the implicit student question from a doc."""
    return _join(
        "你正在把数学预训练原文重构成监督微调数据。",
        "给定一段已经轻清洗的数学原文，请恢复一个学生最自然会提出的问题。",
        "要求：",
        "1. 只输出一个问题，不要回答。",
        "2. 这个问题必须能由原文片段直接回答。",
        "3. 如果原文里本来就有题目，优先保留该题目的自然表述。",
        "4. 问题要具体，不能写成空泛概念句。",
        "5. 只输出问题句，以问号结尾。",
        "",
        "标题：",
        title,
        "",
        "原文片段：",
        source_unit,
    )


def prompt_d_answer_from_doc(question: str, source_unit: str) -> str:
    """LLM #2 (Route A): generate a grounded answer with \\boxed{}."""
    return _join(
        "你正在把数学预训练原文整理成高质量推理型 SFT 数据。",
        "请根据问题和原文片段，写出忠实、清晰、可教学的解答。",
        "要求：",
        "1. 只使用原文片段可支持的信息。",
        "2. 用 2 到 4 个自然步骤写出关键推理。",
        "3. 不要输出「根据原文」「文档里」等元话语。",
        r"4. 最后一行必须单独写成 \boxed{最终答案}。",
        "",
        "问题：",
        question,
        "",
        "原文片段：",
        source_unit,
    )


def prompt_d_self_instruct(
    knowledge_point: str, source_excerpt: str, count: int = 2
) -> str:
    """LLM #1 (Route B): generate student questions around a knowledge point."""
    return _join(
        "你正在围绕一个数学知识点构造学生问题，用于监督微调数据。",
        f"请生成 {count} 个不同但同知识点的学生问题。",
        "要求：",
        "1. 每行只写一个问题。",
        "2. 问题必须自包含，不能依赖外部上下文。",
        "3. 至少有一个问题偏概念解释，至少有一个问题偏简单应用。",
        "4. 不要写答案，不要写编号标题之外的解释。",
        "",
        "知识点：",
        knowledge_point,
        "",
        "参考摘录：",
        source_excerpt,
    )


def prompt_d_answer_from_knowledge(
    question: str, knowledge_point: str, source_excerpt: str
) -> str:
    """LLM #2 (Route B): answer a question grounded in the knowledge point."""
    return _join(
        "你正在把数学知识点目录扩展成高质量 reasoning-SFT。",
        "请根据给定知识点和参考摘录，回答学生问题。",
        "要求：",
        "1. 可以沿用摘录中的定义、公式或解法模板，但不要切换到无关知识。",
        "2. 回答要像老师在讲解，保留必要推理步骤。",
        r"3. 最后一行必须单独写成 \boxed{最终答案或最终结论}。",
        "",
        "知识点：",
        knowledge_point,
        "",
        "问题：",
        question,
        "",
        "参考摘录：",
        source_excerpt,
    )


def prompt_d_quality_judge(
    question: str, answer: str, source_unit: str
) -> str:
    """LLM #3: score answer quality on 4 dimensions."""
    return _join(
        "你正在做数学 SFT 数据的质量筛选。",
        "请根据问题、候选回答和原文片段，给出 4 项 1 到 5 分评分。",
        "评分项：groundedness, reasoning, pedagogy, format。",
        "如果回答出现 prompt 泄漏、格式错误或与原文明显不一致，则 keep=no。",
        "严格按下面格式输出：",
        "groundedness=<1-5>",
        "reasoning=<1-5>",
        "pedagogy=<1-5>",
        "format=<1-5>",
        "keep=<yes/no>",
        "reason=<一句话>",
        "",
        "问题：",
        question,
        "",
        "候选回答：",
        answer,
        "",
        "原文片段：",
        source_unit,
    )


def prompt_d_long_reasoning(question: str, source_unit: str, answer: str = "") -> str:
    """LLM #4: generate detailed reasoning chain (no <think> tags, no boxed)."""
    return _join(
        "你正在为数学监督微调数据补写高质量推理过程。",
        "请只根据题目和原文片段，写出详细但紧凑的推理草稿。",
        "要求：",
        "1. 只输出推理过程，不要输出最终答案，不要加 <think> 标签。",
        "2. 推理需要比普通解答更完整，尽量写出关键中间步骤或关键概念连接。",
        "3. 不要提到「原文片段」「题目要求我」等元话语。",
        "4. 不要写标题，不要写编号模板外壳。",
        "5. 对计算题尽量覆盖列式、化简、检查；对概念题尽量覆盖定义、关系、结论。",
        "6. 至少写出 8 句有效推理，必要时可更长。",
        "7. 目标长度明显长于普通答案。",
        "",
        "题目：",
        question,
        "",
        "原文片段：",
        source_unit,
    )


# ── SFT row builder ───────────────────────────────────────────────────────────
def _make_audit_row(
    route: str,
    source_id: str,
    question: str,
    answer: str,
    source_unit: str,
    quality: dict,
    reasoning_quality: Optional[dict] = None,
    **extra,
) -> dict:
    """Build a full audit record with messages + metadata."""
    row: dict = {
        "route": route,
        "source_id": source_id,
        "source_unit": source_unit,
        "messages": [
            {"role": "system", "content": TRAIN_SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "quality": quality,
    }
    if reasoning_quality is not None:
        row["reasoning_quality"] = reasoning_quality
    row.update(extra)
    return row


# ── Core item processors ──────────────────────────────────────────────────────
def _run_quality_judge(
    question: str, answer: str, source_unit: str, flags: list[str]
) -> tuple[dict, bool]:
    """Rule-only quality gate (LLM judge removed: 0.6B base model fails format).

    The LLM judge was causing ~39% false negatives because Qwen3-0.6B-Base
    cannot reliably output the `groundedness=X` format, so all scores defaulted
    to 1 and keep=False. Quality is now ensured by rule flags alone.
    """
    keep = len(flags) == 0
    judge = {
        "groundedness": 0,
        "reasoning": 0,
        "pedagogy": 0,
        "format": 0,
        "keep": keep,
        "reason": "rule-only gate (LLM judge disabled)",
    }
    return judge, keep


def _run_reasoning(
    question: str, source_unit: str, answer: str
) -> tuple[str, dict, bool]:
    """Generate reasoning chain and return (updated_answer, reasoning_quality, keep)."""
    r_prompt = prompt_d_long_reasoning(question, source_unit, answer)
    r_raw = chat(r_prompt, max_tokens=1500, temp=0.0)
    reasoning = d_clean_reasoning(r_raw)
    r_flags = d_reasoning_flags(reasoning)
    if r_flags:
        return answer, {"keep": False, "flags": r_flags, "think_length": len(reasoning)}, False
    new_answer = d_with_reasoning(reasoning, answer)
    return new_answer, {"keep": True, "flags": [], "think_length": len(reasoning)}, True


def process_route_a_item(item: dict) -> Optional[dict]:
    """
    Route A pipeline for one document (A/B/D class, or non-concept C).

    Steps:
        1. Extract source_unit (structured block or full text excerpt)
        2. Extract explicit question OR backtranslate via LLM
        3. Generate answer via LLM
        4. Rule filter (Layer 1)
        5. LLM quality judge (Layer 2)
        6. LLM reasoning generation (Layer 3)

    Returns:
        Audit record dict, or None if the document yields no usable content.
    """
    doc_id = item["id"]
    text = item["text"]
    title = d_title(text)

    # Step 1: Extract problem-solution unit
    source_unit = d_extract_problem_solution_unit(text)
    if not source_unit:
        # Fallback: first 1400 chars as source
        source_unit = d_clean_text(text[:1400])
    if len(source_unit) < 40:
        return None

    # Step 2: Question extraction or backtranslation
    question = d_extract_explicit_question(source_unit, title)
    if not question:
        q_prompt = prompt_d_backtranslate_question(title, source_unit)
        q_raw = chat(q_prompt, max_tokens=160, temp=0.1)
        question = d_clean_question(
            q_raw, fallback=title if title.endswith("?") else ""
        )
    # Reject questions that contain embedded answers (\\boxed{} or formula in question)
    if question and ("\\boxed{" in question or re.search(r"is\s+\$[^$]+\$[^?]*\?$", question)):
        question = ""
    if not question:
        return None

    # Step 3: Answer generation
    a_prompt = prompt_d_answer_from_doc(question, source_unit)
    a_raw = chat(a_prompt, max_tokens=600, temp=0.0)
    answer = d_clean_answer(a_raw)

    # Retry once if answer is substantive but still missing \boxed{}
    if "\\boxed{" not in answer and len(answer) > 60 and "question_copied" not in d_rule_flags(question, answer):
        retry_raw = _add_boxed_via_llm(question, answer)
        if retry_raw and "\\boxed{" in retry_raw:
            answer = d_clean_answer(retry_raw)

    # Step 4: Rule filter
    flags = d_rule_flags(question, answer)

    # Step 5: LLM quality judge
    judge, keep_quality = _run_quality_judge(question, answer, source_unit, flags)
    score = (
        judge["groundedness"]
        + judge["reasoning"]
        + judge["pedagogy"]
        + judge["format"]
        - 2 * len(flags)
    )
    quality = {"score": score, "keep": keep_quality, "rule_flags": flags, "judge": judge}

    # Step 6: Reasoning generation (only if quality passed)
    reasoning_quality: Optional[dict] = None
    if keep_quality:
        answer, reasoning_quality, r_keep = _run_reasoning(question, source_unit, answer)
        if not r_keep:
            quality["keep"] = False

    return _make_audit_row(
        route="A",
        source_id=doc_id,
        question=question,
        answer=answer,
        source_unit=source_unit,
        quality=quality,
        reasoning_quality=reasoning_quality,
        title=title,
    )


def process_route_b_item(item: dict) -> list[dict]:
    """
    Route B pipeline for one concept document (C class).

    Steps:
        1. Extract knowledge point (title) and concept excerpt
        2. Self-instruct: generate 2 student questions via LLM
        3. For each question: answer → rule filter → quality judge → reasoning

    Returns:
        List of audit records (0–2 per document).
    """
    doc_id = item["id"]
    text = item["text"]
    title = d_title(text)
    knowledge_point = title or "mathematical concept"

    # Step 1: Extract concept excerpt
    source_excerpt = d_extract_concept_excerpt(text)
    if not source_excerpt:
        source_excerpt = d_clean_text(text[:900])
    if len(source_excerpt) < 40:
        return []

    # Step 2: Self-instruct question generation
    si_prompt = prompt_d_self_instruct(knowledge_point, source_excerpt, count=2)
    si_raw = chat(si_prompt, max_tokens=180, temp=0.2)
    questions = d_parse_instruction_list(si_raw, limit=2)
    if not questions:
        questions = [
            f"What is the key idea behind {knowledge_point}?",
            f"How would you explain {knowledge_point} with one short worked example?",
        ]

    results: list[dict] = []
    for question in questions:
        q_hash = hashlib.sha256(question.encode()).hexdigest()[:8]
        source_id = f"{doc_id}::{q_hash}"

        # Step 3: Answer generation
        a_prompt = prompt_d_answer_from_knowledge(question, knowledge_point, source_excerpt)
        a_raw = chat(a_prompt, max_tokens=600, temp=0.0)
        answer = d_clean_answer(a_raw)

        # Retry once if answer is substantive but still missing \boxed{}
        if "\\boxed{" not in answer and len(answer) > 60 and "question_copied" not in d_rule_flags(question, answer):
            retry_raw = _add_boxed_via_llm(question, answer)
            if retry_raw and "\\boxed{" in retry_raw:
                answer = d_clean_answer(retry_raw)

        # Rule filter
        flags = d_rule_flags(question, answer)

        # Quality judge
        judge, keep_quality = _run_quality_judge(question, answer, source_excerpt, flags)
        score = (
            judge["groundedness"]
            + judge["reasoning"]
            + judge["pedagogy"]
            + judge["format"]
            - 2 * len(flags)
        )
        quality = {"score": score, "keep": keep_quality, "rule_flags": flags, "judge": judge}

        # Reasoning generation
        reasoning_quality: Optional[dict] = None
        if keep_quality:
            answer, reasoning_quality, r_keep = _run_reasoning(
                question, source_excerpt, answer
            )
            if not r_keep:
                quality["keep"] = False

        results.append(
            _make_audit_row(
                route="B",
                source_id=source_id,
                question=question,
                answer=answer,
                source_unit=source_excerpt,
                quality=quality,
                reasoning_quality=reasoning_quality,
                knowledge_point=knowledge_point,
            )
        )
    return results


# ── Dedup ─────────────────────────────────────────────────────────────────────
def d_dedup_rows(rows: list[dict], threshold: float = 0.82) -> list[dict]:
    """Deduplicate by exact q_norm match or Jaccard ≥ threshold."""
    kept: list[dict] = []
    for row in rows:
        question = row["messages"][1]["content"]
        answer = row["messages"][2]["content"]
        q_norm = d_normalize(question)
        signature = d_tokens(question + " " + extract_boxed_answer(answer))
        duplicated = any(
            q_norm == prev["_q_norm"]
            or d_jaccard(signature, prev["_sig"]) >= threshold
            for prev in kept
        )
        if not duplicated:
            row_copy = dict(row)
            row_copy["_q_norm"] = q_norm
            row_copy["_sig"] = signature
            kept.append(row_copy)

    cleaned: list[dict] = []
    for row in kept:
        row.pop("_q_norm", None)
        row.pop("_sig", None)
        cleaned.append(row)
    return cleaned


# ── I/O helpers ───────────────────────────────────────────────────────────────
def save_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_processed_ids(raw_path: Path, id_key: str = "source_id") -> set[str]:
    """Load already-processed IDs from an existing raw audit file."""
    if not raw_path.exists():
        return set()
    ids: set[str] = set()
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)[id_key])
            except Exception:
                pass
    return ids


def read_jsonl(path: Path, limit: Optional[int] = None):
    """Yield parsed records from a JSONL file, optionally limited."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            yield json.loads(line.strip())
            count += 1


# ── Route runners ─────────────────────────────────────────────────────────────
def run_route_a(
    input_labels: list[str],
    limit: Optional[int],
    workers: int,
    resume: bool,
) -> None:
    """Run Route A on A/B/D class files (problem-solution style).

    Each input label gets its own raw/sft file pair to avoid collisions
    when running A, B, D separately:
        route_a_A.raw.jsonl / route_a_A.sft_train.jsonl
        route_a_B.raw.jsonl / route_a_B.sft_train.jsonl
        route_a_D.raw.jsonl / route_a_D.sft_train.jsonl
    When multiple labels are passed (e.g. --input all), they share one file:
        route_a_ABD.raw.jsonl / route_a_ABD.sft_train.jsonl
    """
    tag = "".join(sorted(input_labels))  # e.g. "A", "B", "D", "ABD"
    raw_path = OUT_DIR / f"route_a_{tag}.raw.jsonl"

    processed_ids = load_processed_ids(raw_path) if resume else set()
    if resume and processed_ids:
        print(f"[Route A-{tag}] Resume: skipping {len(processed_ids)} already-processed docs")

    docs: list[dict] = []
    for label in input_labels:
        if label not in STAGE3_FILES:
            continue
        for item in read_jsonl(STAGE3_FILES[label]):
            if item["id"] not in processed_ids:
                docs.append(item)

    if limit is not None:
        docs = docs[:limit]

    total = len(docs)
    print(f"[Route A-{tag}] Processing {total} documents with {workers} workers")

    passed = 0
    processed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_route_a_item, doc): doc for doc in docs}
        for future in as_completed(futures):
            processed += 1
            try:
                result = future.result()
                if result is not None:
                    append_jsonl(raw_path, result)
                    q = result.get("quality", {})
                    rq = result.get("reasoning_quality") or {}
                    if q.get("keep") and rq.get("keep"):
                        passed += 1
            except Exception as exc:
                print(f"[Route A-{tag}] Error processing item: {exc}", file=sys.stderr)

            if processed % 200 == 0 or processed == total:
                pct = 100 * processed // total if total else 0
                print(
                    f"[Route A-{tag}] {processed}/{total} ({pct}%) done | {passed} passed quality gates"
                )

    print(f"[Route A-{tag}] Done: {passed}/{total} passed all quality gates → {raw_path}")

    # Write filtered SFT file (messages only, quality-passed rows)
    sft_path = OUT_DIR / f"route_a_{tag}.sft_train.jsonl"
    sft_rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("quality", {}).get("keep") and (r.get("reasoning_quality") or {}).get("keep"):
                    sft_rows.append({"messages": r["messages"]})
            except Exception:
                pass
    save_jsonl(sft_path, sft_rows)
    print(f"[Route A-{tag}] SFT file: {len(sft_rows)} rows → {sft_path}")


def run_route_b(
    limit: Optional[int],
    workers: int,
    resume: bool,
) -> None:
    """Run Route B on C class files (concept docs → knowledge point expansion)."""
    raw_b_path = OUT_DIR / "route_b.raw.jsonl"
    raw_c_as_a_path = OUT_DIR / "route_a_from_c.raw.jsonl"

    # For resume: track processed doc_ids (Route B source_ids have "doc_id::hash" format)
    processed_doc_ids: set[str] = set()
    if resume:
        for rpath in (raw_b_path, raw_c_as_a_path):
            for sid in load_processed_ids(rpath):
                processed_doc_ids.add(sid.split("::")[0])
        if processed_doc_ids:
            print(
                f"[Route B] Resume: skipping ~{len(processed_doc_ids)} already-processed doc_ids"
            )

    # Split C class into concept (Route B) and non-concept (Route A)
    route_b_docs: list[dict] = []
    route_a_docs: list[dict] = []
    for item in read_jsonl(STAGE3_FILES["C"]):
        if item["id"] in processed_doc_ids:
            continue
        title = d_title(item["text"])
        if d_looks_like_concept_doc(title, item["text"]):
            route_b_docs.append(item)
        else:
            route_a_docs.append(item)

    if limit is not None:
        route_b_docs = route_b_docs[:limit]
        route_a_docs = route_a_docs[: max(0, limit - len(route_b_docs))]

    print(
        f"[Route B] Concept docs (Route B): {len(route_b_docs)} | "
        f"Non-concept C docs (Route A): {len(route_a_docs)}"
    )

    total = len(route_b_docs) + len(route_a_docs)
    passed = 0
    processed = 0

    def _process_b(item: dict) -> tuple[str, list[dict]]:
        return "B", process_route_b_item(item)

    def _process_a(item: dict) -> tuple[str, list[dict]]:
        r = process_route_a_item(item)
        return "A", [r] if r else []

    all_tasks = [(_process_b, doc) for doc in route_b_docs] + [
        (_process_a, doc) for doc in route_a_docs
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fn, doc): (fn, doc) for fn, doc in all_tasks
        }
        for future in as_completed(futures):
            processed += 1
            try:
                label, results = future.result()
                for result in results:
                    if result is not None:
                        if label == "B":
                            append_jsonl(raw_b_path, result)
                        else:
                            append_jsonl(raw_c_as_a_path, result)
                        q = result.get("quality", {})
                        rq = result.get("reasoning_quality") or {}
                        if q.get("keep") and rq.get("keep"):
                            passed += 1
            except Exception as exc:
                print(f"[Route B] Error processing item: {exc}", file=sys.stderr)

            if processed % 200 == 0 or processed == total:
                pct = 100 * processed // total if total else 0
                print(
                    f"[Route B] {processed}/{total} ({pct}%) done | {passed} passed quality gates"
                )

    print(f"[Route B] Done: {passed} passed → {raw_b_path}, {raw_c_as_a_path}")

    # Write filtered SFT files (messages only, quality-passed rows)
    for raw_path, sft_name in [
        (raw_b_path, "route_b.sft_train.jsonl"),
        (raw_c_as_a_path, "route_a_from_c.sft_train.jsonl"),
    ]:
        if not raw_path.exists():
            continue
        sft_rows = []
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("quality", {}).get("keep") and (r.get("reasoning_quality") or {}).get("keep"):
                        sft_rows.append({"messages": r["messages"]})
                except Exception:
                    pass
        sft_path = OUT_DIR / sft_name
        save_jsonl(sft_path, sft_rows)
        print(f"[Route B] SFT file: {len(sft_rows)} rows → {sft_path}")


def merge_all() -> None:
    """Merge all route outputs, dedup, and save final training set."""
    print("[Merge] Loading raw audit files...")
    all_rows: list[dict] = []

    # Collect all route_a_*.raw.jsonl files (covers A, B, D, ABD, etc.)
    route_a_files = sorted(OUT_DIR.glob("route_a_*.raw.jsonl"))
    for fpath in route_a_files:
        count = 0
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    all_rows.append(json.loads(line))
                    count += 1
                except Exception:
                    pass
        print(f"[Merge] Loaded {count} rows from {fpath.name}")

    # Route B files (fixed names from run_route_b)
    for fname in ["route_a_from_c.raw.jsonl", "route_b.raw.jsonl"]:
        fpath = OUT_DIR / fname
        if not fpath.exists():
            print(f"[Merge] {fname} not found, skipping")
            continue
        count = 0
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    all_rows.append(json.loads(line))
                    count += 1
                except Exception:
                    pass
        print(f"[Merge] Loaded {count} rows from {fname}")

    print(f"[Merge] Total rows: {len(all_rows)}")

    # Filter to rows that passed all quality gates
    kept = [
        r
        for r in all_rows
        if r.get("quality", {}).get("keep")
        and (r.get("reasoning_quality") or {}).get("keep")
    ]
    print(f"[Merge] Rows passing all quality gates: {len(kept)}")

    # Sort by quality score descending before dedup
    kept.sort(key=lambda x: x.get("quality", {}).get("score", 0), reverse=True)

    # Dedup
    deduped = d_dedup_rows(kept, threshold=0.82)
    print(f"[Merge] After dedup (Jaccard threshold=0.82): {len(deduped)}")

    # Split by route for per-route sft files
    route_a_rows = [r for r in deduped if r.get("route") == "A"]
    route_b_rows = [r for r in deduped if r.get("route") == "B"]
    other_rows = [r for r in deduped if r.get("route") not in ("A", "B")]

    def messages_only(rows: list[dict]) -> list[dict]:
        return [{"messages": r["messages"]} for r in rows]

    save_jsonl(OUT_DIR / "route_a.sft_train.jsonl", messages_only(route_a_rows))
    save_jsonl(OUT_DIR / "route_b.sft_train.jsonl", messages_only(route_b_rows))
    final_sft = messages_only(deduped)
    save_jsonl(OUT_DIR / "final_reasoning_sft_train.jsonl", final_sft)

    print(
        f"[Merge] Saved:\n"
        f"  route_a.sft_train.jsonl       : {len(route_a_rows)}\n"
        f"  route_b.sft_train.jsonl       : {len(route_b_rows)}\n"
        f"  other                         : {len(other_rows)}\n"
        f"  final_reasoning_sft_train.jsonl: {len(final_sft)}"
    )

    if final_sft:
        print("\n[Merge] Sample SFT row:")
        sample = json.dumps(final_sft[0], ensure_ascii=False, indent=2)
        print(sample[:1000])


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: reasoning-SFT pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        choices=["A", "B", "C", "D", "all"],
        default="all",
        help=(
            "Which stage3 file(s) to process. "
            "A/B/D → Route A (problem-solution). "
            "C → Route B (concept expansion). "
            "all → run all routes then merge."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max docs to process (for testing); None = all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel LLM request threads (default: 4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip docs already present in the raw output file",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Skip processing; only merge/dedup existing route outputs",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8888/v1",
        help="vLLM OpenAI-compatible endpoint (default: http://localhost:8888/v1)",
    )
    args = parser.parse_args()

    # Update global vLLM URL and reset lazy client
    global VLLM_BASE_URL, _client
    VLLM_BASE_URL = args.vllm_url
    _client = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify vLLM is reachable (skip check for --merge-only)
    if not args.merge:
        try:
            resp = requests.get(f"{args.vllm_url}/models", timeout=5)
            resp.raise_for_status()
            models = [m["id"] for m in resp.json()["data"]]
            print(f"✓ vLLM connected at {args.vllm_url}, available models: {models}")
        except Exception as exc:
            print(f"✗ Cannot connect to vLLM at {args.vllm_url}: {exc}")
            print("\nPlease start vLLM first:")
            print(
                "  python -m vllm.entrypoints.openai.api_server \\\n"
                "    --model /home/ubuntu/models/Qwen3-0.6B-Base \\\n"
                "    --served-model-name Qwen3-0.6B \\\n"
                "    --port 8888 --max-model-len 9012 \\\n"
                "    --gpu-memory-utilization 0.5 --dtype bfloat16"
            )
            return

    if args.merge:
        merge_all()
        return

    # Run the appropriate route(s)
    if args.input in ("A", "B", "D"):
        run_route_a([args.input], args.limit, args.workers, args.resume)
    elif args.input == "C":
        run_route_b(args.limit, args.workers, args.resume)
    elif args.input == "all":
        run_route_a(["A", "B", "D"], args.limit, args.workers, args.resume)
        run_route_b(args.limit, args.workers, args.resume)
        merge_all()


if __name__ == "__main__":
    main()
