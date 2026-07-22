"""
Stage 4B: 将 stage3_B.jsonl (FORUM_QA 类) 转换为 SFT chat 训练格式。

B 类数据特点：来自数学论坛，包含提问帖 + 回复帖结构。
核心思路：提取第一帖（问题）和后续回复（解答），组成 Q&A 对。

输出格式（每行一条 JSON）:
{
    "id": "<原始id>",
    "messages": [
        {"role": "system",    "content": "<系统提示>"},
        {"role": "user",      "content": "<提问内容>"},
        {"role": "assistant", "content": "<解答内容>"}
    ],
    "source_id": "<原始id>",
    "extraction_strategy": "B1/B2/B3/B4"
}

提取策略：
  B1 : H3 标题为用户名风格 → 第一个 H3 节为问题，后续 H3 节合并为解答
  B2 : 数字加粗分隔（**1. ...#1**）→ 第1段为问题，后续段为解答
  B3 : 有明确 ## Problem/Question + 后续 ### 回复节
  B4 : 兜底 → 复用 A 类策略（S1/S2/S3/S5）
  SKIP: 无法提取，丢弃
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────── 路径配置 ───────────────────────────
ROOT        = Path("/home/ubuntu/Midterm_Project")
INPUT_PATH  = ROOT / "stage3_output" / "stage3_B.jsonl"
OUTPUT_PATH = ROOT / "stage4_output" / "stage4_B_sft.jsonl"
SKIP_PATH   = ROOT / "stage4_output" / "stage4_B_skipped.jsonl"

# ─────────────────────────── SFT 系统提示 ───────────────────────
SYSTEM_PROMPT = (
    "You are a helpful mathematics assistant. "
    "When given a math problem, provide a clear, step-by-step solution "
    "showing all reasoning and calculations."
)

# ─────────────────────────── 正则模式 ───────────────────────────

# B1：H3 标题为用户/帖子分隔符（### Username 或 ### Post by X）
_H3_SEP_RE = re.compile(r"^(###)\s+(.+)$", re.MULTILINE)

# B2：数字加粗分隔符（**1. Date #1** 或 **1.** 开头）
_BOLD_NUM_SEP_RE = re.compile(
    r"^\*\*\s*(\d+)\s*[\.\)]\s*.*?#\s*\1\s*\*\*|"   # **1. ... #1**
    r"^\*\*\s*(\d+)\s*[\.\)]\s*\*\*",                 # **1.**
    re.MULTILINE,
)

# 用户元数据行（帖子头部信息，需清除）
_META_LINE_RE = re.compile(
    r"^[\*_\s]*"
    r"(Junior Member|Senior Member|Super Member|Newbie|Forum (Ph\.D\.|Freshman|Junior|Senior)|"
    r"MHF (Contributor|Helper|Expert)|Administrator|Moderator|"
    r"Join(ed)?|Posts?:|Location:|From:|Awards?:|Thanks? (Received|Given):|"
    r"Status:|Reputation:|Online|Offline|Default:)"
    r".*$",
    re.MULTILINE | re.IGNORECASE,
)

# 引用块标记（> Quote Originally Posted by...）
_QUOTE_RE = re.compile(r"^>.*$", re.MULTILINE)

# 问题/解答明确标题（复用 A 类逻辑）
_PROB_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(problem\s*(statement)?|question|exercise)\s*[\d\.\:]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SOL_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(solution|answer|proof|workings?)\s*[\d\.\:]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# 数学内容检测
_MATH_RE = re.compile(r"\$[^$]+\$|\$\$[\s\S]+?\$\$|\\[a-zA-Z]+|\d+\s*[\+\-\*\/\^=]\s*\d+")

# 质量阈值
MIN_PROBLEM_LEN  = 30
MIN_SOLUTION_LEN = 80


# ──────────────────────── 工具函数 ──────────────────────────────

def _clean_post(text: str) -> str:
    """清除帖子中的用户元数据行和引用块，保留实质内容。"""
    text = _META_LINE_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_heading(line: str) -> str:
    return re.sub(r"^#{1,6}\s*", "", line).strip()


def _get_section_content(text: str, start_match: re.Match, level: int) -> str:
    """提取 start_match 之后、同级或更高级标题之前的内容。"""
    start = start_match.end()
    stop_pat = re.compile(rf"^(?:#{{1,{level}}})\s+", re.MULTILINE)
    m = stop_pat.search(text, start)
    end = m.start() if m else len(text)
    return text[start:end].strip()


def validate(problem: str, solution: str) -> bool:
    if len(problem) < MIN_PROBLEM_LEN:
        return False
    if len(solution) < MIN_SOLUTION_LEN:
        return False
    if not _MATH_RE.search(problem + solution):
        return False
    return True


# ──────────────────────── 提取策略 ──────────────────────────────

def extract_b1(text: str) -> Optional[tuple[str, str]]:
    """B1: H3 标题为用户名/帖子分隔 → 第一节为问题，其余节合并为解答。
    适用于 ### Username 或 ### Post by X 格式的论坛帖。
    """
    sections = list(_H3_SEP_RE.finditer(text))
    if len(sections) < 2:
        return None

    # 检验是否为"用户名/帖子"风格的分隔符
    first_title = sections[0].group(2).strip()
    if len(first_title) > 80:
        return None  # 太长，可能是普通小节标题

    # 提取第一节（提问者内容）
    q_start = sections[0].end()
    q_end   = sections[1].start()
    question_raw = text[q_start:q_end].strip()
    question = _clean_post(question_raw)

    # 若第一节太短（可能是元数据节），尝试用 H3 之前的 preamble
    preamble = _clean_post(text[:sections[0].start()].strip())
    if len(question) < MIN_PROBLEM_LEN and len(preamble) >= MIN_PROBLEM_LEN:
        question = preamble
    elif len(question) < MIN_PROBLEM_LEN and preamble:
        # 把 preamble + 第一节合并作为问题
        question = (preamble + "\n\n" + question).strip()

    # 合并后续所有回复节为解答
    answer_parts = []
    for i in range(1, len(sections)):
        a_start = sections[i].end()
        a_end   = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        part = _clean_post(text[a_start:a_end].strip())
        if part:
            answer_parts.append(part)

    solution = "\n\n".join(answer_parts)

    if question and solution:
        return question, solution
    return None


def extract_b2(text: str) -> Optional[tuple[str, str]]:
    """B2: 数字加粗分隔符（**1. ... #1** 格式）→ 第1帖为问题，后续为解答。"""
    matches = list(_BOLD_NUM_SEP_RE.finditer(text))
    if len(matches) < 2:
        return None

    # 第一帖内容（从第1个分隔符到第2个）
    q_start = matches[0].end()
    q_end   = matches[1].start()
    question = _clean_post(text[q_start:q_end].strip())

    # 后续帖内容合并为解答
    answer_parts = []
    for i in range(1, len(matches)):
        a_start = matches[i].end()
        a_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        part = _clean_post(text[a_start:a_end].strip())
        if part:
            answer_parts.append(part)
    solution = "\n\n".join(answer_parts)

    if question and solution:
        return question, solution
    return None


def extract_b3(text: str) -> Optional[tuple[str, str]]:
    """B3: 有明确 ## Problem/Question 节，后续 ## 或 ### 节为解答。"""
    pm = _PROB_HEADER_RE.search(text)
    if not pm:
        return None
    prob_level = len(pm.group(1))

    # 问题 = Problem 节内容
    question = _get_section_content(text, pm, prob_level)

    # 解答 = Problem 节之后的全部内容
    # 若有明确 Solution 节，优先用它；否则取全部后续内容
    sm = _SOL_HEADER_RE.search(text, pm.end())
    if sm:
        sol_level = len(sm.group(1))
        solution = _get_section_content(text, sm, sol_level)
    else:
        stop_pat = re.compile(rf"^(?:#{{1,{prob_level}}})\s+", re.MULTILINE)
        m = stop_pat.search(text, pm.end())
        sol_start = m.start() if m else pm.end() + len(question)
        solution = _clean_post(text[sol_start:].strip())

    if question and solution:
        return question, solution
    return None


def extract_b4(text: str) -> Optional[tuple[str, str]]:
    """B4: 兜底 — 文档整体结构类似 A 类，用 H1/H2 标题作问题。"""
    # 尝试 H1/H2 标题作问题（等同于 A 类的 S3）
    m = re.search(r"^(#{1,2})\s+(.+)$", text, re.MULTILINE)
    if not m or m.start() > 300:
        return None
    title = m.group(2).strip()

    # 过滤纯导航/目录标题
    if re.search(r"\b(Thread|Discussion|Forum|Pre-University|Math Help)\b", title, re.I):
        # 尝试往下找第一个 H3 或正文段落作为真正的问题
        after = text[m.end():].strip()
        paras = [p.strip() for p in re.split(r"\n{2,}", after) if p.strip()]
        if not paras:
            return None
        first_para = _clean_post(paras[0])
        solution = _clean_post("\n\n".join(paras[1:]))
        # 第一段须含数学或问题关键词
        if not (_MATH_RE.search(first_para) or
                re.search(r"\b(find|prove|show|calculate|solve|determine|if|given)\b",
                          first_para, re.I)):
            return None
        return first_para, solution

    # 标题即问题，后续内容为解答
    solution = _clean_post(text[m.end():].strip())
    if title and solution:
        return title, solution
    return None


def extract_b5(text: str) -> Optional[tuple[str, str]]:
    """B5: **Question** / **Answer** 加粗标注，或 Posted by 单帖连续格式。"""
    # 模式一：**Question** ... **Answer** 加粗标注
    qm = re.search(
        r"^\*\*Question\*\*\s*\n+(.+?)(?=\n\*\*Answer|\n---|\Z)",
        text, re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    am = re.search(
        r"^\*\*Answer[^*]*\*\*\s*\n+(.+)",
        text, re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if qm and am:
        question = _clean_post(qm.group(1).strip())
        solution = _clean_post(am.group(1).strip())
        if question and solution:
            return question, solution

    # 模式二：加粗用户名分隔（**Username** 后跟正文，非 H3）
    bold_user = re.compile(r"^\*\*([A-Za-z][\w\s\.]{1,40})\*\*\s*$", re.MULTILINE)
    posts = list(bold_user.finditer(text))
    if len(posts) >= 2:
        q_start = posts[0].end()
        q_end   = posts[1].start()
        question = _clean_post(text[q_start:q_end].strip())
        answer_parts = []
        for i in range(1, len(posts)):
            a_start = posts[i].end()
            a_end   = posts[i+1].start() if i+1 < len(posts) else len(text)
            part = _clean_post(text[a_start:a_end].strip())
            if part:
                answer_parts.append(part)
        solution = "\n\n".join(answer_parts)
        if question and solution:
            return question, solution

    return None

def process_record(record: dict) -> Optional[dict]:
    text = record["text"]
    result: Optional[tuple[str, str]] = None
    strategy = "SKIP"

    for strat_name, strat_fn in [
        ("B1", extract_b1),
        ("B2", extract_b2),
        ("B3", extract_b3),
        ("B4", extract_b4),
        ("B5", extract_b5),
    ]:
        result = strat_fn(text)
        if result:
            strategy = strat_name
            break

    if result is None:
        return None

    problem  = re.sub(r"\n{3,}", "\n\n", result[0]).strip()
    solution = re.sub(r"\n{3,}", "\n\n", result[1]).strip()

    if not validate(problem, solution):
        return None

    return {
        "id": record["id"],
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": problem},
            {"role": "assistant", "content": solution},
        ],
        "source_id": record["id"],
        "extraction_strategy": strategy,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "kept": 0, "skipped": 0,
             "B1": 0, "B2": 0, "B3": 0, "B4": 0, "B5": 0}

    with (
        open(INPUT_PATH,  encoding="utf-8") as fin,
        open(OUTPUT_PATH, "w", encoding="utf-8") as fout,
        open(SKIP_PATH,   "w", encoding="utf-8") as fskip,
    ):
        for line in fin:
            record = json.loads(line)
            stats["total"] += 1

            entry = process_record(record)
            if entry:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                stats[entry["extraction_strategy"]] += 1
            else:
                fskip.write(line)
                stats["skipped"] += 1

    total = stats["total"]
    kept  = stats["kept"]
    print(f"\n{'='*50}")
    print(f"Stage 4B 格式转换完成")
    print(f"{'='*50}")
    print(f"  总输入  : {total:>6,} 条")
    print(f"  保留    : {kept:>6,} 条 ({kept/total*100:.1f}%)")
    print(f"  丢弃    : {stats['skipped']:>6,} 条 ({stats['skipped']/total*100:.1f}%)")
    print(f"\n策略分布:")
    for s in ("B1", "B2", "B3", "B4", "B5"):
        n = stats[s]
        print(f"  {s}: {n:>5,} ({n/total*100:.1f}%)")
    print(f"\n输出: {OUTPUT_PATH}")
    print(f"跳过: {SKIP_PATH}")


if __name__ == "__main__":
    main()
