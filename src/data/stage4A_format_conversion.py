"""
Stage 4A: 将 stage3_A.jsonl (PROBLEM_SOLUTION 类) 转换为 SFT chat 训练格式。

输出格式（每行一条 JSON）:
{
    "id": "<原始id>",
    "messages": [
        {"role": "system",    "content": "<系统提示>"},
        {"role": "user",      "content": "<数学问题>"},
        {"role": "assistant", "content": "<解答步骤>"}
    ],
    "source_id": "<原始id>",
    "extraction_strategy": "S1/S1b/S2/S3/S4"
}

提取策略优先级:
  S1  : 同时有明确 ## Problem/Question 和 ## Solution/Answer 标题
  S1b : 只有 ## Solution 标题 → H1/H2 标题作问题
  S2  : 只有 ## Problem/Question 标题 → 该标题后所有内容作解答
  S3  : H1/H2 标题为问题形式（含数学符号/问题关键词）→ 标题作问题
  S4  : 兜底 → 第一段作问题，剩余作解答
  SKIP: 无法提取，丢弃
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────── 路径配置 ───────────────────────────
ROOT = Path("/home/ubuntu/Midterm_Project")
INPUT_PATH  = ROOT / "stage3_output" / "stage3_A.jsonl"
OUTPUT_PATH = ROOT / "stage4_output" / "stage4_A_sft.jsonl"
SKIP_PATH   = ROOT / "stage4_output" / "stage4_A_skipped.jsonl"

# ─────────────────────────── SFT 系统提示 ──────────────��────────
SYSTEM_PROMPT = (
    "You are a helpful mathematics assistant. "
    "When given a math problem, provide a clear, step-by-step solution "
    "showing all reasoning and calculations."
)

# ─────────────────────────── 正则模式 ───────────────────────────
# 章节标题：匹配独立成行的 # 标题
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# 明确的 Problem 标题
_PROB_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(problem\s*(statement|description)?|question|exercise|task|challenge)"
    r"\s*[\d\.\:]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# 明确的 Solution 标题
_SOL_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(solution|answer|proof|workings?|derivation|solution\s*approach)"
    r"\s*[\d\.\:]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# 问题类 H1/H2 标题：含数学符号 或 疑问词/动词短语
_Q_TITLE_MATH_RE = re.compile(r"\$|\\[a-zA-Z]+|\^|_\{")
_Q_TITLE_VERB_RE = re.compile(
    r"\b(find|prove|show|calculate|evaluate|solve|determine|compute|what\s+is"
    r"|how\s+(many|much|do|to|can)|simplify|factori[sz]e|if\s+.+then|given\s+that"
    r"|derive|verify|express|expand|integrate|differentiate|rationali[sz]e"
    r"|help|homework|urgently|questions?|problems?|solutions?|exercises?)\b",
    re.IGNORECASE,
)
# 数学主题词（用于 S3b：标题是主题词，但正文含大量数学）
_MATH_TOPIC_RE = re.compile(
    r"\b(algebra|calculus|geometry|trigonometry|statistics|probability"
    r"|theorem|lemma|proof|equation|inequality|function|matrix|vector"
    r"|integral|derivative|limit|series|sequence|combinatorics|number\s+theory"
    r"|polynomial|logarithm|exponential|complex\s+number|linear\s+algebra"
    r"|mechanics|physics|arithmetic|fraction|ratio|proportion|percentage"
    r"|parabola|ellipse|circle|triangle|angle|quadratic|cubic|modular"
    r"|eigenvalue|eigenvector|fourier|laplace|differential|parametric)\b",
    re.IGNORECASE,
)
# 质量过滤
MIN_PROBLEM_LEN  = 30    # 问题最短字符数
MIN_SOLUTION_LEN = 100   # 解答最短字符数
MIN_MATH_IN_SOL  = re.compile(r"\$|\\[a-zA-Z]+|\d+\s*[\+\-\*\/\^=]\s*\d+")


# ──────────────────────── 工具函数 ──────────────────────────────

def _get_section_content(text: str, start_match: re.Match, heading_level: int) -> str:
    """提取 start_match 所指标题之后、同级或更高级标题之前的内容。"""
    start = start_match.end()
    # 构造"同级或更高级标题"的终止模式
    stop_pat = re.compile(
        rf"^(?:#{{1,{heading_level}}})\s+",
        re.MULTILINE,
    )
    m = stop_pat.search(text, start)
    end = m.start() if m else len(text)
    return text[start:end].strip()


def _strip_heading(line: str) -> str:
    """去除 Markdown 标题符号，返回纯文本。"""
    return re.sub(r"^#{1,6}\s*", "", line).strip()


def _clean_text(text: str) -> str:
    """压缩多余空行，去除首尾空白。"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ──────────────────────── 提取策略 ──────────────────────────────

def extract_s1(text: str) -> Optional[tuple[str, str]]:
    """S1: 同时有明确 Problem 和 Solution 标题。"""
    pm = _PROB_HEADER_RE.search(text)
    sm = _SOL_HEADER_RE.search(text)
    if not (pm and sm):
        return None
    prob_level = len(pm.group(1))
    sol_level  = len(sm.group(1))
    problem  = _get_section_content(text, pm, prob_level)
    solution = _get_section_content(text, sm, sol_level)
    if problem and solution:
        return problem, solution
    return None


def extract_s1b(text: str) -> Optional[tuple[str, str]]:
    """S1b: 只有 Solution 标题 → Solution 之前最近一个段落或子标题作问题。"""
    sm = _SOL_HEADER_RE.search(text)
    if not sm:
        return None
    # Problem 标题不存在
    if _PROB_HEADER_RE.search(text):
        return None
    sol_level = len(sm.group(1))
    solution  = _get_section_content(text, sm, sol_level)

    # 在 Solution 标题之前，找最近一个同级或更低级的标题所在段落作为问题
    before_sol = text[: sm.start()].strip()
    # 找 before_sol 中的最后一个标题（该标题下的内容即为问题陈述）
    all_headings = list(_HEADING_RE.finditer(before_sol))
    if all_headings:
        last_h = all_headings[-1]
        h_level = len(last_h.group(1))
        problem_section = _get_section_content(before_sol, last_h, h_level)
        # 若该节为空，用标题本身作问题
        if not problem_section.strip():
            problem_section = _strip_heading(last_h.group(0))
    else:
        # 没有子标题 → 用 solution 前的全部内容
        problem_section = before_sol

    if problem_section and solution:
        return problem_section, solution
    return None


def extract_s2(text: str) -> Optional[tuple[str, str]]:
    """S2: 只有 Problem/Question 标题 → 该节后所有内容作解答。"""
    pm = _PROB_HEADER_RE.search(text)
    if not pm:
        return None
    if _SOL_HEADER_RE.search(text):
        return None   # 留给 S1 处理
    prob_level = len(pm.group(1))
    problem  = _get_section_content(text, pm, prob_level)
    # 解答 = Problem 节之后的所有内容
    stop_pat = re.compile(rf"^(?:#{{1,{prob_level}}})\s+", re.MULTILINE)
    m = stop_pat.search(text, pm.end())
    solution_start = m.start() if m else pm.end() + len(problem)
    solution = text[solution_start:].strip()
    if problem and solution:
        return problem, solution
    return None


def extract_s3(text: str) -> Optional[tuple[str, str]]:
    """S3: H1/H2 标题是疑问形式 → 标题作问题，全文 body 作解答。"""
    # 取文本最开头的 H1 或 H2
    m = re.match(r"^\s*(#{1,2})\s+(.+)$", text.strip(), re.MULTILINE)
    if not m:
        # 允许开头有少量空行
        m = re.search(r"^(#{1,2})\s+(.+)$", text, re.MULTILINE)
        if not m or m.start() > 200:
            return None
    title = m.group(2).strip()
    if not (_Q_TITLE_MATH_RE.search(title) or _Q_TITLE_VERB_RE.search(title)):
        return None
    # 解答 = 标题之后的所有内容
    solution = text[m.end():].strip()
    if title and solution:
        return title, solution
    return None


def extract_s3b(text: str) -> Optional[tuple[str, str]]:
    """S3b: 标题是数学主题词（无显式疑问词），但正文含大量数学内容。
    将标题转化为 'How do you ...' 或直接用标题作 prompt，body 作解答。"""
    m = re.search(r"^(#{1,2})\s+(.+)$", text, re.MULTILINE)
    if not m or m.start() > 200:
        return None
    title = m.group(2).strip()
    # 标题需包含数学主题词，且不含导航/集合类词（避免文章列表页）
    if not _MATH_TOPIC_RE.search(title):
        return None
    # 过滤掉课程目录、目录页等
    if re.search(r"\b(chapter|unit|module|part\s+\d|index|contents?|overview|syllabus)\b", title, re.I):
        return None
    solution = text[m.end():].strip()
    # 正文须含有大量数学（至少5处 LaTeX 表达式）
    if len(re.findall(r"\$[^$]+\$|\$\$[\s\S]+?\$\$", solution)) < 3:
        return None
    if title and solution:
        return title, solution
    return None


def extract_s4(text: str) -> Optional[tuple[str, str]]:
    """S4: 兜底 → 第一个非空段落作问题（若含问题关键词），其余作解答。"""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) < 2:
        return None
    first = paragraphs[0]
    # 去掉首段可能的标题符号
    first_clean = _strip_heading(first)
    if not (_Q_TITLE_VERB_RE.search(first_clean) or _Q_TITLE_MATH_RE.search(first_clean)):
        return None
    solution = "\n\n".join(paragraphs[1:])
    return first_clean, solution


def extract_s5(text: str) -> Optional[tuple[str, str]]:
    """S5: 正文中有 **Question**: / **Problem**: 等加粗标注的问题段落。"""
    _BOLD_Q_RE = re.compile(
        r"^\*\*(question|problem|exercise|task)\**\s*[\d\.\:]*\s*\*?\*?\s*[:：]?\s*(.+?)$",
        re.IGNORECASE | re.MULTILINE,
    )
    _BOLD_A_RE = re.compile(
        r"^\*\*(solution|answer|proof|hint)\**\s*[\d\.\:]*\s*\*?\*?\s*[:：]?\s*",
        re.IGNORECASE | re.MULTILINE,
    )
    qm = _BOLD_Q_RE.search(text)
    if not qm:
        return None
    # 收集问题文本：加粗行本身的内容 + 直到下一个加粗标签或空行前的段落
    q_start = qm.start()
    q_inline = qm.group(2).strip()
    # 找紧跟在问题后的内容（直到下一个 ** 标签或双空行）
    after_q = text[qm.end():]
    next_bold = re.search(r"^\*\*", after_q, re.MULTILINE)
    next_blank = re.search(r"\n{2,}", after_q)
    if next_bold and (not next_blank or next_bold.start() < next_blank.start()):
        extra = after_q[: next_bold.start()].strip()
    elif next_blank:
        extra = after_q[: next_blank.start()].strip()
    else:
        extra = after_q.strip()
    problem = (q_inline + ("\n" + extra if extra else "")).strip()

    # 找解答部分
    am = _BOLD_A_RE.search(text, qm.end())
    if am:
        sol_level_end = am.end()
        # 解答到下一个同级 ** 标签或文末
        next_section = re.search(r"^\*\*[A-Z]", text[sol_level_end:], re.MULTILINE)
        if next_section:
            solution = text[sol_level_end: sol_level_end + next_section.start()].strip()
        else:
            solution = text[sol_level_end:].strip()
    else:
        # 没有显式解答标签 → 问题之后的全部内容作解答
        solution = text[qm.end():].strip()

    if problem and solution:
        return problem, solution
    return None


# ──────────────────────── 质量验证 ──────────────────────────────

def validate(problem: str, solution: str) -> bool:
    """检查 (problem, solution) 对的最低质量要求。"""
    if len(problem) < MIN_PROBLEM_LEN:
        return False
    if len(solution) < MIN_SOLUTION_LEN:
        return False
    # 解答须含数学内容
    if not MIN_MATH_IN_SOL.search(solution):
        return False
    return True


# ──────────────────────── 主流程 ──────────────────────────────

def process_record(record: dict) -> Optional[dict]:
    """对单条记录依次尝试各提取策略，返回 SFT 条目或 None。"""
    text = record["text"]
    result: Optional[tuple[str, str]] = None
    strategy = "SKIP"

    for strat_name, strat_fn in [
        ("S1",  extract_s1),
        ("S1b", extract_s1b),
        ("S2",  extract_s2),
        ("S3",  extract_s3),
        ("S3b", extract_s3b),
        ("S4",  extract_s4),
        ("S5",  extract_s5),
    ]:
        result = strat_fn(text)
        if result:
            strategy = strat_name
            break

    if result is None:
        return None

    problem, solution = result
    problem  = _clean_text(problem)
    solution = _clean_text(solution)

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
             "S1": 0, "S1b": 0, "S2": 0, "S3": 0, "S3b": 0, "S4": 0, "S5": 0}

    with (
        open(INPUT_PATH,  encoding="utf-8") as fin,
        open(OUTPUT_PATH, "w", encoding="utf-8") as fout,
        open(SKIP_PATH,   "w", encoding="utf-8") as fskip,
    ):
        for line in fin:
            record = json.loads(line)
            stats["total"] += 1

            sft_entry = process_record(record)
            if sft_entry:
                fout.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                stats[sft_entry["extraction_strategy"]] += 1
            else:
                fskip.write(line)
                stats["skipped"] += 1

            if stats["total"] % 5000 == 0:
                print(f"  processed {stats['total']:,} / 46675 ...", flush=True)

    # ── 汇报 ──────────────────────────────────────────────────
    total = stats["total"]
    kept  = stats["kept"]
    print(f"\n{'='*50}")
    print(f"Stage 4A 格式转换完成")
    print(f"{'='*50}")
    print(f"  总输入  : {total:>8,} 条")
    print(f"  保留    : {kept:>8,} 条 ({kept/total*100:.1f}%)")
    print(f"  丢弃    : {stats['skipped']:>8,} 条 ({stats['skipped']/total*100:.1f}%)")
    print(f"\n策略分布:")
    for s in ("S1", "S1b", "S2", "S3", "S3b", "S4", "S5"):
        n = stats[s]
        print(f"  {s:4s}: {n:>7,} ({n/total*100:.1f}%)")
    print(f"\n输出: {OUTPUT_PATH}")
    print(f"跳过: {SKIP_PATH}")


if __name__ == "__main__":
    main()
