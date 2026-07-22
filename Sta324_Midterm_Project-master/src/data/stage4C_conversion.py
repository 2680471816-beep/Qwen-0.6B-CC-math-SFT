"""
Stage 4C: 将 stage3_C.jsonl (ARTICLE_TUTORIAL 类) 转换为 SFT chat 训练格式。

C 类数据特点：来自教育类文章、教程、教科书节选，内容以知识讲解为主。
与 A 类不同，C 类文档不一定有明确的"问题-解答"结构，但往往含有：
  - Example 节：概念讲解后跟具体示例（最常见，33.6%）
  - Theorem/Proof 节：定理+证明
  - 嵌入的 Problem/Solution 节（54% 含 solution 关键词）

提取策略（按优先级）：
  优先复用 A 类策略（S1/S1b/S2/S3/S3b/S4/S5），处理含明确 Q&A 结构的记录；
  新增 C-Ex 策略，从 ## Example 节提取演示型 QA 对；
  新增 C-Thm 策略，从 Theorem/Proof 结构提取；
  跳过无任何可提取结构的纯概念文章。

输出格式（每行一条 JSON）:
{
    "id": "<原始id>",
    "messages": [
        {"role": "system",    "content": "<系统提示>"},
        {"role": "user",      "content": "<问题或上下文>"},
        {"role": "assistant", "content": "<解答或示例>"}
    ],
    "source_id": "<原始id>",
    "extraction_strategy": "S1/S1b/.../C-Ex/C-Thm"
}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# ─────────────────────────── 路径配置 ───────────────────────────
ROOT        = Path("/home/ubuntu/Midterm_Project")
INPUT_PATH  = ROOT / "stage3_output" / "stage3_C.jsonl"
OUTPUT_PATH = ROOT / "stage4_output" / "stage4_C_sft.jsonl"
SKIP_PATH   = ROOT / "stage4_output" / "stage4_C_skipped.jsonl"

# ─────────────────────────── SFT 系统提示 ───────────────────────
SYSTEM_PROMPT = (
    "You are a helpful mathematics assistant. "
    "When given a math problem, provide a clear, step-by-step solution "
    "showing all reasoning and calculations."
)

# ─────────────────────────── 正则模式 ───────────────────────────

# A 类策略复用：问题/解答标题
_PROB_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(problem\s*(statement|description)?|question|exercise|task|challenge)"
    r"\s*[\d\.\:]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SOL_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(solution|answer|proof|workings?|derivation|solution\s*approach)"
    r"\s*[\d\.\:]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# C-Ex：Example 节标题（允许 "Example 1:"、"Example: ..." 等变体）
_EX_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(example\s*[\d\.\:]*.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# C-Thm：定理/引理标题
_THM_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(theorem|lemma|corollary|proposition)\s*[\d\.\:]*.*?$",
    re.IGNORECASE | re.MULTILINE,
)
# C-Thm：证明节标题
_PRF_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(proof|derivation)\s*[\d\.\:]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# H1/H2 标题
_HEADING_RE    = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# 问题类标题判断
_Q_TITLE_MATH_RE = re.compile(r"\$|\\[a-zA-Z]+|\^|_\{")
_Q_TITLE_VERB_RE = re.compile(
    r"\b(find|prove|show|calculate|evaluate|solve|determine|compute|what\s+is"
    r"|how\s+(many|much|do|to|can)|simplify|factori[sz]e|if\s+.+then|given\s+that"
    r"|derive|verify|express|expand|integrate|differentiate|rationali[sz]e"
    r"|help|homework|urgently|questions?|problems?|solutions?|exercises?)\b",
    re.IGNORECASE,
)
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

# 数学内容检测
_MATH_CONTENT_RE = re.compile(
    r"\$[^$]+\$|\$\$[\s\S]+?\$\$|\\[a-zA-Z]+|\d+\s*[\+\-\*\/\^=]\s*\d+"
)

# 质量阈值
MIN_PROBLEM_LEN  = 30
MIN_SOLUTION_LEN = 100


# ──────────────────────── 工具函数 ──────────────────────────────

def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


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
    if not _MATH_CONTENT_RE.search(problem + solution):
        return False
    return True


# ──────────────────────── 复用 A 类策略 ─────────────────────────

def extract_s1(text: str) -> Optional[tuple[str, str]]:
    """S1: 同时有明确 ## Problem 和 ## Solution 标题。"""
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
    """S1b: 只有 ## Solution 标题，取前一节最近标题下段落作问题。"""
    sm = _SOL_HEADER_RE.search(text)
    if not sm:
        return None
    if _PROB_HEADER_RE.search(text):
        return None
    sol_level = len(sm.group(1))
    solution  = _get_section_content(text, sm, sol_level)
    before_sol = text[:sm.start()].strip()
    all_headings = list(_HEADING_RE.finditer(before_sol))
    if all_headings:
        last_h = all_headings[-1]
        h_level = len(last_h.group(1))
        problem = _get_section_content(before_sol, last_h, h_level)
        if not problem.strip():
            problem = _strip_heading(last_h.group(0))
    else:
        problem = before_sol
    if problem and solution:
        return problem, solution
    return None


def extract_s2(text: str) -> Optional[tuple[str, str]]:
    """S2: 只有 ## Problem/Question 标题，之后全文作解答。"""
    pm = _PROB_HEADER_RE.search(text)
    if not pm:
        return None
    if _SOL_HEADER_RE.search(text):
        return None
    prob_level = len(pm.group(1))
    problem    = _get_section_content(text, pm, prob_level)
    stop_pat   = re.compile(rf"^(?:#{{1,{prob_level}}})\s+", re.MULTILINE)
    m = stop_pat.search(text, pm.end())
    solution_start = m.start() if m else pm.end() + len(problem)
    solution = text[solution_start:].strip()
    if problem and solution:
        return problem, solution
    return None


def extract_s3(text: str) -> Optional[tuple[str, str]]:
    """S3: H1/H2 标题含疑问词或数学符号，全文 body 作解答。"""
    m = re.search(r"^(#{1,2})\s+(.+)$", text, re.MULTILINE)
    if not m or m.start() > 200:
        return None
    title = m.group(2).strip()
    if not (_Q_TITLE_MATH_RE.search(title) or _Q_TITLE_VERB_RE.search(title)):
        return None
    solution = text[m.end():].strip()
    if title and solution:
        return title, solution
    return None


def extract_s3b(text: str) -> Optional[tuple[str, str]]:
    """S3b: H1/H2 为数学主题词，正文含 ≥3 处 LaTeX。"""
    m = re.search(r"^(#{1,2})\s+(.+)$", text, re.MULTILINE)
    if not m or m.start() > 200:
        return None
    title = m.group(2).strip()
    if not _MATH_TOPIC_RE.search(title):
        return None
    if re.search(r"\b(chapter|unit|module|part\s+\d|index|contents?|overview|syllabus)\b",
                 title, re.I):
        return None
    solution = text[m.end():].strip()
    if len(re.findall(r"\$[^$]+\$|\$\$[\s\S]+?\$\$", solution)) < 3:
        return None
    if title and solution:
        return title, solution
    return None


def extract_s4(text: str) -> Optional[tuple[str, str]]:
    """S4: 首段含问题关键词，其余作解答。"""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) < 2:
        return None
    first = _strip_heading(paragraphs[0])
    if not (_Q_TITLE_VERB_RE.search(first) or _Q_TITLE_MATH_RE.search(first)):
        return None
    solution = "\n\n".join(paragraphs[1:])
    return first, solution


def extract_s5(text: str) -> Optional[tuple[str, str]]:
    """S5: **Question**: / **Problem**: 加粗标注格式。"""
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
    q_inline = qm.group(2).strip()
    after_q  = text[qm.end():]
    next_bold = re.search(r"^\*\*", after_q, re.MULTILINE)
    next_blank = re.search(r"\n{2,}", after_q)
    if next_bold and (not next_blank or next_bold.start() < next_blank.start()):
        extra = after_q[:next_bold.start()].strip()
    elif next_blank:
        extra = after_q[:next_blank.start()].strip()
    else:
        extra = after_q.strip()
    problem = (q_inline + ("\n" + extra if extra else "")).strip()
    am = _BOLD_A_RE.search(text, qm.end())
    if am:
        sol_level_end = am.end()
        next_section  = re.search(r"^\*\*[A-Z]", text[sol_level_end:], re.MULTILINE)
        solution = (text[sol_level_end:sol_level_end + next_section.start()].strip()
                    if next_section else text[sol_level_end:].strip())
    else:
        solution = text[qm.end():].strip()
    if problem and solution:
        return problem, solution
    return None


# ──────────────────────── C 类专用策略 ──────────────────────────

def extract_c_ex(text: str) -> Optional[tuple[str, str]]:
    """C-Ex: 从 ## Example 节提取演示型 QA 对。

    逻辑：
    - 找第一个 Example 节
    - Example 之前的概念讲解（最近的标题节或 preamble）作为"问题背景+问题"
    - Example 节的内容作为"解答/演示"
    - 若文档有多个 Example 节，每个都尝试生成独立 QA 对（取第一个成功的）
    """
    matches = list(_EX_HEADER_RE.finditer(text))
    if not matches:
        return None

    for ex_m in matches:
        ex_level = len(ex_m.group(1))
        ex_title = ex_m.group(2).strip()  # "Example 1: Solving..."

        # Example 节的内容（解答侧）
        solution = _get_section_content(text, ex_m, ex_level)
        if len(solution) < MIN_SOLUTION_LEN:
            continue

        # 构建问题：Example 之前的内容
        before = text[:ex_m.start()].strip()
        if not before:
            continue

        # 从 before 中提取最近一个标题节作为概念背景
        all_before_headings = list(_HEADING_RE.finditer(before))
        if all_before_headings:
            last_h = all_before_headings[-1]
            h_level = len(last_h.group(1))
            concept = _get_section_content(before, last_h, h_level)
            concept_title = last_h.group(2).strip()
            if concept:
                # 构成"问题"：概念标题作为主题，"请展示一个例子"
                problem = f"{concept_title}\n\n{concept}"
            else:
                problem = concept_title
        else:
            # 无标题，用 preamble 整体作为问题背景
            problem = before

        problem = _clean_text(problem)
        solution = _clean_text(solution)

        # Example 标题中若含有具体问题描述（"Example 1: Find..."），用它替代长段 problem
        if re.search(r"(find|prove|show|calculate|solve|evaluate|determine|compute)", ex_title, re.I):
            problem = ex_title
        # 若 problem 以 "Example" 开头（说明只有标题，无具体背景），改用上层标题构造问题
        elif problem.lower().startswith("example"):
            # 往上找更高层标题
            higher_hdrs = [h for h in all_before_headings[:-1]] if all_before_headings else []
            if higher_hdrs:
                top_h = higher_hdrs[-1]
                top_title = top_h.group(2).strip()
                top_level = len(top_h.group(1))
                top_content = _get_section_content(before, top_h, top_level)
                problem = _clean_text(f"{top_title}\n\n{top_content}") if top_content else top_title
            else:
                # 实在没有背景，用 Example 标题本身 + solution 首句构成问题
                first_line = solution.split("\n")[0].strip()
                problem = first_line if len(first_line) >= MIN_PROBLEM_LEN else ex_title

        if validate(problem, solution):
            return problem, solution

    return None


def extract_c_thm(text: str) -> Optional[tuple[str, str]]:
    """C-Thm: 从 Theorem/Lemma + Proof 结构提取。

    逻辑：
    - 找 ## Theorem / ## Lemma 节（定理陈述）
    - 找其后对应的 ## Proof 节（证明）
    - 定理陈述作为"问题"（State and prove...）
    - Proof 节内容作为"解答"
    """
    thm_m = _THM_HEADER_RE.search(text)
    if not thm_m:
        return None

    thm_level = len(thm_m.group(1))
    thm_title = thm_m.group(2).strip()
    theorem   = _get_section_content(text, thm_m, thm_level)

    # 在 Theorem 节之后找 Proof 节
    prf_m = _PRF_HEADER_RE.search(text, thm_m.end())
    if not prf_m:
        return None

    prf_level = len(prf_m.group(1))
    proof     = _get_section_content(text, prf_m, prf_level)

    if not theorem or not proof:
        return None

    # 问题 = 定理标题 + 定理内容（让模型"证明"它）
    problem = f"{thm_title}\n\n{theorem}" if theorem else thm_title
    problem = _clean_text(problem)
    proof   = _clean_text(proof)

    if validate(problem, proof):
        return problem, proof
    return None


# ──────────────────────── 主流程 ────────────────────────────────

def process_record(record: dict) -> Optional[dict]:
    text = record["text"]
    result: Optional[tuple[str, str]] = None
    strategy = "SKIP"

    for strat_name, strat_fn in [
        ("S1",    extract_s1),
        ("S1b",   extract_s1b),
        ("S2",    extract_s2),
        ("S3",    extract_s3),
        ("C-Ex",  extract_c_ex),
        ("S3b",   extract_s3b),
        ("C-Thm", extract_c_thm),
        ("S5",    extract_s5),
        ("S4",    extract_s4),
    ]:
        result = strat_fn(text)
        if result:
            strategy = strat_name
            break

    if result is None:
        return None

    problem  = _clean_text(result[0])
    solution = _clean_text(result[1])

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
             "S1": 0, "S1b": 0, "S2": 0, "S3": 0,
             "C-Ex": 0, "S3b": 0, "C-Thm": 0, "S5": 0, "S4": 0}

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

            if stats["total"] % 5000 == 0:
                print(f"  processed {stats['total']:,} / 27682 ...", flush=True)

    total = stats["total"]
    kept  = stats["kept"]
    print(f"\n{'='*50}")
    print(f"Stage 4C 格式转换完成")
    print(f"{'='*50}")
    print(f"  总输入  : {total:>8,} 条")
    print(f"  保留    : {kept:>8,} 条 ({kept/total*100:.1f}%)")
    print(f"  丢弃    : {stats['skipped']:>8,} 条 ({stats['skipped']/total*100:.1f}%)")
    print(f"\n策略分布:")
    for s in ("S1", "S1b", "S2", "S3", "C-Ex", "S3b", "C-Thm", "S5", "S4"):
        n = stats[s]
        if n:
            print(f"  {s:8s}: {n:>7,} ({n/total*100:.1f}%)")
    print(f"\n输出: {OUTPUT_PATH}")
    print(f"跳过: {SKIP_PATH}")


if __name__ == "__main__":
    main()
