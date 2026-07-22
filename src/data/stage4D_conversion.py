"""
Stage 4D: 将 stage3_D.jsonl (TEXTBOOK 类) 转换为 SFT chat 训练格式。

D 类数据特点：来自教材网页、教材书目页、教材习题集等，内容多样：
  - 真实教材内容（~76 条）：含概念讲解、例题、习题、定理证明
  - 书目/目录页（~67 条）：无解答内容，仅含章节列表和ISBN信息
  - 整除/因数数字页（~5 条）：列举某数因子，无学习价值
  - NCERT类习题解答：含题目和完整解答步骤

策略设计：
  1. 首先复用 A/C 类已有策略（S1/S1b/S2/S3/S3b/S4/S5/C-Ex/C-Thm）；
  2. 新增 D-Inl 策略：提取段落内嵌的 Example/Exercise 块（含 "Example:" 或 "Example N" 格式）；
  3. 新增 D-Q 策略：提取选择题/填空题格式（含 "Answer:"/"Option" 关键词）；
  4. 新增 D-Step 策略：提取含 PROBLEM/STEP-BY-STEP SOLUTION 结构的教材题解；
  5. 书目页/数字页：质量过滤自动丢弃（无法构建有效 QA）。

输出格式（每行一条 JSON）:
{
    "id": "<原始id>",
    "messages": [
        {"role": "system",    "content": "<系统提示>"},
        {"role": "user",      "content": "<问题或上下文>"},
        {"role": "assistant", "content": "<解答或示例>"}
    ],
    "source_id": "<原始id>",
    "extraction_strategy": "S1/S1b/.../D-Inl/D-Q/D-Step"
}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# ─────────────────────────── 路径配置 ───────────────────────────
ROOT        = Path("/home/ubuntu/Midterm_Project")
INPUT_PATH  = ROOT / "stage3_output" / "stage3_D.jsonl"
OUTPUT_PATH = ROOT / "stage4_output" / "stage4_D_sft.jsonl"
SKIP_PATH   = ROOT / "stage4_output" / "stage4_D_skipped.jsonl"

# ─────────────────────────── SFT 系统提示 ───────────────────────
SYSTEM_PROMPT = (
    "You are a helpful mathematics assistant. "
    "When given a math problem, provide a clear, step-by-step solution "
    "showing all reasoning and calculations."
)

# ─────────────────────────── 正则模式 ───────────────────────────

# 复用 A/C 类：问题/解答标题
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

# C-Ex 复用：Example 节标题
_EX_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(example\s*[\d\.\:]*.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# C-Thm 复用：定理/引理/证明标题
_THM_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(theorem|lemma|corollary|proposition)\s*[\d\.\:]*.*?$",
    re.IGNORECASE | re.MULTILINE,
)
_PRF_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(proof|derivation)\s*[\d\.\:]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# 通用标题
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# 问题类标题判断（S3/S3b）
_Q_TITLE_MATH_RE = re.compile(r"\$|\\[a-zA-Z]+|\^|_\{")
_Q_TITLE_VERB_RE = re.compile(
    r"\b(find|prove|show|calculate|evaluate|solve|determine|compute|what\s+is"
    r"|how\s+(many|much|do|to|can)|simplify|factori[sz]e|if\s+.+then|given\s+that"
    r"|derive|verify|express|expand|integrate|differentiate|rationali[sz]e"
    r"|help|homework|urgently|questions?|problems?|solutions?|exercises?)\\b",
    re.IGNORECASE,
)
_MATH_TOPIC_RE = re.compile(
    r"\b(algebra|calculus|geometry|trigonometry|statistics|probability"
    r"|theorem|lemma|proof|equation|inequality|function|matrix|vector"
    r"|integral|derivative|limit|series|sequence|combinatorics|number\s+theory"
    r"|polynomial|logarithm|exponential|complex\s+number|linear\s+algebra"
    r"|mechanics|physics|arithmetic|fraction|ratio|proportion|percentage"
    r"|parabola|ellipse|circle|triangle|angle|quadratic|cubic|modular"
    r"|eigenvalue|eigenvector|fourier|laplace|differential|parametric"
    r"|mensuration|determinant|set\s+theory|factori[sz]ation|divisor)\\b",
    re.IGNORECASE,
)

# D-Inl：段落内嵌示例（"Example:" "Example N:" "Example N." 格式，非标题行）
_INLINE_EX_RE = re.compile(
    r"(?:^|\n)\*{0,2}(Example\s*[\d]*[\.\:])\*{0,2}\s*\n"
    r"(.+?)"
    r"(?=\n\*{0,2}Example\s*[\d]*[\.\:]|\n#{1,4}\s|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# D-Q：选择题/判断题（含 Answer: / Option A/B/C 结构）
_MCQ_QUESTION_RE = re.compile(
    r"^#{1,4}\s+(.+?)\n([\s\S]*?)"
    r"(?:\*{0,2}Answer[\*:]{0,3}|Option\s+[A-D][\.\:])",
    re.IGNORECASE | re.MULTILINE,
)
_MCQ_ANSWER_RE = re.compile(
    r"(?:\*{0,2}Answer[\*:]{0,3}|Correct answer[\s\*:]+)\s*(.+?)(?=\n\n|\n#{1,4}|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# D-Step：PROBLEM + STEP-BY-STEP SOLUTION 格式（教材习题解答页）
_STEP_PROB_RE = re.compile(
    r"\*{0,2}PROBLEM[\*:]{0,3}\s*\n([\s\S]*?)(?=\*{0,2}STEP|SOLUTION)",
    re.IGNORECASE,
)
_STEP_SOL_RE = re.compile(
    r"\*{0,2}STEP-BY-STEP\s+SOLUTION[\*:]{0,3}\s*\n([\s\S]+?)(?=\n#{1,4}\s|\Z)",
    re.IGNORECASE,
)

# 数学内容检测
_MATH_CONTENT_RE = re.compile(
    r"\$[^$]+\$|\$\$[\s\S]+?\$\$|\\[a-zA-Z]+|\d+\s*[\+\-\*\/\^=]\s*\d+"
)

# 书目页/目录页检测（用于快速丢弃）
_CATALOG_RE = re.compile(
    r"isbn[\s\-]*(?:10|13)?[\s\-:]|table of contents|published by|author[\s:]+\w"
    r"|chapter 1.*chapter 2|prentice hall|mcgraw|pearson|addison.wesley"
    r"|step-by-step solutions\s*\n.*available on ios"
    r"|related interests|uploaded by",
    re.IGNORECASE | re.DOTALL,
)

# 整除/因数页检测
_DIVISOR_RE = re.compile(
    r"divisors of \d+|list of.*?positive divisors",
    re.IGNORECASE,
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
    """三项质量门控：长度 + 数学内容。"""
    if len(problem) < MIN_PROBLEM_LEN:
        return False
    if len(solution) < MIN_SOLUTION_LEN:
        return False
    if not _MATH_CONTENT_RE.search(problem + solution):
        return False
    return True


# ──────────────────────── 复用 A/C 类策略 ───────────────────────

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
    """S3: H1/H2 标题本身是个问题（含动词或数学表达式），剩余全文作解答。"""
    first_h = _HEADING_RE.search(text)
    if not first_h:
        return None
    h_level = len(first_h.group(1))
    if h_level > 2:
        return None
    title = first_h.group(2).strip()
    if not (_Q_TITLE_VERB_RE.search(title) or _Q_TITLE_MATH_RE.search(title)):
        return None
    problem  = title
    solution = _get_section_content(text, first_h, h_level)
    if problem and solution:
        return problem, solution
    return None


def extract_s3b(text: str) -> Optional[tuple[str, str]]:
    """S3b: H1/H2 标题为数学主题词，之后全文作解答（教材章节式）。"""
    first_h = _HEADING_RE.search(text)
    if not first_h:
        return None
    h_level = len(first_h.group(1))
    if h_level > 2:
        return None
    title = first_h.group(2).strip()
    if not _MATH_TOPIC_RE.search(title):
        return None
    solution = _get_section_content(text, first_h, h_level)
    if len(solution) < 200:
        return None
    problem = f"Explain the concept of {title} with definitions and examples."
    return problem, solution


def extract_s4(text: str) -> Optional[tuple[str, str]]:
    """S4: 首段含问题关键词，以首段作问题，其余作解答。"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return None
    first = paras[0]
    if not _Q_TITLE_VERB_RE.search(first):
        return None
    if first.startswith("#"):
        return None
    solution = "\n\n".join(paras[1:])
    return first, solution


def extract_s5(text: str) -> Optional[tuple[str, str]]:
    """S5: **Question**: 加粗标注格式。"""
    m_q = re.search(r"\*{1,2}(?:question|problem)\*{0,2}[\s\*:]+(.+?)(?=\*{1,2}(?:answer|solution)\*{0,2}[\s\*:]|\Z)",
                    text, re.IGNORECASE | re.DOTALL)
    m_a = re.search(r"\*{1,2}(?:answer|solution)\*{0,2}[\s\*:]+(.+?)(?=\n\n#{1,4}|\Z)",
                    text, re.IGNORECASE | re.DOTALL)
    if not (m_q and m_a):
        return None
    problem  = m_q.group(1).strip()
    solution = m_a.group(1).strip()
    if problem and solution:
        return problem, solution
    return None


def extract_c_ex(text: str) -> Optional[tuple[str, str]]:
    """C-Ex: 从 ## Example 节提取演示型 QA。"""
    matches = list(_EX_HEADER_RE.finditer(text))
    if not matches:
        return None
    for ex_m in matches:
        ex_level = len(ex_m.group(1))
        solution = _get_section_content(text, ex_m, ex_level)
        if not solution or len(solution) < MIN_SOLUTION_LEN:
            continue
        before = text[:ex_m.start()].strip()
        all_before_headings = list(_HEADING_RE.finditer(before))
        if all_before_headings:
            last_h = all_before_headings[-1]
            h_level = len(last_h.group(1))
            concept = _get_section_content(before, last_h, h_level)
            concept_title = _strip_heading(last_h.group(0))
            problem = f"{concept_title}\n\n{concept}" if concept else concept_title
        else:
            problem = before
        if problem.lower().startswith("example"):
            higher = [h for h in all_before_headings if len(h.group(1)) < ex_level]
            if higher:
                parent_h = higher[-1]
                p_level  = len(parent_h.group(1))
                problem  = _get_section_content(before, parent_h, p_level)
            else:
                first_line = solution.split("\n")[0].strip()
                problem = first_line if first_line else problem
        if validate(problem, solution):
            return _clean_text(problem), _clean_text(solution)
    return None


def extract_c_thm(text: str) -> Optional[tuple[str, str]]:
    """C-Thm: 从 Theorem/Lemma + Proof 结构提取。"""
    thm_matches = list(_THM_HEADER_RE.finditer(text))
    prf_matches = list(_PRF_HEADER_RE.finditer(text))
    if not (thm_matches and prf_matches):
        return None
    for thm_m in thm_matches:
        thm_level = len(thm_m.group(1))
        thm_title = _strip_heading(thm_m.group(0))
        statement = _get_section_content(text, thm_m, thm_level)
        prf_after = [p for p in prf_matches if p.start() > thm_m.end()]
        if not prf_after:
            continue
        prf_m     = prf_after[0]
        prf_level = len(prf_m.group(1))
        proof     = _get_section_content(text, prf_m, prf_level)
        problem  = f"State and prove the following theorem:\n\n**{thm_title}**: {statement}"
        solution = f"**Proof:**\n\n{proof}"
        if validate(problem, solution):
            return _clean_text(problem), _clean_text(solution)
    return None


# ──────────────────────── D 类专属策略 ──────────────────────────

def extract_d_inl(text: str) -> Optional[tuple[str, str]]:
    """D-Inl: 提取段落内嵌的 'Example N:' 块（非标题行）。

    适用于 Record 15 类型：正文中 "Example: Solve 3x - 9 = 0." + 解答段落。
    避免与 C-Ex 重复（C-Ex 处理标题行 ##，D-Inl 处理段落内行）。
    """
    # 匹配 "Example:" 或 "Example N:" 引导的内联块
    pattern = re.compile(
        r"(?m)^(?!#).*?(?:Example\s*\d*[\.\:])\s*(.+?)(?=\n\n(?!Example)|\Z)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    for m in matches:
        # 提取示例内容（问题部分）
        example_block = m.group(0).strip()
        # 找示例之后的解答段落
        rest = text[m.end():].strip()
        # 解答段落：取连续非空行直到下一个空段或标题
        sol_match = re.match(
            r"((?:\*{0,2}Solution[\*\s:]*|Step\s*\d+[\.\:]?).+?)(?=\n\n|\n#{1,4}|\Z)",
            rest,
            re.IGNORECASE | re.DOTALL,
        )
        if sol_match:
            solution = sol_match.group(1).strip()
        else:
            # 取紧接的两段作为解答
            paras = [p.strip() for p in rest.split("\n\n") if p.strip()]
            solution = "\n\n".join(paras[:2]) if paras else ""

        problem = example_block
        if validate(problem, solution):
            return _clean_text(problem), _clean_text(solution)
    return None


def extract_d_q(text: str) -> Optional[tuple[str, str]]:
    """D-Q: 提取选择题/判断题格式。

    适用于 Record 80 类型：
      ## If A = {7,8,9}, then relation R = {(8,9)} is:
      - A. Symmetric only
      - B. Transitive only
      **Answer:** Option C is correct.
      **Hint:** ...
    """
    # 找 MCQ 结构：标题 + 选项行 + Answer 行
    m_q = re.search(
        r"^(#{1,4})\s+(.+?)\n"              # 标题即问题
        r"((?:\s*[-\*]\s+\*{0,2}[A-D]\..*\n)+)"  # 选项行
        r"(?=.*?\*{0,2}Answer)",
        text, re.MULTILINE | re.DOTALL,
    )
    m_a = _MCQ_ANSWER_RE.search(text)
    if not (m_q and m_a):
        return None

    question_title = m_q.group(2).strip()
    options        = m_q.group(3).strip()
    answer_raw     = m_a.group(1).strip()

    # 构建问题：标题 + 选项
    problem = f"{question_title}\n\n{options}"

    # 构建解答：Answer + 后续 Hint/Explanation（若有）
    hint_match = re.search(
        r"\*{0,2}Hint[\*\s:]+(.+?)(?=\n\n#{1,4}|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if hint_match:
        solution = f"{answer_raw}\n\n**Explanation:**\n{hint_match.group(1).strip()}"
    else:
        solution = answer_raw

    if validate(problem, solution):
        return _clean_text(problem), _clean_text(solution)
    return None


def extract_d_step(text: str) -> Optional[tuple[str, str]]:
    """D-Step: 提取 **PROBLEM:** + **STEP-BY-STEP SOLUTION:** 格式。

    适用于 Record 70 类型（教材习题解答页）：
      ### Chapter 7: Problem 73X1
      **PROBLEM:**
      Construction Problem ...
      **STEP-BY-STEP SOLUTION:**
      1. Convert ...
    """
    # 尝试匹配 PROBLEM: 开始的结构
    m_prob = _STEP_PROB_RE.search(text)
    m_sol  = _STEP_SOL_RE.search(text)

    if not (m_prob and m_sol):
        # 宽松版：寻找 "**PROBLEM**" 或 "PROBLEM:" 后跟解答段落
        m_prob = re.search(r"\*{0,2}PROBLEM\*{0,2}[\s:\*]+(.+?)(?=\*{0,2}STEP|\*{0,2}SOLUTION)", text, re.IGNORECASE | re.DOTALL)
        m_sol  = re.search(r"\*{0,2}(?:STEP-BY-STEP\s+)?SOLUTION\*{0,2}[\s:\*]+(.+?)(?=\n#{1,4}|\Z)", text, re.IGNORECASE | re.DOTALL)
        if not (m_prob and m_sol):
            return None

    problem  = m_prob.group(1).strip()
    solution = m_sol.group(1).strip()

    # 截断句检测：problem 以小写字母开头，说明被错误截断，丢弃
    if problem and problem[0].islower():
        return None

    # 若 problem 太短，向上查找章节标题补充上下文
    if len(problem) < MIN_PROBLEM_LEN:
        heading_m = _HEADING_RE.search(text)
        if heading_m:
            problem = f"{_strip_heading(heading_m.group(0))}\n\n{problem}"

    if validate(problem, solution):
        return _clean_text(problem), _clean_text(solution)
    return None


# ──────────────────────── 快速丢弃检测 ──────────────────────────

def is_low_value(text: str) -> bool:
    """检测书目页/整除数字页等低价值文档，快速丢弃。"""
    if _DIVISOR_RE.search(text):
        return True
    # 书目/目录页：含 ISBN 且长度 < 4000，通常无真实解答内容
    if re.search(r"isbn[\s\-]*(?:10|13)?[\s\-:]", text, re.IGNORECASE):
        if len(text) < 4000:
            return True
    # Table of Contents 类
    if re.search(r"table of contents|chapter \d+.*\n.*chapter \d+", text, re.IGNORECASE | re.DOTALL):
        if not re.search(r"example|solution|proof|theorem", text, re.IGNORECASE):
            return True
    # 上传者/相关兴趣类（Scribd 格式）
    if re.search(r"uploaded by|related interests", text, re.IGNORECASE):
        if len(text) < 3000:
            return True
    return False


# ──────────────────────── 主流程 ────────────────────────────────

STRATEGIES: list[tuple[str, callable]] = [
    ("S1",     extract_s1),
    ("S1b",    extract_s1b),
    ("S2",     extract_s2),
    ("S3",     extract_s3),
    ("C-Ex",   extract_c_ex),
    ("S3b",    extract_s3b),
    ("C-Thm",  extract_c_thm),
    ("D-Step", extract_d_step),
    ("D-Q",    extract_d_q),
    ("D-Inl",  extract_d_inl),
    ("S5",     extract_s5),
    ("S4",     extract_s4),
]


def process_record(record: dict) -> Optional[dict]:
    """处理单条记录，返回 SFT 格式字典或 None。"""
    text = record.get("text", "")
    rid  = record.get("id", "")

    # 快速丢弃低价值文档
    if is_low_value(text):
        return None

    for strat_name, strat_fn in STRATEGIES:
        result = strat_fn(text)
        if result is None:
            continue
        problem, solution = result
        if not validate(problem, solution):
            continue
        return {
            "id": rid,
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": _clean_text(problem)},
                {"role": "assistant", "content": _clean_text(solution)},
            ],
            "source_id": rid,
            "extraction_strategy": strat_name,
        }
    return None


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    kept_records: list[dict]    = []
    skipped_records: list[dict] = []
    strategy_counts: dict[str, int] = {}
    low_value_count = 0

    for record in records:
        text = record.get("text", "")
        if is_low_value(text):
            low_value_count += 1
            skipped_records.append(record)
            continue

        result = process_record(record)
        if result:
            kept_records.append(result)
            strat = result["extraction_strategy"]
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
        else:
            skipped_records.append(record)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in kept_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(SKIP_PATH, "w", encoding="utf-8") as f:
        for r in skipped_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kept  = len(kept_records)
    skipped = len(skipped_records)

    print(f"Stage 4D 转换完成")
    print(f"  总输入  : {total:>6} 条")
    print(f"  低价值丢弃: {low_value_count:>4} 条（书目/数字页）")
    print(f"  保留    : {kept:>6} 条 ({kept/total*100:.1f}%)")
    print(f"  跳过    : {skipped:>6} 条 ({skipped/total*100:.1f}%)")
    print(f"\n策略命中分布:")
    for name, _ in STRATEGIES:
        cnt = strategy_counts.get(name, 0)
        if cnt:
            pct = cnt / kept * 100 if kept else 0
            print(f"  {name:<8}: {cnt:>4} 条 ({pct:.1f}%)")
    print(f"\n输出: {OUTPUT_PATH}")
    print(f"跳过: {SKIP_PATH}")


if __name__ == "__main__":
    main()
