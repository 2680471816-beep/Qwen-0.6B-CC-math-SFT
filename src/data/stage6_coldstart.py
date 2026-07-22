"""
Stage 6: 高质量冷启动数据提取（Cold-Start Bootstrapping）

从 stage3_output/ 出发，通过纯正则化方式提取含 \\boxed{} 的高质量 SFT 数据。
目标：3,000–8,000 条绝对干净的"火种数据"，质量第一，数量第二。

核心设计原则：
  - 不强加 <think> 标签：保留原始步骤，不添加不存在的推理链
  - system prompt 为空字符串（与 CLAUDE.md 格式对齐）
  - \\boxed{} 作为强制过滤条件（硬性要求）
  - 七层纯规则质量门控，零 GPU 消耗

输出格式：
  {
    "messages": [
      {"role": "system",    "content": ""},
      {"role": "user",      "content": "题目"},
      {"role": "assistant", "content": "步骤解答...\\n\\n\\\\boxed{答案}"}
    ]
  }

用法：
  python src/data/stage6_coldstart.py --input A
  python src/data/stage6_coldstart.py --input A --dry-run
  python src/data/stage6_coldstart.py --input all
  python src/data/stage6_coldstart.py --merge
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────── 路径配置 ───────────────────────────
ROOT        = Path("/home/ubuntu/Midterm_Project")
STAGE3_DIR  = ROOT / "stage3_output"
OUT_DIR     = ROOT / "outputs" / "coldstart"

INPUT_FILES = {
    "A": STAGE3_DIR / "stage3_A.jsonl",
    "B": STAGE3_DIR / "stage3_B.jsonl",
    "C": STAGE3_DIR / "stage3_C.jsonl",
    "D": STAGE3_DIR / "stage3_D.jsonl",
}
TOTAL_COUNTS = {"A": 46675, "B": 641, "C": 27682, "D": 151}

FINAL_OUTPUT  = OUT_DIR / "coldstart_sft_train.jsonl"
AUDIT_OUTPUT  = OUT_DIR / "coldstart_audit.jsonl"

# ─────────────────────────── 质量阈值 ───────────────────────────
MIN_PROBLEM_LEN  = 30       # 问题最短字符数
MAX_PROBLEM_LEN  = 2000     # 问题最长字符数
MIN_SOLUTION_LEN = 150      # 解答最短字符数（比 stage4 更严格）
MAX_SOLUTION_LEN = 6000     # 解答最长字符数
# 按 1 token ≈ 3.5 字符估算，系统消息+格式开销约 20 tokens
MAX_TOTAL_CHARS  = int(9012 * 3.5)

# ─────────────────────────── 正则模式 ───────────────────────────

# 提取策略相关
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
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
_EX_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(example\s*[\d\.\:]*.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_THM_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(theorem|lemma|corollary|proposition)\s*[\d\.\:]*.*?$",
    re.IGNORECASE | re.MULTILINE,
)
_PRF_HEADER_RE = re.compile(
    r"^(#{1,4})\s*(proof|derivation)\s*[\d\.\:]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
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

# L1: \boxed{} 检测
_BOXED_RE      = re.compile(r"\\boxed\{[^}]*\}")
_BOXED_FULL_RE = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)

# 答案注入：结论句中的行内 LaTeX
_CONCL_INLINE_RE = re.compile(
    r"(?:thus|therefore|hence|so|consequently|we\s+get|we\s+have|we\s+obtain"
    r"|the\s+(?:final\s+)?answer\s+is|is\s+equal\s+to|equals?|gives?\s+us"
    r"|result\s+is|value\s+is|solution\s+is)"
    r"[^$\n]{0,60}\$([^$\n]{1,120})\$[.,;)!\s]*$",
    re.IGNORECASE | re.MULTILINE,
)
# 行内 LaTeX 以 = 开头的结论句
_EQ_INLINE_RE = re.compile(
    r"(?:thus|therefore|hence|so|consequently|we\s+get|the\s+answer\s+is)"
    r"[^$\n]{0,40}\$\s*([a-zA-Z_]\s*=\s*[^$\n]{1,80})\$[.,;)!\s]*$",
    re.IGNORECASE | re.MULTILINE,
)
# 最后一个展示公式块（$$...$$）结尾为简单表达式
_DISPLAY_FINAL_RE = re.compile(
    r"\$\$\s*\n*\s*(?:=\s*)?([^\n$]{1,100})\s*\n*\s*\$\$\s*$",
    re.DOTALL,
)
# "Correct answer: $X$" 或 "Answer: X"
_ANSWER_LABEL_RE = re.compile(
    r"(?:correct\s+)?[Aa]nswer\s*[:\-]\s*\$?\s*([A-Za-z0-9\+\-\\\{\}\s\^\_\(\)\/\.\,]{1,80}?)\$?\s*[.\n]?$",
    re.MULTILINE,
)
# 最后一行 "= VALUE" 模式（display math 中的最终等式）
_LAST_EQ_RE = re.compile(
    r"=\s*([A-Za-z0-9\\\{\}\s\^\_\(\)\/\.\,\+\-\*]{1,80})\s*$",
)

# 噪声词（不应出现在 \boxed{} 内容中）
_BOXED_NOISE_RE = re.compile(
    r"\b(approximately|units?|meters?|cm|kg|seconds?|minutes?|hours?|See|Note|Since|"
    r"where|which|this|that|Therefore|Thus|Hence)\b",
    re.IGNORECASE,
)

# L3: 数学密度
_LATEX_EXPR_RE = re.compile(r"\$[^$\n]+\$|\$\$[\s\S]+?\$\$")
_CORE_MATH_RE  = re.compile(r"\\(?:frac|sqrt|sum|int|prod|lim|infty|pm|times|div"
                             r"|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma"
                             r"|leq|geq|neq|approx|equiv|cdot|ldots|binom|vec|hat"
                             r"|mathbb|mathbf|text|left|right|begin|end)\b")

# L4: 噪声检测
_NOISE_RE = re.compile(
    r"\b(see\s+also|references?|external\s+links?|further\s+reading"
    r"|posted\s+by|comments?\s*:|reply\s*:|author\s*:|edited\s+by"
    r"|click\s+here|subscribe|newsletter|copyright|all\s+rights\s+reserved"
    r"|privacy\s+policy|terms\s+of\s+service|cookies?)\b",
    re.IGNORECASE,
)
_URL_RE        = re.compile(r"https?://\S+")
_CODE_BLOCK_RE = re.compile(r"```[\w]*\n[\s\S]+?```")

# L6: 问题独立性
_DANGLING_START_RE = re.compile(
    r"^(it\b|this\b|these\b|those\b|the\s+above|the\s+following|as\s+shown"
    r"|from\s+the\s+above|in\s+the\s+figure|see\s+figure|refer\s+to)",
    re.IGNORECASE,
)
_EXTERNAL_REF_RE = re.compile(
    r"\b(click\s+here|see\s+figure|as\s+shown\s+in|refer\s+to\s+the"
    r"|in\s+the\s+diagram|from\s+the\s+graph|the\s+table\s+below)\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def _get_section_content(text: str, start_match: re.Match, heading_level: int) -> str:
    """提取 start_match 所指标题之后、同级或更高级标题之前的内容。"""
    start = start_match.end()
    stop_pat = re.compile(rf"^(?:#{{1,{heading_level}}})\s+", re.MULTILINE)
    m = stop_pat.search(text, start)
    end = m.start() if m else len(text)
    return text[start:end].strip()


def _strip_heading(line: str) -> str:
    return re.sub(r"^#{1,6}\s*", "", line).strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（1 token ≈ 3.5 字符）。"""
    return int(len(text) / 3.5)


# ═══════════════════════════════════════════════════════════════
#  答案注入：从 solution 末尾识别最终答案，添加 \boxed{}
# ═══════════════════════════════════════════════════════════════

def _is_clean_answer(ans: str) -> bool:
    """判断提取的答案是否干净可注入（非截断、非噪声词、长度合理）。"""
    ans = ans.strip()
    if not ans or len(ans) > 100:
        return False
    # 不以操作符或逗号开头
    if ans[0] in ("+", ",", ".", ";", "=", "*", "/"):
        return False
    # 不含换行
    if "\n" in ans:
        return False
    # 排除只含噪声词的情况
    if _BOXED_NOISE_RE.fullmatch(ans.strip()):
        return False
    return True


def _try_inject_boxed(solution: str) -> Optional[str]:
    """尝试从 solution 末尾识别最终答案并注入 \\boxed{}。

    仅在可以高度置信地识别最终答案时才注入，否则返回 None。
    注入格式：在 solution 末尾添加 '\\n\\n\\boxed{answer}'。
    """
    if _BOXED_RE.search(solution):
        return None  # 已有 \boxed{}，不需要注入

    tail = solution[-600:]  # 只在末尾 600 字符中搜索

    # ── 优先级 1：结论句中的行内 LaTeX（取最后一个匹配，避免取到中间步骤）──
    # e.g. "Thus, the probability is $\frac{11}{16}$."
    matches = list(_CONCL_INLINE_RE.finditer(tail))
    if matches:
        m = matches[-1]  # 取最后一个结论句，更可能是最终答案
        ans = m.group(1).strip()
        if _is_clean_answer(ans):
            return solution.rstrip() + f"\n\n\\boxed{{{ans}}}"

    # ── 优先级 2：= VALUE 形式的行内 LaTeX（取最后一个匹配）──
    # e.g. "Hence, $x = 5$"
    matches = list(_EQ_INLINE_RE.finditer(tail))
    if matches:
        m = matches[-1]
        ans = m.group(1).strip()
        if _is_clean_answer(ans):
            return solution.rstrip() + f"\n\n\\boxed{{{ans}}}"

    # ── 优先级 3：最后一个展示公式块（$$ VALUE $$）──
    # e.g. "$$\n= \ln 3\n$$"
    m = _DISPLAY_FINAL_RE.search(solution)
    if m:
        ans = m.group(1).strip()
        # 去掉开头的 "= " 前缀（展示公式常以 = 开头）
        ans = re.sub(r"^=\s*", "", ans).strip()
        if _is_clean_answer(ans):
            return solution.rstrip() + f"\n\n\\boxed{{{ans}}}"

    # ── 优先级 4："Answer: X" 标注（取最后一个匹配）──
    matches = list(_ANSWER_LABEL_RE.finditer(tail))
    if matches:
        m = matches[-1]
        ans = m.group(1).strip()
        if _is_clean_answer(ans):
            return solution.rstrip() + f"\n\n\\boxed{{{ans}}}"

    # ── 优先级 5：最后一个行内 LaTeX，且前方 80 字符含结论词 ──
    inline_matches = list(re.finditer(r"\$([^$\n]{1,80})\$", tail))
    if inline_matches:
        last_m = inline_matches[-1]  # 取最后一个
        context_before = tail[max(0, last_m.start() - 80):last_m.start()]
        concl_words = re.compile(
            r"\b(thus|therefore|hence|so|answer|result|value|solution|equal|gives?)\b",
            re.IGNORECASE,
        )
        if concl_words.search(context_before):
            ans = last_m.group(1).strip()
            # 排除太长的表达式（可能是中间步骤）
            if _is_clean_answer(ans) and len(ans) <= 60:
                return solution.rstrip() + f"\n\n\\boxed{{{ans}}}"

    return None  # 无法高置信度识别最终答案



def check_layer1_boxed(solution: str, relaxed: bool = False) -> tuple[bool, str]:
    """L1: \\boxed{} 硬性要求（调用前已由 _try_inject_boxed 尝试注入）。
    - 必须存在 \\boxed{内容}
    - 严格模式：只允许 1 个（多个说明含多道题，结构混乱）
    - 宽松模式：允许最多 3 个，取最后一个验证
    - \\boxed{} 必须在 solution 后半段（位置 > 40%）
    """
    matches = list(_BOXED_FULL_RE.finditer(solution))
    if not matches:
        return False, "no_boxed"
    if not relaxed and len(matches) > 1:
        return False, f"multiple_boxed_{len(matches)}"
    if relaxed and len(matches) > 3:
        return False, f"too_many_boxed_{len(matches)}"
    # 检查 \boxed{} 内容非空
    last_match = matches[-1]
    content = last_match.group(1).strip()
    if not content:
        return False, "empty_boxed"
    if len(content) > 100:
        return False, "boxed_too_long"
    # \boxed{} 必须在后半段
    position_ratio = last_match.start() / max(len(solution), 1)
    if position_ratio < 0.35:
        return False, "boxed_too_early"
    return True, "ok"


def check_layer2_length(problem: str, solution: str) -> tuple[bool, str]:
    """L2: 长度过滤。"""
    if len(problem) < MIN_PROBLEM_LEN:
        return False, f"problem_too_short_{len(problem)}"
    if len(problem) > MAX_PROBLEM_LEN:
        return False, f"problem_too_long_{len(problem)}"
    if len(solution) < MIN_SOLUTION_LEN:
        return False, f"solution_too_short_{len(solution)}"
    if len(solution) > MAX_SOLUTION_LEN:
        return False, f"solution_too_long_{len(solution)}"
    total_chars = len(problem) + len(solution)
    if total_chars > MAX_TOTAL_CHARS:
        return False, f"total_too_long_{total_chars}"
    return True, "ok"


def check_layer3_math_density(solution: str) -> tuple[bool, str]:
    """L3: 数学内容密度——解答必须含足够的数学符号。"""
    latex_count = len(_LATEX_EXPR_RE.findall(solution))
    core_count  = len(_CORE_MATH_RE.findall(solution))
    if latex_count >= 3:
        return True, "ok"
    if core_count >= 2:
        return True, "ok"
    # 备选：含有基本算术运算符
    arith_count = len(re.findall(r"\d+\s*[\+\-\*\/\^=]\s*\d+", solution))
    if arith_count >= 4:
        return True, "ok"
    return False, f"low_math_density_latex={latex_count}_core={core_count}_arith={arith_count}"


def check_layer4_noise(solution: str) -> tuple[bool, str]:
    """L4: 噪声/截断检测。"""
    # 网页元数据关键词
    if _NOISE_RE.search(solution):
        kw = _NOISE_RE.search(solution).group(0)
        return False, f"noise_keyword:{kw[:30]}"
    # URL 过多
    url_count = len(_URL_RE.findall(solution))
    if url_count > 2:
        return False, f"too_many_urls_{url_count}"
    # 代码块为主（非数学代码）
    code_blocks = _CODE_BLOCK_RE.findall(solution)
    code_chars = sum(len(b) for b in code_blocks)
    if code_chars > len(solution) * 0.35:
        return False, "code_block_dominant"
    # 截断检测：solution 以逗号/介词/连词结尾
    last_50 = solution.rstrip()[-50:].lower()
    truncation_endings = (", ", " the ", " a ", " of ", " and ", " or ",
                          " in ", " to ", " for ", " is ", " are ", " be ")
    for ending in truncation_endings:
        if last_50.endswith(ending.rstrip()):
            return False, f"truncated_ending"
    return True, "ok"


def check_layer5_answer_completeness(solution: str) -> tuple[bool, str]:
    """L5: 答案完整性——\\boxed{} 必须在 solution 末尾附近。"""
    # \boxed{} 必须出现在最后 300 字符内
    tail = solution[-300:]
    if not _BOXED_RE.search(tail):
        return False, "boxed_not_at_end"
    # \boxed{内容} 不能为空
    m = list(_BOXED_FULL_RE.finditer(solution))
    if not m:
        return False, "no_boxed_content"
    last_content = m[-1].group(1).strip()
    if len(last_content) < 1:
        return False, "empty_boxed_content"
    return True, "ok"


def check_layer6_problem_independence(problem: str) -> tuple[bool, str]:
    """L6: 问题独立性——problem 须自包含，不能依赖外部上下文。"""
    problem_stripped = problem.strip()
    # 以指代词开头
    if _DANGLING_START_RE.match(problem_stripped):
        return False, "dangling_reference_start"
    # 含外部引用关键词
    if _EXTERNAL_REF_RE.search(problem_stripped):
        kw = _EXTERNAL_REF_RE.search(problem_stripped).group(0)
        return False, f"external_reference:{kw[:30]}"
    return True, "ok"


class Deduplicator:
    """L7: 基于指纹 + 2-gram 的去重器。"""

    def __init__(self, sim_threshold: float = 0.8):
        self.sim_threshold = sim_threshold
        self._fingerprints: set[str] = set()
        self._shingles_list: list[frozenset] = []

    @staticmethod
    def _fingerprint(problem: str) -> str:
        normalized = re.sub(r"\s+", " ", problem.strip().lower())
        return normalized[:120]

    @staticmethod
    def _shingles(text: str, k: int = 2) -> frozenset:
        tokens = re.sub(r"\s+", " ", text.strip().lower()).split()
        if len(tokens) < k:
            return frozenset(tokens)
        return frozenset(tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1))

    def _jaccard(self, a: frozenset, b: frozenset) -> float:
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0

    def is_duplicate(self, problem: str) -> bool:
        """返回 True 表示是重复，应丢弃。"""
        fp = self._fingerprint(problem)
        if fp in self._fingerprints:
            return True
        # 2-gram Jaccard
        sh = self._shingles(problem)
        for existing_sh in self._shingles_list:
            if self._jaccard(sh, existing_sh) >= self.sim_threshold:
                return True
        return False

    def add(self, problem: str) -> None:
        fp = self._fingerprint(problem)
        self._fingerprints.add(fp)
        self._shingles_list.append(self._shingles(problem))


def apply_quality_gates(
    problem: str,
    solution: str,
    deduplicator: Deduplicator,
    relaxed_boxed: bool = False,
) -> tuple[bool, str]:
    """依次执行七层质量门控，返回 (通过, 失败原因)。"""
    ok, reason = check_layer1_boxed(solution, relaxed=relaxed_boxed)
    if not ok:
        return False, f"L1:{reason}"
    ok, reason = check_layer2_length(problem, solution)
    if not ok:
        return False, f"L2:{reason}"
    ok, reason = check_layer3_math_density(solution)
    if not ok:
        return False, f"L3:{reason}"
    ok, reason = check_layer4_noise(solution)
    if not ok:
        return False, f"L4:{reason}"
    ok, reason = check_layer5_answer_completeness(solution)
    if not ok:
        return False, f"L5:{reason}"
    ok, reason = check_layer6_problem_independence(problem)
    if not ok:
        return False, f"L6:{reason}"
    # L7: 去重
    if deduplicator.is_duplicate(problem):
        return False, "L7:duplicate"
    return True, "ok"


# ═══════════════════════════════════════════════════════════════
#  提取策略（复用 stage4 已验证的正则逻辑）
# ═══════════════════════════════════════════════════════════════

def extract_s1(text: str) -> Optional[tuple[str, str]]:
    """S1: 同时有明确 Problem 和 Solution 标题。"""
    pm = _PROB_HEADER_RE.search(text)
    sm = _SOL_HEADER_RE.search(text)
    if not (pm and sm):
        return None
    problem  = _get_section_content(text, pm, len(pm.group(1)))
    solution = _get_section_content(text, sm, len(sm.group(1)))
    if problem and solution:
        return problem, solution
    return None


def extract_s1b(text: str) -> Optional[tuple[str, str]]:
    """S1b: 只有 Solution 标题，Solution 前最近一节作问题。"""
    sm = _SOL_HEADER_RE.search(text)
    if not sm:
        return None
    if _PROB_HEADER_RE.search(text):
        return None
    solution = _get_section_content(text, sm, len(sm.group(1)))
    before_sol = text[:sm.start()].strip()
    all_headings = list(_HEADING_RE.finditer(before_sol))
    if all_headings:
        last_h = all_headings[-1]
        problem = _get_section_content(before_sol, last_h, len(last_h.group(1)))
        if not problem.strip():
            problem = _strip_heading(last_h.group(0))
    else:
        problem = before_sol
    if problem and solution:
        return problem, solution
    return None


def extract_s2(text: str) -> Optional[tuple[str, str]]:
    """S2: 只有 Problem 标题，之后全文作解答。"""
    pm = _PROB_HEADER_RE.search(text)
    if not pm:
        return None
    if _SOL_HEADER_RE.search(text):
        return None
    prob_level = len(pm.group(1))
    problem = _get_section_content(text, pm, prob_level)
    stop_pat = re.compile(rf"^(?:#{{1,{prob_level}}})\s+", re.MULTILINE)
    m = stop_pat.search(text, pm.end())
    solution_start = m.start() if m else pm.end() + len(problem)
    solution = text[solution_start:].strip()
    if problem and solution:
        return problem, solution
    return None


def extract_s3(text: str) -> Optional[tuple[str, str]]:
    """S3: H1/H2 标题是疑问形式（含动词/数学符号），全文 body 作解答。"""
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
    """S3b: H1/H2 为数学主题词，正文含大量 LaTeX（教程型）。"""
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
    if len(_LATEX_EXPR_RE.findall(solution)) < 3:
        return None
    if title and solution:
        return title, solution
    return None


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
    next_bold  = re.search(r"^\*\*", after_q, re.MULTILINE)
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
        sol_end = am.end()
        next_sec = re.search(r"^\*\*[A-Z]", text[sol_end:], re.MULTILINE)
        solution = (text[sol_end:sol_end + next_sec.start()].strip()
                    if next_sec else text[sol_end:].strip())
    else:
        solution = text[qm.end():].strip()
    if problem and solution:
        return problem, solution
    return None


def extract_s4(text: str) -> Optional[tuple[str, str]]:
    """S4: 兜底——首段含问题关键词作问题，其余作解答。"""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) < 2:
        return None
    first = _strip_heading(paragraphs[0])
    if not (_Q_TITLE_VERB_RE.search(first) or _Q_TITLE_MATH_RE.search(first)):
        return None
    solution = "\n\n".join(paragraphs[1:])
    return first, solution


def extract_c_ex(text: str) -> Optional[tuple[str, str]]:
    """C-Ex: 从 ## Example 节提取演示型 QA（用于 C 类教程文档）。"""
    matches = list(_EX_HEADER_RE.finditer(text))
    if not matches:
        return None
    for ex_m in matches:
        ex_level = len(ex_m.group(1))
        ex_title = ex_m.group(2).strip()
        solution = _get_section_content(text, ex_m, ex_level)
        if len(solution) < MIN_SOLUTION_LEN:
            continue
        before = text[:ex_m.start()].strip()
        if not before:
            continue
        all_before_headings = list(_HEADING_RE.finditer(before))
        if all_before_headings:
            last_h = all_before_headings[-1]
            concept = _get_section_content(before, last_h, len(last_h.group(1)))
            concept_title = last_h.group(2).strip()
            problem = f"{concept_title}\n\n{concept}" if concept else concept_title
        else:
            problem = before
        # 若 Example 标题本身含问题动词，直接用标题作 problem
        if re.search(r"\b(find|prove|show|calculate|solve|evaluate|determine|compute)\b",
                     ex_title, re.I):
            problem = ex_title
        problem  = _clean_text(problem)
        solution = _clean_text(solution)
        if len(problem) >= MIN_PROBLEM_LEN and len(solution) >= MIN_SOLUTION_LEN:
            return problem, solution
    return None


def extract_c_thm(text: str) -> Optional[tuple[str, str]]:
    """C-Thm: 从 Theorem/Lemma + Proof 结构提取。"""
    thm_m = _THM_HEADER_RE.search(text)
    if not thm_m:
        return None
    thm_title = thm_m.group(2).strip()
    theorem   = _get_section_content(text, thm_m, len(thm_m.group(1)))
    prf_m = _PRF_HEADER_RE.search(text, thm_m.end())
    if not prf_m:
        return None
    proof = _get_section_content(text, prf_m, len(prf_m.group(1)))
    if not (theorem and proof):
        return None
    problem = _clean_text(f"{thm_title}\n\n{theorem}")
    solution = _clean_text(proof)
    if len(problem) >= MIN_PROBLEM_LEN and len(solution) >= MIN_SOLUTION_LEN:
        return problem, solution
    return None


# 策略优先级列表（按数据类型分组）
STRATEGIES_A = [
    ("S1",   extract_s1),
    ("S1b",  extract_s1b),
    ("S2",   extract_s2),
    ("S3",   extract_s3),
    ("S5",   extract_s5),
    ("S3b",  extract_s3b),
    ("S4",   extract_s4),
]
STRATEGIES_B = [
    ("S1",   extract_s1),
    ("S1b",  extract_s1b),
    ("S3",   extract_s3),
    ("S5",   extract_s5),
    ("S4",   extract_s4),
]
STRATEGIES_C = [
    ("S1",    extract_s1),
    ("S1b",   extract_s1b),
    ("S2",    extract_s2),
    ("S3",    extract_s3),
    ("C-Ex",  extract_c_ex),
    ("C-Thm", extract_c_thm),
    ("S5",    extract_s5),
]
STRATEGIES_D = [
    ("S1",    extract_s1),
    ("S1b",   extract_s1b),
    ("S2",    extract_s2),
    ("S3",    extract_s3),
    ("C-Ex",  extract_c_ex),
    ("C-Thm", extract_c_thm),
    ("S5",    extract_s5),
    ("S4",    extract_s4),
]
STRATEGY_MAP = {"A": STRATEGIES_A, "B": STRATEGIES_B, "C": STRATEGIES_C, "D": STRATEGIES_D}


# ═══════════════════════════════════════════════════════════════
#  主处理逻辑
# ═══════════════════════════════════════════════════════════════

def process_file(
    input_key: str,
    deduplicator: Deduplicator,
    dry_run: bool = False,
    relaxed_boxed: bool = False,
    limit: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """处理单个输入文件，返回 (kept_records, audit_records)。"""
    input_path = INPUT_FILES[input_key]
    strategies = STRATEGY_MAP[input_key]
    total_input = TOTAL_COUNTS[input_key]

    kept:  list[dict] = []
    audit: list[dict] = []

    # 统计各层过滤数
    layer_stats: dict[str, int] = {
        "no_extraction": 0,
        "L1": 0, "L2": 0, "L3": 0, "L4": 0,
        "L5": 0, "L6": 0, "L7": 0,
    }
    strategy_counts: dict[str, int] = {}
    injected_count = 0
    total = 0

    print(f"\n{'='*60}")
    print(f"处理 {input_key} 类: {input_path.name}  (共 {total_input:,} 条)")
    print(f"{'='*60}")

    with open(input_path, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            if limit and total >= limit:
                break
            record = json.loads(line)
            total += 1

            text = record.get("text", "")
            rid  = record.get("id", f"unk_{total}")

            # ── 提取阶段 ──
            extracted = None
            used_strategy = "SKIP"
            for strat_name, strat_fn in strategies:
                result = strat_fn(text)
                if result:
                    extracted = result
                    used_strategy = strat_name
                    break

            if extracted is None:
                layer_stats["no_extraction"] += 1
                if dry_run:
                    audit.append({"id": rid, "input": input_key,
                                  "strategy": "SKIP", "filter_reason": "no_extraction"})
                continue

            problem, solution = extracted
            problem  = _clean_text(problem)
            solution = _clean_text(solution)

            # ── 答案注入（若 solution 无 \boxed{} 则尝试从结尾识别并注入）──
            injected = False
            if not _BOXED_RE.search(solution):
                injected_sol = _try_inject_boxed(solution)
                if injected_sol:
                    solution = injected_sol
                    injected = True

            # ── 七层质量门控 ──
            passed, reason = apply_quality_gates(
                problem, solution, deduplicator, relaxed_boxed=relaxed_boxed
            )

            layer = reason.split(":")[0] if ":" in reason else reason
            if not passed:
                gate = reason.split(":")[0]
                layer_stats[gate] = layer_stats.get(gate, 0) + 1
                if dry_run:
                    audit.append({"id": rid, "input": input_key,
                                  "strategy": used_strategy, "filter_reason": reason,
                                  "problem_preview": problem[:80],
                                  "solution_tail": solution[-80:]})
                continue

            # ── 通过，加入结果 ──
            deduplicator.add(problem)
            strategy_counts[used_strategy] = strategy_counts.get(used_strategy, 0) + 1
            if injected:
                injected_count += 1

            entry = {
                "messages": [
                    {"role": "system",    "content": ""},
                    {"role": "user",      "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            }
            audit_entry = {
                **entry,
                "source_id": rid,
                "input_class": input_key,
                "extraction_strategy": used_strategy,
                "boxed_injected": injected,
                "char_counts": {"problem": len(problem), "solution": len(solution)},
                "boxed_count": len(_BOXED_RE.findall(solution)),
            }
            kept.append(entry)
            audit.append(audit_entry)

            if total % 5000 == 0:
                print(f"  [{input_key}] 已处理 {total:,} 条，保留 {len(kept):,} 条...",
                      flush=True)

    # ── 汇报 ──
    print(f"\n  [{input_key}] 总输入: {total:,} 条")
    print(f"  [{input_key}] 无法提取: {layer_stats['no_extraction']:,} 条")
    for layer_name in ("L1", "L2", "L3", "L4", "L5", "L6", "L7"):
        cnt = layer_stats.get(layer_name, 0)
        if cnt:
            print(f"  [{input_key}] {layer_name} 过滤: {cnt:,} 条")
    print(f"  [{input_key}] 最终保留: {len(kept):,} 条 ({len(kept)/max(total,1)*100:.1f}%)")
    print(f"  [{input_key}]   其中 boxed 注入: {injected_count:,} 条 / 原生含 boxed: {len(kept)-injected_count:,} 条")
    if strategy_counts:
        print(f"  [{input_key}] 策略分布: " +
              ", ".join(f"{k}={v}" for k, v in sorted(strategy_counts.items())))

    return kept, audit


def merge_outputs(output_dir: Path) -> None:
    """将 outputs/coldstart/ 下各类的 sft_train 文件合并为最终文件。"""
    per_class_files = list(output_dir.glob("coldstart_*_sft.jsonl"))
    if not per_class_files:
        print("未找到分类输出文件，请先运行各输入类型。")
        return

    all_records: list[dict] = []
    for f in sorted(per_class_files):
        count = 0
        with open(f, encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    all_records.append(json.loads(line))
                    count += 1
        print(f"  合并 {f.name}: {count:,} 条")

    # 全局去重（防止跨文件重复）
    dedup = Deduplicator()
    final: list[dict] = []
    for rec in all_records:
        problem = rec["messages"][1]["content"]
        if not dedup.is_duplicate(problem):
            dedup.add(problem)
            final.append(rec)

    with open(FINAL_OUTPUT, "w", encoding="utf-8") as fout:
        for rec in final:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n合并完成: {len(final):,} 条 → {FINAL_OUTPUT}")
    if len(all_records) > len(final):
        print(f"  跨文件去重移除: {len(all_records) - len(final):,} 条")


# ═══════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage6: 高质量冷启动 SFT 数据提取"
    )
    parser.add_argument(
        "--input", choices=["A", "B", "C", "D", "all"], default="A",
        help="处理哪类 stage3 数据（默认 A）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只统计，不写输出文件（快速验证过滤效果）"
    )
    parser.add_argument(
        "--relaxed-boxed", action="store_true",
        help="宽松模式：允许 solution 中最多 3 个 \\boxed{}（默认只允许 1 个）"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="每个输入文件最多处理多少条（调试用）"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="合并各类输出为最终 final_sft_train.jsonl"
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 仅合并模式 ──
    if args.merge:
        print("\n=== 合并模式 ===")
        merge_outputs(OUT_DIR)
        return

    # ── 确定处理哪些输入 ──
    keys_to_process = list(INPUT_FILES.keys()) if args.input == "all" else [args.input]

    # ── 共享去重器（跨类去重） ──
    deduplicator = Deduplicator()

    all_kept:  list[dict] = []
    all_audit: list[dict] = []

    for key in keys_to_process:
        kept, audit = process_file(
            key,
            deduplicator,
            dry_run=args.dry_run,
            relaxed_boxed=args.relaxed_boxed,
            limit=args.limit,
        )
        all_kept.extend(kept)
        all_audit.extend(audit)

        # ── 写分类输出（非 dry-run） ──
        if not args.dry_run and kept:
            sft_path   = OUT_DIR / f"coldstart_{key}_sft.jsonl"
            audit_path = OUT_DIR / f"coldstart_{key}_audit.jsonl"
            with open(sft_path, "w", encoding="utf-8") as f:
                for rec in kept:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(audit_path, "w", encoding="utf-8") as f:
                for rec in audit:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"\n  → 写入 {sft_path.name}: {len(kept):,} 条")
            print(f"  → 写入 {audit_path.name}: {len(audit):,} 条（含过滤记录）")

    # ── 总结 ──
    print(f"\n{'='*60}")
    print(f"全部完成  |  总保留: {len(all_kept):,} 条")
    if args.dry_run:
        print("  （dry-run 模式，未写入文件）")
    print(f"{'='*60}")

    # ── 若全量运行且多类，自动合并 ──
    if not args.dry_run and args.input == "all" and len(keys_to_process) > 1:
        print("\n自动合并所有分类输出...")
        merge_outputs(OUT_DIR)


if __name__ == "__main__":
    main()
