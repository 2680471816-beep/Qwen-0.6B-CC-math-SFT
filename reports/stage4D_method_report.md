# Stage 4D 处理方法报告：D 类教材数据（TEXTBOOK）

**日期**：2026-04-10
**作者**：Member A（数据处理）
**脚本**：`src/data/stage4D_conversion.py`
**输入**：`stage3_output/stage3_D.jsonl`（151 条）
**输出**：`stage4_output/stage4_D_sft.jsonl`（55 条）

---

## 零、输入数据说明

### 0.1 JSONL 字段结构

每条记录为 JSON 格式，核心字段如下：

```json
{
  "id": "e3f7a2...",
  "text": "# Numbers Algebra More Algebra...\n\n## 1 Numbers\n...",
  "metadata": {
    "finemath_int_scores": 4,
    "finemath_scores": 4.19,
    "category": "math"
  }
}
```

`text` 字段为网页正文，使用 Markdown 格式，平均长度约 4,570 字符（最短 1,057，最长 43,319）。

### 0.2 D 类（TEXTBOOK）分类标准

Stage 3 分类器将以下特征的文档归入 D 类：
- 内容来自教材正文、教材题解页、教材出版物介绍
- 包含系统性定义、定理、章节体系或教材习题编号
- 与 A 类（独立题目）不同，D 类通常以"章节内容"为粒度，内容更结构化

D 类共 **151 条**，全部含有数学内容（LaTeX 或算术表达式），134 条（88.7%）含有 Markdown 标题。

---

## 一、数据结构分析

### 1.1 文档类型分布

经人工抽样和规则分析，151 条 D 类数据分为两大类：

| 类型 | 条数（估算） | 说明 |
|------|-------------|------|
| **可提取内容（真实教材）** | ~104 条（68.9%） | 含教材正文、定义、例题、习题解答 |
| **低价值文档** | ~47 条（31.1%） | 书目/ISBN页、整除数字页、纯目录页 |

**可提取文档细分：**

| 子类型 | 条数（估算） | 说明 |
|--------|-------------|------|
| 教材正文讲解 | ~40 条 | 含概念+例题结构，C-Ex 可提取 |
| 教材题解页 | ~22 条 | 含 STEP-BY-STEP SOLUTION 或 Problem/Solution 标注 |
| 段落内嵌示例 | ~33 条 | "Example N:" 嵌在段落中，非标题行 |
| NCERT 习题解答 | ~6 条 | 印度国家课程框架，含标准题目+解答 |
| 选择题格式 | ~3 条 | 含 A/B/C/D 选项 + Answer 解析 |

**低价值文档细分：**

| 子类型 | 条数 | 特征 |
|--------|------|------|
| 书目/ISBN 页 | ~40 条 | 含 ISBN 编号、出版社、作者列表，无解答内容 |
| 整除/因数数字页 | ~4 条 | "Divisors of N: [列表]" 格式，无学习价值 |
| 纯目录页 | ~3 条 | 仅列出 Chapter 1/2/3... 无内容 |

### 1.2 四种典型文档模式

**模式一：教材正文（概念讲解 + 标题式 Example 节）**

```
## Sets
Sets are a fundamental concept...

### Definitions
- **Empty Set**: ...
- **Subset**: ...

### Example 1
Let $A = \{1,2,3\}$ and $B = \{3,4,5\}$. Then:
- $A \cup B = \{1,2,3,4,5\}$
- $A \cap B = \{3\}$
```

→ **C-Ex 策略可提取**：以 "Sets / Definitions" 节作为问题，Example 节内容作为解答。

---

**模式二：教材题解页（PROBLEM + STEP-BY-STEP SOLUTION）**

```
# Hutchinson's Basic Mathematical Skills

### Chapter 7: Problem 73X1

**PROBLEM:**
A railing for a deck requires pieces of cedar 4 ft 8 in., ...

**STEP-BY-STEP SOLUTION:**
1. Convert all measurements to inches:
   - 4 ft 8 in. = 4×12+8 = 56 inches
   ...
```

→ **D-Step 策略**（新策略）：匹配 `**PROBLEM:**` 和 `**STEP-BY-STEP SOLUTION:**` 关键词，分别提取问题和解答。

---

**模式三：段落内嵌示例（"Example N:" 格式，非标题行）**

```
## 1.1 Numbers and Bases

You've grown up with the decimal system...

Example: Convert $D6$ to binary.

$D_h = 13$ in decimal which is $8 + 4 + 1 = 1101_b$.

Example: Convert $197$ to binary and hex.
Ignoring the remainders...
```

→ **D-Inl 策略**（新策略）：匹配段落中 "Example:" 格式引导的块，以前一个示例（含上下文）作为问题，后续解答段落作为解答。

---

**模式四：选择题 + 答案解析**

```
# Properties of Sets

## If $A = \{7,8,9\}$, then relation $R = \{(8,9)\}$ in $A$ is:

- **A.** Symmetric only
- **B.** Symmetric and transitive only
- **C.** Transitive only
- **D.** Equivalence

**Answer:** Option C is correct.
**Hint:** To determine the properties...
```

→ **D-Q 策略**（新策略）：匹配标题形式的问题 + 选项行 + Answer/Hint 结构，构建选择题 QA 对。

---

**模式五：书目/ISBN 页（低价值，直接丢弃）**

```
# Algebra and Trigonometry, Fourth Edition

**Authors**: Judith A. Beecher; Judith A. Penna
**Publisher**: Pearson
**ISBN-13**: 978-0-321-69398-3

## Price Information
- **Our Price**: $85.99
```

→ **is_low_value() 函数**：检测到 ISBN 且文档长度 < 4,000 chars，直接丢弃，不进入策略流程。

---

## 二、问答对构建策略

D 类处理脚本复用了 A/C 类全部策略，并新增 3 种 D 专属策略，按优先级依次执行：

### 2.1 完整策略表

| 优先级 | 策略名 | 类型 | 命中数 | 说明 |
|--------|--------|------|--------|------|
| 1 | **S1** | 复用 | 0 | 同时含 ## Problem + ## Solution 标题 |
| 2 | **S1b** | 复用 | 5 | 仅有 ## Solution 标题，取前节内容作问题 |
| 3 | **S2** | 复用 | 8 | 仅有 ## Problem/Question 标题，后续全文作解答 |
| 4 | **S3** | 复用 | 0 | H1/H2 标题本身是问题句（含动词/数学表达式） |
| 5 | **C-Ex** | 复用 | 25 | `## Example` 节提取演示型 QA（最大命中） |
| 6 | **S3b** | 复用 | 0 | 标题为数学主题词，后续全文作解答 |
| 7 | **C-Thm** | 复用 | 0 | Theorem + Proof 标题对提取 |
| 8 | **D-Step** | **新增** | 5 | `**PROBLEM:**` + `**STEP-BY-STEP SOLUTION:**` 格式 |
| 9 | **D-Q** | **新增** | 1 | 选择题 + Answer/Hint 解析格式 |
| 10 | **D-Inl** | **新增** | 11 | 段落内嵌 `Example N:` 块（非标题行） |
| 11 | **S5** | 复用 | 0 | `**Question**:` / `**Answer**:` 加粗标注 |
| 12 | **S4** | 复用 | 0 | 首段含问题关键词，首段作问题 |

### 2.2 D-Step 策略详解

**适用场景**：教材习题解答网站（如 Chegg、Bartleby）将题目格式化为 `**PROBLEM:**` + `**STEP-BY-STEP SOLUTION:**` 的固定结构。

**核心正则**：
```python
_STEP_PROB_RE = re.compile(
    r"\*{0,2}PROBLEM[\*:]{0,3}\s*\n([\s\S]*?)(?=\*{0,2}STEP|SOLUTION)",
    re.IGNORECASE,
)
_STEP_SOL_RE = re.compile(
    r"\*{0,2}STEP-BY-STEP\s+SOLUTION[\*:]{0,3}\s*\n([\s\S]+?)(?=\n#{1,4}\s|\Z)",
    re.IGNORECASE,
)
```

**示例输出**：

> **[USER]** Construction Problem: A railing for a deck requires pieces of cedar 4 ft 8 in., 11 ft 7 in., and 9 ft 3 in. long. What is the total length of material that is needed?
>
> **[ASST]** 1. Convert all measurements to inches for consistency: - 4 ft 8 in. = 4×12+8 = 56 inches ...

### 2.3 D-Q 策略详解

**适用场景**：NCERT 等印度课程教材的选择题页，题目以 `## <问题>` 标题形式出现，选项为 `- **A.**` 格式列表，末尾含 `**Answer:**` 和 `**Hint:**`。

**提取逻辑**：
1. 正则匹配标题（问题）+ 选项列表 + Answer 关键词；
2. 提取答案文本；
3. 若存在 `**Hint:**` 段落，追加为解释，丰富解答内容；
4. 构建：问题 = 标题 + 选项；解答 = 正确答案 + Hint 解析。

**示例输出**：

> **[USER]** If $A = \{7,8,9\}$, then the relation $R = \{(8,9)\}$ in $A$ is:
> - **A.** Symmetric only
> - **B.** Symmetric and transitive only
> - **C.** Transitive only
> - **D.** Equivalence
>
> **[ASST]** Option C is correct. **Explanation:** To determine the properties of the given relation, we will check whether it is symmetric, transitive, or an equivalence relation...

### 2.4 D-Inl 策略详解

**适用场景**：教材正文中，示例以段落行 `Example N:` 或 `Example:` 引导，而非 Markdown 标题行，C-Ex 策略无法匹配（C-Ex 依赖 `^#{1,4}\s+Example` 标题正则）。

**提取逻辑**：
1. 匹配非标题行中的 `Example N:` 引导块；
2. 以第一个 Example 块（含引导文字和内联解答）作为问题；
3. 寻找其后的 `Solution:` / `Step N:` 段落；若无，则取紧接的 1-2 段作为解答；
4. 通过质量门控（问题 ≥ 30 chars，解答 ≥ 100 chars，含数学内容）过滤。

**示例输出**：

> **[USER]** Convert $D6$ to binary. $D_h = 13$ in decimal which is $8+4+1 = 1101_b$. $6_h = 6$ in decimal which is $4+2 = 0110_b$. So $D6_h = 1101\ 0110_b$. The easiest way to convert from decimal into binary...
>
> **[ASST]** Convert $197$ to binary and hex. Ignoring the remainders $197 \div 2 = 98$, $98 \div 2 = 49$...

### 2.5 低价值文档快速丢弃（is_low_value）

在进入策略流程前，`is_low_value()` 函数对以下类型文档**直接丢弃**，避免无意义的策略尝试：

| 检测规则 | 条数 | 说明 |
|---------|------|------|
| 含 ISBN 且长度 < 4,000 chars | ~40 条 | 书目/定价/购买信息页 |
| 含 "Divisors of N" / "List of positive divisors" | ~4 条 | 因数列表，无学习价值 |
| 含 "Table of Contents" + 章节列表，无 solution/example | ~3 条 | 纯目录页 |
| 含 "uploaded by" / "related interests" 且长度 < 3,000 chars | 0 条 | Scribd 嵌入页 |

共快速丢弃 **47 条**，占总输入的 31.1%。

### 2.6 质量门控

所有策略提取结果均经过三项质量过滤：

| 过滤条件 | 阈值 | 目的 |
|---------|------|------|
| 问题最小长度 | ≥ 30 chars | 排除空标题、单词问题 |
| 解答最小长度 | ≥ 100 chars | 排除"见图"等无实质性解答 |
| 数学内容存在性 | 至少 1 处 LaTeX / 算术表达式 | 确保内容与数学相关 |

---

## 三、遇到的问题与解决方法

### 问题 1：D-Step 提取到截断句作为问题

**现象**：Record 0（大型教材文档）中，正文里有一句 "...because you have no more symbols left. What do you do?"，其中 "do you do?" 被 `_STEP_PROB_RE` 误匹配为 `PROBLEM:` 后的内容，导致问题文本以小写字母 "because" 开头，是截断句。

**原因**：`_STEP_PROB_RE` 的宽松版正则（无 `^` 行首锚定）在长文档中可能错误命中段落内部的 "PROBLEM" 字样。

**解决**：在 `extract_d_step()` 中添加截断句检测——若提取的问题文本第一个字符为小写字母，说明是从句子中间截断，直接返回 `None`：

```python
# 截断句检测：problem 以小写字母开头，说明被错误截断，丢弃
if problem and problem[0].islower():
    return None
```

修复后 D-Step 丢弃 1 条误匹配，共保留 5 条高质量结果。

---

### 问题 2：C-Ex 无法匹配段落内嵌 Example（非标题行）

**现象**：D 类文档中约 33 条记录的示例以 `Example N:` 出现在正文段落内（非 Markdown 标题行），C-Ex 策略的正则 `^#{1,4}\s+Example` 依赖行首 `#` 标题标记，无法命中。

**原因**：D 类教材比 C 类文章更常见非标题化示例，因为许多教材原始格式（PDF/Word）转 Markdown 后示例块保持段落形式。

**解决**：新增 **D-Inl 策略**，专门匹配段落内嵌的 `Example N:` 格式：

```python
pattern = re.compile(
    r"(?m)^(?!#).*?(?:Example\s*\d*[\.\:])\s*(.+?)(?=\n\n(?!Example)|\Z)",
    re.DOTALL,
)
```

`(?!#)` 排除标题行，`(?=\n\n(?!Example)|\Z)` 以双空行或文档末尾为段落边界。D-Inl 最终命中 11 条。

---

### 问题 3：大量书目/ISBN 页无法提取任何有效 QA

**现象**：D 类中约 40 条记录为教材书目介绍页（含 ISBN、出版社、价格等），这类页面结构化但无任何解答内容，所有 12 种策略均命中失败，导致大量无意义策略调用。

**原因**：Stage 3 分类器识别到 "textbook" 相关关键词（书名、ISBN、教材术语）将其归为 D 类，但此类页面本身不含教学内容。

**解决**：在进入策略流程前增加 `is_low_value()` 快速检测函数，对含 ISBN 且长度较短的文档直接丢弃，避免无效策略调用：

```python
def is_low_value(text: str) -> bool:
    if _DIVISOR_RE.search(text):
        return True
    if re.search(r"isbn[\s\-]*(?:10|13)?[\s\-:]", text, re.IGNORECASE):
        if len(text) < 4000:
            return True
    ...
```

共快速丢弃 47 条，处理效率大幅提升。

---

## 四、最终结果

### 4.1 总体统计

| 项目 | 数值 |
|------|------|
| 总输入 | 151 条 |
| 低价值快速丢弃 | 47 条（31.1%） |
| **保留（SFT 数据）** | **55 条（36.4%）** |
| 跳过（无可提取结构） | 96 条（63.6%） |
| 平均问题长度 | 709 chars |
| 平均解答长度 | 1,383 chars |

### 4.2 策略命中分布

| 策略 | 命中数 | 占保留比例 | 类型 |
|------|--------|-----------|------|
| C-Ex | 25 | 45.5% | 复用（最大贡献） |
| D-Inl | 11 | 20.0% | **新增** |
| S2 | 8 | 14.5% | 复用 |
| S1b | 5 | 9.1% | 复用 |
| D-Step | 5 | 9.1% | **新增** |
| D-Q | 1 | 1.8% | **新增** |

C-Ex 策略（来自 C 类）贡献了 D 类 45.5% 的保留量，说明 D 类教材与 C 类文章在 Example 节结构上高度相似，策略复用效果显著。

### 4.3 输出格式

每条保留记录为标准 SFT chat 格式（JSONL），含 `extraction_strategy` 字段便于溯源：

```json
{
  "id": "e3f7a2...",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful mathematics assistant. When given a math problem, provide a clear, step-by-step solution showing all reasoning and calculations."
    },
    {
      "role": "user",
      "content": "Venn Diagrams\n\nVenn diagrams are a useful tool for visualizing the relationships..."
    },
    {
      "role": "assistant",
      "content": "1. Let $A = \\{1,2,3\\}$ and $B = \\{3,4,5\\}$. Then:\n   - $A \\cup B = \\{1,2,3,4,5\\}$..."
    }
  ],
  "source_id": "e3f7a2...",
  "extraction_strategy": "C-Ex"
}
```

---

## 五、A/B/C/D 四类对比

| 维度 | A 类 | B 类 | C 类 | D 类 |
|------|------|------|------|------|
| 数据量 | 46,675 | 641 | 27,682 | 151 |
| 主要格式 | 独立题目+解答 | 论坛多帖 | 知识文章+示例 | 教材正文/题解页 |
| 低价值比例 | ~5% | ~10% | ~5% | **31.1%（书目页）** |
| 主力策略 | S1（明确Q&A标题） | B1（H3用户分隔） | C-Ex（Example节） | C-Ex（复用） |
| 新增策略 | — | B1-B5 | C-Ex、C-Thm | D-Step、D-Q、D-Inl |
| 保留率 | 49.5% | 65.8% | 50.8% | **36.4%（最低）** |
| 保留条数 | 23,102 | 422 | 14,064 | **55** |

D 类保留率最低（36.4%），主要原因是 31.1% 的记录为书目/ISBN 页，属于 Stage 3 分类误判导致的噪声。真实教材内容的保留率（剔除低价值后）约为：55 / 104 ≈ **52.9%**，与 A/C 类接近。

---

## 六、数据汇总

| 来源 | 输入 | 保留 | 保留率 | 输出文件 |
|------|------|------|--------|---------|
| A 类 PROBLEM_SOLUTION | 46,675 | 23,102 | 49.5% | `stage4_A_sft.jsonl` |
| B 类 FORUM_QA | 641 | 422 | 65.8% | `stage4_B_sft.jsonl` |
| C 类 ARTICLE_TUTORIAL | 27,682 | 14,064 | 50.8% | `stage4_C_sft.jsonl` |
| D 类 TEXTBOOK | 151 | 55 | 36.4% | `stage4_D_sft.jsonl` |
| **合计** | **75,149** | **37,643** | **50.1%** | — |

**最终 SFT 训练数据集：37,643 条**，可用于 Qwen3-0.6B-Base 的后训练微调。
