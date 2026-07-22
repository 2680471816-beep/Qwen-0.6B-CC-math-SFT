# Stage 4B 处理方法报告

**处理对象**：stage3_B.jsonl（FORUM_QA 类，641 条）
**处理脚本**：`src/data/stage4B_conversion.py`
**输出文件**：`stage4_output/stage4_B_sft.jsonl`（422 条 SFT 训练数据）
**完成日期**：2026-04-09

---

## 零、输入数据说明

### JSONL 文件结构

每行一个 JSON 对象，与 A 类格式相同：

```json
{
  "id": "9f35e5bf-3261-...",
  "text": "# Hypergeometric Distribution\n\n**Thread: ...**\n\n**1. October 7th, 2009 #1**\n**Walcott89**\n...",
  "metadata": { "finemath_scores": 4.12, "nemocurator_scores": 2.51, ... },
  "stage3_label": "B"
}
```

B 类数据的 `text` 字段来自**数学论坛网页**，内容是完整的帖子讨论串，包含提问帖和若干回复帖，通常带有大量论坛特有的格式标记（用户元数据、引用块、时间戳等）。

### B 类的分类标准（Stage 3 中确定）

Stage 3 分类时，B 类（FORUM_QA）的判定规则为：

> 文档中出现 ≥2 处论坛特征词：`Posted by` / `Reply #N` / `Thread starter` / `Member Posts` / `Junior Member` / `Joined` 等

这意味着 B 类文档**一定来自论坛**，但不代表结构统一——不同论坛（Math Help Forum、Stack Exchange、Brilliant.org、个人博客论坛等）的页面渲染格式差异很大，转为 Markdown 后呈现出截然不同的帖子分隔方式。

---

## 一、数据结构分析

### 1.1 格式特征统计

对 641 条 B 类数据进行格式特征扫描：

| 格式特征 | 数量 | 占比 |
|---------|------|------|
| 含 `### H3` 标题（用户名/帖子分隔） | 366 | 57.1% |
| 含 Post/帖子标记（Post 1、#1 等） | 372 | 58.0% |
| 含明确 `## Problem/Question` 节 | 135 | 21.1% |
| 含明确 `## Solution/Answer` 节 | 50 | 7.8% |
| 含多楼回复（Post 2/3 以上） | 84 | 13.1% |
| 含用户元数据（Joined/Posts/Member 等） | 322 | 50.2% |
| 含引用块（`>` 开头） | 31 | 4.8% |
| 以 H1 标题开头 | 533 | 83.2% |

文本长度：p10 = 1,441 chars，p50 = 2,577 chars，p90 = 7,440 chars，最长 47,693 chars。

### 1.2 四种典型文档模式

通过对数据的逐条分析，识别出以下四种主要结构模式：

---

**模式一：H3 标题为用户名（最常见，约 57%）**

来自 Math Help Forum、Purple Math 等论坛，每位用户的帖子以 `### 用户名` 开头：

```markdown
# Hypergeometric Distribution                   ← H1：帖子主题（即数学问题）

### Walcott89 (Newbie)                           ← H3：提问者
I would appreciate help with these two problems:
1. If 7 cards are dealt from 52...
2. A committee of size 3 from 4 doctors...

### Soroban (Super Member)                       ← H3：回复者1
Hello Walcott89! For problem 1:
$\binom{12}{2}\binom{40}{5} / \binom{52}{7}$...

### Plato (MHF Contributor)                      ← H3：回复者2
For problem 2, let X = number of doctors...
```

**模式二：数字加粗分隔符（约 5%）**

来自部分论坛爬取后的格式，用 `**1. 日期 #1**` 标记每楼：

```markdown
# Prime Number Problem

**1. Nov 1st 2007, 04:35 AM #1**
**Revilo**

Let $n$ be a positive integer. Show that there is a prime $p$ with $p | n$ and $p \le \sqrt{n}$.

**2. Nov 1st 2007, 05:28 AM #2**
**angel.white**

Okay, $n$ is not prime, so it has a divisor other than itself...
```

**模式三：明确 Problem/Solution 节（约 21%）**

部分论坛（如 Homework Help 类网站）在爬取时被整理为规范格式：

```markdown
# Geometry: Tangent and Secant

### Problem Statement
Given: Line $t$ tangent to circle $O$ at point $P$, secant $l \parallel t$...
To Prove: Chords $AB$ and $BP$ are congruent.

### Proof
1. By tangent-secant theorem...
2. Since $l \parallel t$, arc $AP$ = arc $BP$...
```

**模式四：加粗用户名分隔或纯文本问答（约 8%）**

部分页面无标准 Markdown 标题，用加粗文本或 `**Question**` / `**Answer**` 作为结构标记：

```markdown
**Question**
Two men David and Clifton and their wives Kim and Allison go shopping...
If each couple spent the same two-digit sum, who is Kim's husband?

**Answer by bonanova**
For some arrangement of couples, let David buy $d$ books...
$d^2 + k^2 = c^2 + a^2$ where $a=1$...
```

---

## 二、问答对构建策略

### 核心思路

B 类数据的 Q&A 构建原则与 A 类不同：

- **A 类**：文档本身就是一道题的完整叙述，问题和解答在同一页面
- **B 类**：文档是一个讨论串，**第一楼 = 提问**，**后续楼层 = 解答**

因此，提取策略的核心是**识别楼层分隔符**，将第一楼内容作为 user 侧输入，后续楼层合并作为 assistant 侧输出。

### 五种提取策略（按优先级）

---

#### B1：H3 标题为用户名分隔（命中 258 条，40.2%）

**适用场景**：文档中 `### 用户名` 是每楼的分隔标记（模式一）。

**识别方式**：找到所有 H3 标题，检查第一个 H3 标题长度 ≤80 字符（用户名通常较短）。

**提取方式**：
1. 提取第一个 H3 节的内容作为问题
2. 若第一节内容过短（通常只有元数据），尝试将 H3 之前的 preamble（帖子主题描述）与第一节合并
3. 第二个 H3 节起的所有内容合并为解答

**示例**：
```
Q: I'm supposed to find the area between $y = x^{1/3}$ and the x-axis on $[-1, 8]$.
   I divided into two regions: $A_1 = [-1,0]$, $A_2 = [0,8]$. I found $A_2 = 12$
   and $A_1 = 3/4$, but my teacher got a different answer. What am I doing wrong?

A: You'll notice your function is symmetric about the origin, so
   $A_1 = \int_0^1 x^{1/3} dx = \frac{3}{4}$ as you say.
   Further, $A_2 = \int_0^8 x^{1/3} dx = 12$.
   The total area is $A_1 + A_2 = 12.75$.
```

---

#### B2：数字加粗分隔符（命中 15 条，2.3%）

**适用场景**：`**1. 日期 #1**` 格式标记每楼（模式二）。

**识别方式**：正则匹配 `**N. ... #N**` 或 `**N.**` 格式，找到 ≥2 个分隔符。

**提取方式**：第 1 个分隔符后的内容为问题，第 2 个及以后的内容合并为解答。

---

#### B3：明确 Problem/Solution 节（命中 14 条，2.2%）

**适用场景**：文档有规范的 `## Problem Statement` 和后续回复节（模式三）。

**提取方式**：
- 问题 = `## Problem` 节内容
- 若有 `## Solution`/`## Proof` 节，取该节内容为解答
- 否则取 Problem 节之后的全部内容为解答

---

#### B4：兜底策略（命中 122 条，19.0%）

**适用场景**：无论坛特有的多楼结构，文档整体类似 A 类（H1/H2 标题即问题，正文即解答）。这类记录虽被 Stage 3 分类为 B 类，但实际上是单帖提问+回复内嵌在同一段落的形式。

**提取方式**：
- 若 H1/H2 标题是纯导航词（Thread/Discussion/Forum），则取标题之后第一个实质段落作为问题
- 否则直接用 H1/H2 标题作为问题，全文 body 作为解答

---

#### B5：加粗标注格式（命中 13 条，2.0%）

**适用场景**：使用 `**Question**` / `**Answer**` 加粗标注，或 `**用户名**` 加粗行分隔（模式四）。

**提取方式**（按子模式依次尝试）：
1. 若有 `**Question**` + `**Answer**` 标注，分别提取两节内容
2. 若有 `**用户名**` 独行加粗分隔，取第一用户名后的内容为问题，其余为解答

---

### 清洗处理

每条提取出的帖子内容都经过以下清洗：

| 清洗操作 | 说明 | 正则示例 |
|---------|------|---------|
| 删除用户元数据行 | 去掉 `Joined/Posts:/Status:/Location:` 等行 | `^(Joined\|Posts:).*$` |
| 删除引用块 | 去掉 `> Quote Originally Posted by...` 引用 | `^>.*$` |
| 压缩多余空行 | 连续 3 行以上空行 → 2 行 | `\n{3,}` → `\n\n` |

### 质量过滤

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| 问题最短长度 | 30 chars | 过短说明提取到纯元数据或空节 |
| 解答最短长度 | 80 chars | B 类解答通常比 A 类短，略微放宽 |
| 全文含数学内容 | 至少 1 处 | 问题或解答任一处含 LaTeX/算术表达式 |

---

## 三、遇到的问题与解决方法

### 问题 1：B1 策略的第一节内容为纯元数据

**现象**：部分论坛帖中，第一个 `### 用户名` 节只包含用户基本信息（注册时间、帖子数、头衔等），实质的问题陈述在这一节之前（即 H1 标题下的 preamble）或与元数据混在同一节中。原始实现直接取第一节内容，导致提取到的"问题"只有几十字符的元数据，触发最短长度过滤被丢弃。

**解决**：修改 B1 提取逻辑，当第一节内容长度 < 30 chars 时：
- 优先用 H3 之前的 preamble（帖子主题描述区域）作为问题
- 若 preamble 也不够，将 preamble 与第一节合并

修复后，有 129 条原本因"问题过短"被丢弃的记录得以保留。

---

### 问题 2：大量记录无多楼分隔符

**现象**：约 101 条记录（15.8%）B1~B4 均无法命中，诊断发现这类文档用 `**Question**`/`**Answer**` 加粗文本或 `**用户名**` 独行加粗作为分隔，既无 H3 标题也无数字编号。

**解决**：新增 B5 策略，专门处理两种加粗分隔格式，从中挽回 13 条高质量记录。剩余 219 条确认为结构过于碎片化（多楼交叉引用、纯讨论无解答等），合理丢弃。

---

### 问题 3：B2 问题节含用户名残留

**现象**：数字加粗分隔格式（`**1. #1**`）后紧跟 `**用户名**` 行，提取出的问题开头出现 `**Walcott89**` 这样的用户名，不适合直接作为训练数据。

**解决**：`_clean_post()` 函数的元数据过滤正则覆盖了独立的加粗用户名行（匹配 `^**短用户名**` 模式），在清洗步骤中自动去除。

---

## 四、最终结果

| 策略 | 描述 | 数量 | 占比 |
|------|------|------|------|
| B1 | H3 用户名分隔 | 258 | 40.2% |
| B2 | 数字加粗分隔 | 15 | 2.3% |
| B3 | 明确 Problem/Solution 节 | 14 | 2.2% |
| B4 | 兜底（H1/H2 标题作问题） | 122 | 19.0% |
| B5 | 加粗标注格式 | 13 | 2.0% |
| 丢弃 | 结构碎片化或无数学内容 | 219 | 34.2% |
| **合计保留** | | **422** | **65.8%** |

**质量指标**：
- 解答含数学内容率：**100%**
- 问题中位长度：**232 chars**（比 A 类 86 chars 更长，因论坛提问通常有完整背景描述）
- 解答中位长度：**1,965 chars**（比 A 类 1,650 chars 更长，来自多位回复者）
- 解答长度范围：80 ~ 23,500 chars

---

## 五、B 类与 A 类的对比

| 维度 | A 类（PROBLEM_SOLUTION） | B 类（FORUM_QA） |
|------|------------------------|----------------|
| 数据来源 | 教育网站、习题集、博客 | 数学论坛讨论串 |
| 核心结构 | 单文档含题目+解答 | 多楼帖：第一楼提问，后续楼回答 |
| 提取难点 | 题目/解答边界不统一 | 楼层分隔符格式多样 |
| 问题长度 | 较短（中位 86 chars） | 较长（中位 232 chars，含背景） |
| 解答来源 | 单一（文档作者） | 多位回复者，内容更丰富 |
| 保留率 | 49.5% | 65.8% |
| 噪声类型 | LaTeX 分隔符不规范 | 用户元数据、引用块、时间戳 |
