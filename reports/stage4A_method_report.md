# Stage 4A 处理方法报告

**处理对象**：stage3_A.jsonl（PROBLEM_SOLUTION 类，46,675 条）
**处理脚本**：`src/data/format_conversion.py`
**输出文件**：`stage4_output/stage4_A_sft.jsonl`（23,102 条 SFT 训练数据）
**完成日期**：2026-04-09

---

## 零、输入数据格式说明

### JSONL 文件结构

`stage3_A.jsonl` 每行是一个 JSON 对象，格式如下：

```json
{
  "id": "ad89af57-ee9d-456b-8...",
  "text": "# How to Solve the Equation $8z^3 + 1 = 0$\n\n## Problem Statement\n\nFind the cube roots...\n\n## Solution\n\nStep 1: ...",
  "metadata": {
    "finemath_scores": 4.19,
    "nemocurator_scores": 2.23,
    "category": "math",
    ...
  },
  "stage3_label": "A"
}
```

`text` 字段是核心内容——这是从数学网页爬取的**原始正文**，以 **Markdown** 格式存储。Stage 4A 的任务就是解析这个字段，把其中的问题和解答分离出来。

---

### Markdown 标题层级

Markdown 是一种纯文本格式语言，用 `#` 号表示标题级别，`#` 越少，级别越高：

```
# 一级标题（H1）     ← 最大，相当于整篇文章的大标题（通常只有一个）
## 二级标题（H2）    ← 相当于章节标题
### 三级标题（H3）   ← 相当于小节标题
#### 四级标题（H4）  ← 更细的子节
```

在我们的数据中，不同来源网页的标题层级用法差异很大：

| 数据来源类型 | 典型标题使用方式 |
|------------|---------------|
| 教育网站（Khan Academy 风格） | H1 = 页面标题，H2 = `Problem Statement` / `Solution` |
| 数学论坛（Stack Exchange 风格）| H1 = 帖子标题（即题目），H3 = 各回答者的解法 |
| 博客 / 笔记 | H1 = 文章标题，H2/H3 = 推导步骤 |
| 习题集页面 | H1 = 章节名，H2 = `Question 1` / `Question 2`... |

---

### 数据中常见的四种文档模式

**模式一：规范双标题（教育/学术网站）**
```markdown
# How to Solve the Equation $8z^3 + 1 = 0$    ← H1：页面标题

## Problem Statement                            ← H2：问题节
Find the cube roots of -1...

## Solution                                     ← H2：解答节
Step 1: Factor the expression...
Step 2: Apply De Moivre's theorem...
```

**模式二：论坛帖子（有问题标题，无解答标题）**
```markdown
# Sum of all 4-digit numbers with digits 2,4,6,8    ← H1：帖子标题（就是题目）

## Question                                          ← H2：提问正文
I realize digits can be chosen in 4! = 24 ways...

### Solution by emakarov                             ← H3：用户A的回答
Let the 24 numbers be $a_1, \dots, a_{24}$...

### Solution by another_user                         ← H3：用户B的回答
Alternatively, by symmetry...
```

**模式三：H1 即题目，无内部分节**
```markdown
# What is the total cost to heat the greenhouse?    ← H1：直接就是问题

The temperature is $F(t) = 22 + 20\cos(\pi t/12)$...
The heater kicks on when $F(t) \le 36$...
[整个正文既是背景说明也是解答，无进一步分节]
```

**模式四：无标题，用加粗文本代替结构**
```markdown
**Question**: Solve the equation $3x - 5 = 10$.

**Solution**: Add 5 to both sides: $3x = 15$. Divide by 3: $x = 5$.
```

这四种模式覆盖了绝大多数 A 类数据，也直接对应了后文 S1~S5 七种提取策略的设计依据。

---

## 一、任务目标

A 类数据来自 Stage 3 分类中被识别为"题目+解答"型的文档。这些文档均为 Markdown 格式的网页正文，包含数学问题及其解答，但**结构形式差异极大**——有的有规范的标题标注，有的只有题目标题，有的用加粗文本标注问题，有的则题目和解答连在一起没有任何结构标记。

目标是将这 46,675 条原始文本转换为标准 SFT chat 训练格式：

```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful mathematics assistant. ..."},
    {"role": "user",      "content": "<数学问题>"},
    {"role": "assistant", "content": "<解答步骤>"}
  ],
  "extraction_strategy": "S1/S1b/..."
}
```

---

## 二、数据结构分析

在设计提取策略之前，首先对 46,675 条数据的格式分布进行了统计分析：

| 格式特征 | 数量 | 占比 |
|---------|------|------|
| 同时有明确 `## Problem` 和 `## Solution` 标题 | 6,178 | 13.2% |
| 有明确 Problem/Question 标题 | 15,709 | 33.7% |
| 有明确 Solution/Answer 标题 | 8,949 | 19.2% |
| H1/H2 标题含疑问词或数学符号 | 38,950 | 83.4% |
| 无 H1 标题（以 H2/H3 开头） | 11,690 | 25.0% |
| 含多个 Q&A 对（Question 1/2/3...） | 2,328 | 5.0% |
| 含问题动词（find/prove/calculate...） | 41,024 | 87.9% |

关键发现：
- **83.4% 的文档** H1/H2 标题本身就是问题陈述，是最广泛适用的提取方式
- **只有 13.2%** 同时具备规范的双标题结构，是最"干净"但覆盖率最低的情形
- 不同格式之间存在大量重叠，须按优先级依次尝试

各提取策略与文档模式的对应关系：

| 策略 | 利用的标题信号 | 对应文档模式 |
|------|-------------|------------|
| S1  | H2 同时有 `Problem` + `Solution` | 模式一 |
| S1b | H2 只有 `Solution` | 模式一变体（问题未单独标题） |
| S2  | H2 只有 `Problem/Question` | 模式二（论坛帖子，无解答标题） |
| S3  | H1/H2 标题本身含疑问词或数学符号 | 模式三 |
| S3b | H1/H2 是数学主题词，正文含大量 LaTeX | 模式三变体 |
| S4  | 无标题，首段含问题关键词 | 模式三/四的退化情形 |
| S5  | 正文中 `**Question**:` 加粗标注 | 模式四 |

---

## 三、提取策略设计

根据数据结构多样性，设计了 7 种提取策略，**按优先级从高到低**依次尝试，命中即停止：

### S1：明确双标题（最高优先级）

**适用场景**：文档同时含有 `## Problem`/`## Question` 和 `## Solution`/`## Answer` 标题。

**提取方式**：
- 用正则定位两个标题的位置
- 提取各自标题下的内容（到同级或更高级标题为止）
- 问题 = Problem 节内容，解答 = Solution 节内容

**命中数**：3,826 条（8.2%）

**示例**：
```
## Problem Statement
Consider three lengths a, b, c chosen randomly...

## Solution
Assume lengths from uniform distribution on [0,1]...
```

---

### S1b：只有 Solution 标题

**适用场景**：有 `## Solution` 标题，但无对应的 `## Problem` 标题。

**提取方式**：
- 在 Solution 标题之前，找距离最近的子标题（`## Problem #7` 等）
- 提取该子标题下的内容作为问题
- 若无子标题，取 Solution 之前的全部内容作为问题

**命中数**：1,815 条（3.9%）

**早期问题**：最初直接将 Solution 之前的全部文本作为问题，导致问题部分包含多个不相关题目的描述（文档含多道题，但只有最后一题的 Solution）。修复方法是改为取 Solution 之前**最近**的子标题节，精确定位到配对问题。

---

### S2：只有 Problem 标题

**适用场景**：有 `## Problem`/`## Question` 标题，但无 Solution 标题。

**提取方式**：
- 提取 Problem 节内容作为问题
- Problem 节之后的全部内容作为解答（通常是回答者的推导过程）

**命中数**：4,877 条（10.4%）

**示例**：论坛帖子式文档，用户贴出题目，下面是多个回答者的解法。

---

### S3：问题式标题

**适用场景**：H1/H2 标题本身是一个数学问题（含疑问词、动词或数学表达式）。

**识别条件**（满足其一）：
- 标题含数学符号：`$...$`、`\frac`、`^`、`_{` 等 LaTeX
- 标题含问题动词：find / prove / show / calculate / evaluate / solve / determine / simplify / integrate / differentiate / rationalize / help / homework / questions / problems 等

**提取方式**：标题文本 = 问题，标题之后全部内容 = 解答

**命中数**：6,787 条（14.5%）

**迭代过程**：初版动词列表较短（仅 10 个词），导致大量含 "homework"、"problems"、"questions" 等词的合法题目标题未被识别。扩展动词列表后命中量从约 3,400 增至 6,787。

---

### S3b：数学主题词标题

**适用场景**：H1/H2 标题不含疑问词，但是数学领域主题词（如 "Probability of Getting a Full House"、"Writing a Matrix as a Product of Elementary Matrices"），且正文含有丰富的数学内容。

**识别条件**：
- 标题含数学主题词：algebra / calculus / probability / theorem / integral / matrix / eigenvalue / fourier / differential 等（共 35 个词）
- 非课程目录类（排除含 chapter / unit / module / overview / syllabus 的标题）
- 正文含 ≥3 处 LaTeX 表达式（`$...$` 或 `$$...$$`）

**提取方式**：标题 = 问题，标题后全文 = 解答

**命中数**：2,816 条（6.0%）

---

### S4：首段问题式

**适用场景**：无法从标题结构识别，但第一个非空段落本身包含问题关键词或数学表达式。

**提取方式**：
- 取第一段（去掉 Markdown 标题符号）作为问题
- 其余段落合并作为解答

**命中数**：1,440 条（3.1%）

---

### S5：加粗标注格式

**适用场景**：正文中使用 `**Question**:` / `**Problem**:` 等加粗标签明确标注问题。

**提取方式**：
- 用正则定位 `**Question**:` 行，提取行内及随后内容作为问题
- 若有对应 `**Solution**:` / `**Answer**:` 标签，提取其后内容作解答
- 否则取问题标注之后全部内容作解答

**命中数**：1,541 条（3.3%）

**示例**：
```
**Question**: Solve the equation $3x - 5 = 10$.

**Solution**: Add 5 to both sides: $3x = 15$. Divide by 3: $x = 5$.
```

---

## 四、质量过滤

每条提取结果在写入输出前须通过以下三项质量检查：

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| 问题最短长度 | 30 chars | 过短的"问题"通常是标题残留或噪声 |
| 解答最短长度 | 100 chars | 保证解答有实质内容 |
| 解答含数学内容 | 至少 1 处 | LaTeX 命令、数学公式或算术表达式 |

最终 **100%** 的保留记录均含数学内容（LaTeX 或算术表达式）。

---

## 五、遇到的问题与解决方法

### 问题 1：S1b 提取到多道题的混合内容

**现象**：部分文档含多个 `## Problem #N` 小节，只有最后一个有 `## Solution`。S1b 原始实现将 Solution 之前所有内容（多道题）都作为问题，导致"问题"文本混乱冗长。

**解决**：改为在 Solution 之前的内容中，找**最近一个子标题**，只取该子标题下的段落作为问题，与 Solution 精准配对。

---

### 问题 2：S3 动词列表覆盖不足

**现象**：大量含 "homework"、"questions"、"problems" 等词的合法问题标题（如 "Homework Help: Calculus"）未被 S3 识别，约有 4,400 条本可挽回的记录被丢弃。

**解决**：将动词/关键词列表从 10 个扩展至 20 个，增加 rationalize / homework / questions / problems / solutions / exercises / help / urgently 等常见问题标题词，命中量从 ~3,400 增至 6,787。

---

### 问题 3：大量记录既无问题标题也无解答标题

**现象**：约 61% 的被跳过记录（~15,000 条）无任何结构标题可用，S1–S4 均无法命中。分析发现这类文档多为课程单元、习题表单、导航式页面，本身不适合作 Q&A 对。

**解决**：新增 S5 策略，专门捕获用 `**Question**:` / `**Problem**:` 加粗格式标注的问题，从中挽回 1,541 条高质量数据。剩余无法提取的记录（约 23,573 条）确认为结构不适合或质量不足，合理丢弃。

---

## 六、最终结果

| 策略 | 描述 | 数量 | 占比 |
|------|------|------|------|
| S1  | 明确双标题 | 3,826 | 8.2% |
| S1b | 只有 Solution 标题 | 1,815 | 3.9% |
| S2  | 只有 Problem 标题 | 4,877 | 10.4% |
| S3  | 问题式 H1/H2 标题 | 6,787 | 14.5% |
| S3b | 数学主题词标题 + 丰富 LaTeX | 2,816 | 6.0% |
| S4  | 首段含问题关键词 | 1,440 | 3.1% |
| S5  | `**Question**:` 加粗标注 | 1,541 | 3.3% |
| 丢弃 | 无法提取或质量不足 | 23,573 | 50.5% |
| **合计保留** | | **23,102** | **49.5%** |

**质量指标**：
- 解答含数学内容率：**100%**
- 问题中位长度：**86 chars**
- 解答中位长度：**1,650 chars**
- 解答长度范围：100 ~ 47,723 chars

---

## 七、后续建议

1. **多 QA 对文档**（约 5%，2,328 条）目前整体作一个 QA 对处理，可进一步拆分为多个独立训练样本，估计可多获得 5,000~8,000 条。
2. **剩余 23,573 条跳过记录**中约有 10~15% 含完整 Q&A 内容但结构特殊（如嵌套列表式题目），可通过更细粒度的解析规则继续挽回。
3. **解答过长的样本**（最长 47,723 chars）在训练时可能超过模型 context window，建议在训练前按 token 长度截断或过滤（>9,012 tokens 的记录）。
