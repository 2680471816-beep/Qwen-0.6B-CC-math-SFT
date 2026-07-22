# Stage 5 Pipeline 技术说明报告

**文件**：`src/data/stage5_pipeline.py`
**日期**：2026-04-15
**作者**：数据处理模块（Member A）

---

## 一、背景与目标

Stage 4 通过规则从 stage3 分类结果中提取了 37,643 条 SFT 数据，但质量存在根本性缺陷：

- A 类文档中仅 **3.4%** 的答案包含 `\boxed{}`，评测时无法提取最终答案
- 全部数据无 `<think>...</think>` 推理链，模型无法学习 thinking 模式
- 部分答案含原始网页噪声（Comments、References、截断句）

Stage 5 的目标是：**从 stage3 分类结果出发，用本地 vLLM（Qwen3-0.6B-Base）重新生成高质量 reasoning-SFT 数据**，使每条数据满足：

```
<think>
详细推理过程（≥8句，覆盖列式、化简、验证）
</think>
简洁正式解答。

\boxed{最终答案}
```

实际测试通过率：**~44%**，预期全量产出：**~41,000 条 reasoning-SFT 样本**。

---

## 二、输入数据

### 2.1 数据来源

Pipeline 读取 Stage 3 分类后的四个文件：

| 文件 | 条数 | 文档类型 | 处理路线 |
|------|------|---------|---------|
| `stage3_output/stage3_A.jsonl` | 46,675 | PROBLEM_SOLUTION（题目+解答型） | Route A |
| `stage3_output/stage3_B.jsonl` | 641 | FORUM_QA（论坛问答帖） | Route A |
| `stage3_output/stage3_C.jsonl` | 27,682 | ARTICLE_TUTORIAL（教程/教材型） | Route A 或 Route B（见分类原则） |
| `stage3_output/stage3_D.jsonl` | 151 | TEXTBOOK（教材型） | Route A |
| **合计** | **75,149** | — | — |

### 2.2 单条记录格式

```json
{
  "id": "<uuid>",
  "text": "<数学相关网页正文，Markdown格式>"
}
```

---

## 三、文档分类原则（C 类内部分流）

A、B、D 类文档格式明确（本身含题目和解答结构），直接走 **Route A**。

C 类（ARTICLE_TUTORIAL）文档内容差异较大，需进一步判断。

### 3.1 判断函数：`d_looks_like_concept_doc(title, text)`

**正向信号（有以下任意一项则倾向为概念文档）**：

- 标题含关键词：`lesson`、`understanding`、`how to`、`guide`、`fraction`、`equation`、`graph`、`probability` 等
- 正文前 900 字符含关键词：`objective`、`big idea`、`definition`、`properties`、`vocabulary`、`examples` 等

**负向信号（有以下任意一项则否定）**：

- 标题以 `?` 结尾且长度 > 40（通常是问题帖而非教程）
- 正文含 `problem statement`
- 正文含 `## solution` 但无 `objective`/`big idea`

**分流结果**：

```
stage3_C.jsonl (27,682条)
    ├─ 概念/教程文档（约 18,000 条）→ Route B
    └─ 含题目结构的文档（约 9,682 条）→ Route A
```

---

## 四、Route A：结构提取 + LLM 增强

**适用文档**：stage3_A、stage3_B、stage3_D，以及 C 类中含题目结构的文档。

### 4.1 完整处理流程

```
原始文档 text
    │
    ▼
Step 1: 提取 source_unit（问题+解答单元）
    ├─ 按 Markdown 模式匹配（7种正则）：
    │     ## Problem Statement...## Solution
    │     ## Interview Question...## Answer
    │     **Question:**...**Answer:**
    │     ## Example...## Solution  等
    ├─ 匹配成功 → 截取前 1400 字符作为 source_unit
    └─ 匹配失败 → 取文档前 1400 字符作为 fallback
    │
    ▼
Step 2: 问题提取
    ├─ 显式抽取（d_extract_explicit_question）：
    │     扫描 **Question:、**Problem: 等标记行
    │     或逐行寻找以 ? 结尾、长度 12~220 的句子
    │     跳过含答案特征的行（= 数字、\frac{}{}、\boxed{} 等）
    ├─ 抽取到问题后检查：含 \boxed{} 或 "is $X$?" 格式 → 拒绝
    └─ 失败则 LLM 反向重构（LLM #1）：
          prompt_d_backtranslate_question(title, source_unit)
          → 从原文反推学生最可能提出的问题
          → 输出一个以问号结尾的具体问题
    │
    ▼
Step 3: 答案生成（LLM #2）
    prompt_d_answer_from_doc(question, source_unit)
    max_tokens=600，temperature=0.0
    生成后经 d_clean_answer() 处理：
    - 截断 ## Q / ## Question 后的段落溢出
    - 清除 <think>...</think> 残留和 prompt 泄漏词
    - 规范化 \boxed{}（保留最后一个）
    - 失败则运行 _rescue_boxed()（7种模式兜底提取答案）
    - _rescue_boxed 也失败则 _add_boxed_via_llm() 重试一次
    │
    ▼
Step 4: 规则质量门控（Layer 1，零 GPU）
    d_rule_flags(question, answer) 检查：
    - missing_boxed：answer 不含 \boxed{}
    - prompt_leak：answer 含 "instruction"、"source unit" 等泄漏词
    - too_short：answer 长度 < 50 字符
    - thin_reasoning：换行数 < 1 且句号数 < 1
    - question_copied：answer 几乎只是复制了问题
                       （answer 含 \boxed{} 时豁免此检查）
    有任意 flag → 标记 keep=False，跳过后续 LLM 步骤
    │
    ▼
Step 5: 推理链生成（LLM #3，仅 quality.keep=True 时执行）
    prompt_d_long_reasoning(question, source_unit)
    max_tokens=1500，temperature=0.0
    要求：
    - 只输出推理过程，不输出最终答案，不加 <think> 标签
    - 至少写出 8 句有效推理
    - 覆盖列式、化简、验证（计算题）或定义、关系、结论（概念题）
    │
    ▼
Step 6: 推理链验证（d_reasoning_flags）
    - reasoning_too_short：推理链 < 180 字符
    - reasoning_prompt_leak：含泄漏词
    - reasoning_too_thin：换行 < 2 且句号 < 4
    - reasoning_too_long：推理链 > 7000 字符
    通过后组装：
    "<think>{reasoning}</think>\n{answer}"
```

### 4.2 Prompt 函数

**LLM #1 — `prompt_d_backtranslate_question(title, source_unit)`**

```
你正在把数学预训练原文重构成监督微调数据。
给定一段已经轻清洗的数学原文，请恢复一个学生最自然会提出的问题。
要求：
1. 只输出一个问题，不要回答。
2. 这个问题必须能由原文片段直接回答。
3. 如果原文里本来就有题目，优先保留该题目的自然表述。
4. 问题要具体，不能写成空泛概念句。
5. 只输出问题句，以问号结尾。

标题：{title}

原文片段：
{source_unit}
```

**LLM #2 — `prompt_d_answer_from_doc(question, source_unit)`**

```
你正在把数学预训练原文整理成高质量推理型 SFT 数据。
请根据问题和原文片段，写出忠实、清晰、可教学的解答。
要求：
1. 只使用原文片段可支持的信息。
2. 用 2 到 4 个自然步骤写出关键推理。
3. 不要输出「根据原文」「文档里」等元话语。
4. 最后一行必须单独写成 \boxed{最终答案}。

问题：
{question}

原文片段：
{source_unit}
```

**LLM #3 — `prompt_d_long_reasoning(question, source_unit)`**

```
你正在为数学监督微调数据补写高质量推理过程。
请只根据题目和原文片段，写出详细但紧凑的推理草稿。
要求：
1. 只输出推理过程，不要输出最终答案，不要加 <think> 标签。
2. 推理需要比普通解答更完整，尽量写出关键中间步骤或关键概念连接。
3. 不要提到「原文片段」「题目要求我」等元话语。
4. 不要写标题，不要写编号模板外壳。
5. 对计算题尽量覆盖列式、化简、检查；对概念题尽量覆盖定义、关系、结论。
6. 至少写出 8 句有效推理，必要时可更长。
7. 目标长度明显长于普通答案。

题目：
{question}

原文片段：
{source_unit}
```

### 4.3 预期产出

| 输入 | 文档数 | 预期通过率 | 预期条数 |
|------|--------|-----------|---------|
| stage3_A | 46,675 | ~44% | ~20,540 |
| stage3_B | 641 | ~44% | ~282 |
| stage3_D | 151 | ~44% | ~66 |
| C 类非概念文档 | ~9,682 | ~44% | ~4,260 |

---

## 五、Route B：知识点扩展

**适用文档**：C 类中被判定为概念/教程型的文档（约 18,000 条）

### 5.1 设计思路

概念文档本身没有明确题目，Route B 改为**围绕知识点构造学生问题**，再生成答案，每篇文档产出 2 条 QA。

### 5.2 完整处理流程

```
原始文档 text
    │
    ▼
Step 1: 提取知识点和概念摘录
    knowledge_point = 文档 H1 标题
    source_excerpt = d_extract_concept_excerpt(text, max_chars=900)
        扫描段落，优先保留含以下关键词的段落：
        objective / big idea / definition / example /
        vocabulary / properties / strategy
        不足则取前 3 段
    │
    ▼
Step 2: 自指令生成 2 个学生问题（LLM #1）
    prompt_d_self_instruct(knowledge_point, source_excerpt, count=2)
    max_tokens=180，temperature=0.2
    解析输出：d_parse_instruction_list(raw, limit=2)
    失败时 fallback：
    - "What is the key idea behind {knowledge_point}?"
    - "How would you explain {knowledge_point} with one short worked example?"
    │
    ▼（以下对每个问题分别执行）
Step 3: 答案生成（LLM #2）
    prompt_d_answer_from_knowledge(question, knowledge_point, source_excerpt)
    max_tokens=600，temperature=0.0
    同样经过 d_clean_answer()、_rescue_boxed()、_add_boxed_via_llm() 处理
    │
    ▼
Step 4~6: 与 Route A 相同
    规则门控 → 推理链生成 → 推理链验证
    source_unit 用 source_excerpt 代替
```

### 5.3 Prompt 函数

**LLM #1 — `prompt_d_self_instruct(knowledge_point, source_excerpt, count=2)`**

```
你正在围绕一个数学知识点构造学生问题，用于监督微调数据。
请生成 2 个不同但同知识点的学生问题。
要求：
1. 每行只写一个问题。
2. 问题必须自包含，不能依赖外部上下文。
3. 至少有一个问题偏概念解释，至少有一个问题偏简单应用。
4. 不要写答案，不要写编号标题之外的解释。

知识点：
{knowledge_point}

参考摘录：
{source_excerpt}
```

**LLM #2 — `prompt_d_answer_from_knowledge(question, knowledge_point, source_excerpt)`**

```
你正在把数学知识点目录扩展成高质量 reasoning-SFT。
请根据给定知识点和参考摘录，回答学生问题。
要求：
1. 可以沿用摘录中的定义、公式或解法模板，但不要切换到无关知识。
2. 回答要像老师在讲解，保留必要推理步骤。
3. 最后一行必须单独写成 \boxed{最终答案或最终结论}。

知识点：
{knowledge_point}

问题：
{question}

参考摘录：
{source_excerpt}
```

### 5.4 预期产出

- 约 18,000 篇概念文档，每篇产出 2 个问题
- 预期通过率 ~44%，产出 **~15,840 条**

---

## 六、三层质量控制详细说明

### 6.1 Layer 1：规则门控（零 GPU）

函数：`d_rule_flags(question, answer)`

| Flag | 触发条件 | 说明 |
|------|---------|------|
| `missing_boxed` | answer 不含 `\boxed{}` | 评测时无法提取答案 |
| `prompt_leak` | answer 含泄漏词 | 模型把 prompt 内容输出到答案里 |
| `too_short` | `len(answer) < 50` | 答案内容过少 |
| `thin_reasoning` | 换行数 < 1 且句号数 < 1 | 答案没有任何推理结构 |
| `question_copied` | answer 几乎只复制问题且无 `\boxed{}` | 模型没有实质作答 |

> **设计说明**：`question_copied` 在 answer 含 `\boxed{}` 时豁免，因为"The result is X.\n\n\boxed{X}"这类答案在 `\boxed{}` 存在时是合法数据，即使开头复述了问题。

### 6.2 Layer 2（已移除）：原 LLM 质量评分

原设计中 Layer 2 调用 `prompt_d_quality_judge` 让模型对 groundedness / reasoning / pedagogy / format 四个维度打分。

**移除原因**：Qwen3-0.6B-Base 为基座模型，无法稳定输出 `groundedness=X` 格式，解析失败时所有分数默认为 1、keep=False，导致约 **39% 的误杀**。移除后通过率从 ~20% 提升到 ~44%，质量由规则层和推理链验证层充分保障。

### 6.3 Layer 3：推理链验证

函数：`d_reasoning_flags(text)`

| Flag | 触发条件 |
|------|---------|
| `reasoning_too_short` | 推理链 < 180 字符 |
| `reasoning_prompt_leak` | 含 "source unit"、"instruction" 等泄漏词 |
| `reasoning_too_thin` | 换行 < 2 且句号 < 4 |
| `reasoning_too_long` | 推理链 > 7000 字符（超出模型上下文限制） |

### 6.4 `\boxed{}` 兜底机制

当答案生成后不含 `\boxed{}` 时，依次尝试：

**① `_rescue_boxed(text)`**：用 7 种正则模式从答案末尾提取最终答案：

1. 行末 `$$ ... $$` 显示数学
2. `= <value>` 结尾
3. `is/are <number>` 结尾
4. `**value**` 粗体结尾
5. 行末行内数学 `$expr$`
6. `answer/result/value is/= <anything>`
7. 末行独立数字

**② `_add_boxed_via_llm(question, answer)`**：如果 rescue 也失败，调用 LLM 重新识别最终答案并补充 `\boxed{}`。

---

## 七、筛选与合并机制

### 7.1 单路线运行后自动输出

每路线运行完成后，自动从 raw 文件读取并过滤，输出对应的 sft_train 文件：

```python
# 筛选条件
quality.keep == True AND reasoning_quality.keep == True
```

只保留 `messages` 字段，去除所有审计元数据。

### 7.2 `merge_all()` 跨路线合并

`--merge` 命令执行以下流程：

```
所有 route_a_*.raw.jsonl（A/B/D 各自独立文件）
route_a_from_c.raw.jsonl（C类非概念文档）
route_b.raw.jsonl（C类概念文档）
    │
    ▼ 过滤：quality.keep=True AND reasoning_quality.keep=True
    │
    ▼ 按 quality.score 降序排列（质量高的优先保留）
    │
    ▼ d_dedup_rows(threshold=0.82)
        - 精确归一化匹配（问题文本标准化后完全相同）
        - Jaccard 相似度 ≥ 0.82 视为语义重复
    │
    ├─ route_a.sft_train.jsonl   （Route A 通过样本）
    ├─ route_b.sft_train.jsonl   （Route B 通过样本）
    └─ final_reasoning_sft_train.jsonl  ★ 最终训练集
```

**merge 从 raw 而非 sft_train 读取的原因**：
- raw 文件保留 `quality.score`，合并时可按质量排序，去重时优先保留高分条目
- 可随时调整过滤阈值重新生成最终训练集，无需重跑 GPU pipeline

### 7.3 最终训练集格式

```json
{
  "messages": [
    {"role": "system",    "content": ""},
    {"role": "user",      "content": "题目"},
    {"role": "assistant", "content": "<think>\n推理...\n</think>\n解答\n\n\\boxed{答案}"}
  ]
}
```

---

## 八、输出文件说明

```
outputs/full_pipeline/
├── route_a_A.raw.jsonl           # --input A 全量审计
├── route_a_B.raw.jsonl           # --input B 全量审计
├── route_a_D.raw.jsonl           # --input D 全量审计
├── route_a_A.sft_train.jsonl     # --input A 过滤后训练集
├── route_a_B.sft_train.jsonl     # --input B 过滤后训练集
├── route_a_D.sft_train.jsonl     # --input D 过滤后训练集
├── route_a_from_c.raw.jsonl      # C 类非概念文档审计
├── route_a_from_c.sft_train.jsonl
├── route_b.raw.jsonl             # C 类概念文档审计
├── route_b.sft_train.jsonl
├── route_a.sft_train.jsonl       # merge 后 Route A 汇总（去重）
├── route_b.sft_train.jsonl       # merge 后 Route B 汇总（去重）
└── final_reasoning_sft_train.jsonl  ★ 最终训练集（跨路线去重）
```

> 各路线使用独立文件名（`route_a_A`、`route_a_B`、`route_a_D`），分开运行时不会互相覆盖。

**raw.jsonl 审计字段**：

```json
{
  "messages": [...],
  "route": "A",
  "source_id": "原始文档 uuid",
  "source_unit": "用于生成的原文片段",
  "quality": {
    "score": 0,
    "keep": true,
    "rule_flags": [],
    "judge": {"groundedness": 0, "reasoning": 0, "pedagogy": 0, "format": 0,
              "keep": true, "reason": "rule-only gate (LLM judge disabled)"}
  },
  "reasoning_quality": {
    "keep": true,
    "flags": [],
    "think_length": 520
  }
}
```

---

## 九、预期产出汇总

| 路线 | 文档数 | 预期通过率 | 预期条数 |
|------|--------|-----------|---------|
| Route A（A+B+D） | ~47,467 | ~44% | ~20,900 |
| Route A（C 类非概念） | ~9,682 | ~44% | ~4,260 |
| Route B（C 类概念，每文档2题） | ~18,000 × 2 | ~44% | ~15,840 |
| **合计（去重前）** | | | **~41,000** |

---

## 十、模型调用方式

### 10.1 vLLM 服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Qwen3-0.6B-Base \
    --served-model-name Qwen3-0.6B \
    --port 8888 \
    --max-model-len 9012 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16
```

### 10.2 `chat()` 核心调用

```python
def chat(prompt, max_tokens=240, temp=0.0):
    response = get_client().chat.completions.create(
        model="Qwen3-0.6B",
        temperature=temp,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {"role": "system", "content": "只输出任务要求的内容..."},
            {"role": "user",   "content": prompt},
        ],
    )
    text = response.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()
```

| 参数 | 值 | 原因 |
|------|-----|------|
| `enable_thinking=False` | 关闭 | 工具调用阶段不需要 think 块 |
| `temperature=0.0` | 确定性 | 答案生成、推理链生成均用 0 |
| `temperature=0.2` | 轻微随机 | 自指令生成需要多样性 |

### 10.3 各步骤 token 预算

| 步骤 | max_tokens | 说明 |
|------|-----------|------|
| 问题反向重构（LLM #1） | 160 | 只输出一个问题 |
| 答案生成（LLM #2） | 600 | 含推理步骤的完整解答 |
| boxed 兜底补充 | 700 | 重写完整解答 |
| 推理链生成（LLM #3） | 1500 | 详细推理草稿 |
| 自指令生成（Route B LLM #1） | 180 | 输出 2 个问题 |

---

## 十一、多线程机制

使用 `ThreadPoolExecutor`，每个线程处理一整篇文档的全部 LLM 调用。

- **Route A**：每篇最多 3 次 LLM 调用（问题重构 + 答案 + 推理链）
- **Route B**：每篇最多 5 次（自指令 + 2×（答案 + 推理链））
- **断点续传**（`--resume`）：启动时从 raw 文件加载已处理 ID，跳过重复文档
- **默认 workers=4**，全量运行建议 `--workers 8`

---

## 十二、快速使用参考

```bash
# Step 1: 启动 vLLM（tmux 中）
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Qwen3-0.6B-Base \
    --served-model-name Qwen3-0.6B \
    --port 8888 --max-model-len 9012 \
    --gpu-memory-utilization 0.5 --dtype bfloat16

# Step 2: 小样本测试（先验证质量）
python src/data/stage5_pipeline.py --input A --limit 100
python src/data/stage5_pipeline.py --input C --limit 100

# Step 3: 全量分开运行（各自独立文件，不互相覆盖）
python src/data/stage5_pipeline.py --input A --workers 8
python src/data/stage5_pipeline.py --input B --workers 8
python src/data/stage5_pipeline.py --input D --workers 8
python src/data/stage5_pipeline.py --input C --workers 8

# Step 4: 跨路线合并去重，生成最终训练集
python src/data/stage5_pipeline.py --merge

# 中断后续传
python src/data/stage5_pipeline.py --input A --workers 8 --resume
```
