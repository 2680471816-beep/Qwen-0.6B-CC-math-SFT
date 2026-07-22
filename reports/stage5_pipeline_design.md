# Stage 5 数据重洗 + Reasoning-SFT Pipeline 设计文档

**版本**：v1.0
**日期**：2026-04-12
**依赖**：Stage 1~3 输出（stage3_A/B/C/D.jsonl）、本地 vLLM 服务（Qwen3-0.6B）

---

## 一、为什么重洗

Stage 4 的规则提取虽得到 37,643 条数据，但存在三类根本性缺陷：

| 缺陷 | 现象 | 对训练的影响 |
|------|------|------------|
| **答案格式不对齐** | 几乎无 `\boxed{}`（A类仅 3.4%，C类 0.2%） | 模型学不到答案定位格式，评测时无法提取答案 |
| **无推理过程** | 无 `<think>` 块 | 无法学习 thinking 模式，推理能力弱 |
| **答案质量参差** | 部分答案含"Comments and References"尾部噪声、截断句 | 引入错误示范，SFT 反而降低能力 |

Stage 4 的提取价值在于识别文档类型和边界——这部分不需要重做（Stage 3 分类结果保留）。
重洗的起点是 **stage3_output/**，用 LLM 重新生成高质量的答案和推理链。

---

## 二、训练方法与数据格式的关系

### 2.1 训练目标

使用 **SFT（Supervised Fine-Tuning）** 方法，在 Qwen3-0.6B-Base 上微调，
目标是让模型学会：
1. 先输出 `<think>...</think>` 内的完整推理链
2. 再输出简洁的正式答案，末尾以 `\boxed{最终答案}` 收尾

### 2.2 最终训练数据格式（严格固定）

```json
{
  "messages": [
    {"role": "system",    "content": ""},
    {"role": "user",      "content": "题目（自然语言 + LaTeX）"},
    {"role": "assistant", "content": "<think>\n详细推理过程...\n关键步骤...\n验证...\n</think>\n简洁正式解答。\n\n\\boxed{最终答案}"}
  ]
}
```

**关键约束说明：**
- `system.content` 必须为空字符串 `""`（与 Qwen3 官方推荐对齐）
- `<think>` 块内容：至少 8 句有效推理，覆盖列式、化简、验证
- `\boxed{}` 必须在 assistant content 末尾，且只能出现一次（最终答案）
- 每条 token 总长度控制在 **9,012 tokens** 以内（模型推理上限）

### 2.3 Qwen3 Chat Template 适配

Qwen3 的 apply_chat_template 在 SFT 时会将上述格式渲染为：

```
<|im_start|>system
<|im_end|>
<|im_start|>user
题目<|im_end|>
<|im_start|>assistant
<think>
推理...
</think>
答案

\boxed{X}<|im_end|>
```

训练时 loss 只计算 assistant 部分（即 `<|im_start|>assistant` 之后）。

### 2.4 自我蒸馏（Self-Distillation）预留设计

**第一轮 SFT 完成后**，可以进行自我蒸馏：

```
第一轮 SFT 模型
    ↓ 对训练集问题重新生成答案（temperature=0.7，采样多条）
    ↓ 保留质量高于原始答案的生成结果
    ↓ 筛选后作为第二轮 SFT 的训练数据
    ↓ 第二轮 SFT 模型（更强）
```

为支持自我蒸馏，每条训练数据需额外保存：
- `source_id`：原始文档 id（用于追溯）
- `route`：处理路线（A/B/C）
- `quality_score`：质量评分（用于后续过滤）

这些字段在训练时过滤掉，但保存在 audit 文件中。

---

## 三、数据来源与路线规划

### 3.1 输入数据

| 文件 | 条数 | 内容类型 |
|------|------|---------|
| `stage3_A.jsonl` | 46,675 | 题目+解答型文档（PROBLEM_SOLUTION） |
| `stage3_B.jsonl` | 641 | 论坛问答帖（FORUM_QA） |
| `stage3_C.jsonl` | 27,682 | 教程文章型文档（ARTICLE_TUTORIAL） |
| `stage3_D.jsonl` | 151 | 教材型文档（TEXTBOOK） |
| **合计** | **75,149** | — |

### 3.2 三条处理路线

#### Route A：结构提取 + LLM 增强（适用 A/B/D 类）

适用文档：本身含有明确的题目和解答结构。

```
原始文档 (text)
    │
    ├─ 规则提取候选单元
    │   ├─ d_extract_problem_solution_unit()
    │   └─ 按 A 类 S1~S5 七种策略识别问题边界
    │
    ├─ 问题提取
    │   ├─ 显式抽取：d_extract_explicit_question()
    │   └─ 反向重构：prompt_d_backtranslate_question()  ← LLM #1
    │
    ├─ 答案生成（LLM 重新生成，确保 \boxed{} 格式）
    │   └─ prompt_d_answer_from_doc(q, source_unit)      ← LLM #2
    │
    ├─ 规则质量门控：d_rule_flags()
    │   过滤：missing_boxed / prompt_leak / too_short / thin_reasoning
    │
    ├─ LLM 质量评分：prompt_d_quality_judge(q, a, source_unit) ← LLM #3
    │   过滤：keep=yes 且 groundedness≥3 且 reasoning≥3 且 format≥4
    │
    └─ 推理链生成：prompt_d_long_reasoning(q, source_unit) ← LLM #4
        过滤：d_reasoning_flags()（长度、泄漏、稀薄度检查）
        组装：d_with_reasoning(think, answer)
```

**适用类型**：
- stage3_A：全量（46,675条），预期命中率 ~60%，产出 ~28,000 条
- stage3_B：全量（641条），预期命中率 ~70%，产出 ~450 条
- stage3_D：全量（151条），预期命中率 ~50%，产出 ~75 条

#### Route B：知识点扩展（适用 C 类）

适用文档：概念讲解文章，无明确题目结构。

```
原始文档 (text)
    │
    ├─ 概念类型判断：d_looks_like_concept_doc()
    │   非概念文档 → 转 Route A 处理
    │
    ├─ 知识点摘录：d_extract_concept_excerpt()
    │   提取 Objective / Definition / Example 节内容
    │
    ├─ 生成学生问题：prompt_d_self_instruct(knowledge_point, excerpt, count=2) ← LLM #1
    │   产出 2 个不同角度的问题（1 个概念理解，1 个简单应用）
    │
    ├─ 答案生成：prompt_d_answer_from_knowledge(q, knowledge_point, excerpt)   ← LLM #2
    │   每个问题单独生成答案，天然含 \boxed{}
    │
    ├─ 规则质量门控：d_rule_flags()
    │
    ├─ LLM 质量评分：prompt_d_quality_judge(q, a, source_unit=excerpt)         ← LLM #3
    │
    └─ 推理链生成：prompt_d_long_reasoning(q, source_unit=excerpt)              ← LLM #4
        组装：d_with_reasoning(think, answer)
```

**适用类型**：
- stage3_C：先用 d_looks_like_concept_doc() 过滤出真正的概念文章
  - 概念文章 → Route B，预期 ~18,000 条命中，每条产出 1-2 个 QA，约 ~22,000 条
  - 含 Example 节的文章 → Route A，产出 ~3,000 条

#### Route C：去重 + 合并

```
Route A 产出 + Route B 产出
    │
    ├─ d_dedup_rows(threshold=0.82)  Jaccard 去重
    │
    └─ 最终合并输出
```

---

## 四、完整 Pipeline 流程图

```
stage3_A.jsonl (46,675)  ─────────────────────────────────────────────┐
stage3_B.jsonl (641)     ─────────────────────────────────────────────┤
stage3_D.jsonl (151)     ─────────────────────────────────────────────┤
                                                                       ▼
                                                              Route A 处理
                                                         ┌─────────────────────┐
                                                         │ 1. 规则提取 source_unit│
                                                         │ 2. 显式/反向重构问题   │
                                                         │ 3. LLM 生成答案       │
                                                         │ 4. 规则过滤           │
                                                         │ 5. LLM 质量评分       │
                                                         │ 6. LLM 生成 <think>   │
                                                         │ 7. reasoning 过滤     │
                                                         └─────────┬───────────┘
                                                                   │
stage3_C.jsonl (27,682)  ──→ d_looks_like_concept_doc ──→ 概念文章  │
                                          │                         │
                                          │ 含Example节 ──────────→ Route A
                                          ▼
                                     Route B 处理
                              ┌─────────────────────┐
                              │ 1. 提取知识点摘录     │
                              │ 2. LLM 生成2个问题   │
                              │ 3. LLM 生成答案      │
                              │ 4. 规则过滤          │
                              │ 5. LLM 质量评分      │
                              │ 6. LLM 生成 <think>  │
                              │ 7. reasoning 过滤    │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   Route C 合并去重   │
                              │ d_dedup_rows(0.82)  │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────────────────────┐
                              │  最终输出                            │
                              │  final_reasoning_sft_train.jsonl    │
                              │  预期：40,000~55,000 条              │
                              └─────────────────────────────────────┘
```

---

## 五、质量控制体系

### 5.1 规则层（零 GPU 消耗）

```python
d_rule_flags(question, answer) 返回以下标志：
  - "missing_boxed"    : 答案中无 \boxed{}
  - "prompt_leak"      : 答案含 prompt 泄漏词（instruction/source unit/request 等）
  - "too_short"        : 答案长度 < 80 chars
  - "thin_reasoning"   : 答案换行数 < 1 且句号数 < 2（推理过于稀薄）
  - "question_copied"  : 答案几乎原样复制了问题文本

有任意 flag → 跳过，不进入 LLM 评分环节
```

### 5.2 LLM 评分层

```
prompt_d_quality_judge(question, answer, source_unit)
输出格式：
  groundedness=<1-5>   答案与原文的一致性
  reasoning=<1-5>      推理步骤质量
  pedagogy=<1-5>       教学清晰度
  format=<1-5>         格式规范性
  keep=<yes/no>
  reason=<一句话>

通过条件：keep=yes AND groundedness≥3 AND reasoning≥3 AND format≥4
```

### 5.3 推理链验证层

```
d_reasoning_flags(think_text) 返回：
  - "reasoning_too_short"     : < 220 chars
  - "reasoning_prompt_leak"   : 含 prompt 泄漏词
  - "reasoning_too_thin"      : 换行数 < 2 且句号数 < 4
  - "reasoning_too_long"      : > 7000 chars（可能失控）

有任意 flag → 丢弃该条，不进入最终输出
```

### 5.4 Token 长度过滤

```
训练前最终过滤：
  - 用 tokenizer 计算每条 messages 的总 token 数
  - 超过 9,012 tokens → 丢弃（约估计 1-3% 的数据）
  - 过短（< 200 tokens）→ 丢弃
```

---

## 六、输出文件结构

```
outputs/full_pipeline/
├── route_a.raw.jsonl              # Route A 全量（含质量评分、推理标志）
├── route_a.sft_train.jsonl        # Route A 通过筛选的训练样本
├── route_b.raw.jsonl              # Route B 全量
├── route_b.sft_train.jsonl        # Route B 通过筛选的训练样本
├── route_c.dedup.jsonl            # 去重后合并
└── final_reasoning_sft_train.jsonl  ★ 最终训练集
```

每条 audit 记录（raw.jsonl）额外字段：
```json
{
  "messages": [...],
  "source_id": "原始文档id",
  "route": "A/B",
  "source_unit": "用于生成的原文片段",
  "quality": {
    "score": 18,
    "keep": true,
    "rule_flags": [],
    "judge": {"groundedness": 4, "reasoning": 5, "pedagogy": 4, "format": 5, "keep": true}
  },
  "reasoning_quality": {
    "keep": true,
    "flags": [],
    "think_length": 420
  }
}
```

最终训练集（final_reasoning_sft_train.jsonl）只保留：
```json
{
  "messages": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "<think>推理...</think>\n答案\n\n\\boxed{X}"}
  ]
}
```

---

## 七、自我蒸馏衔接方案

### 7.1 第一轮 SFT 完成后的自我蒸馏流程

```
第一轮 SFT 模型 (checkpoint_v1)
    │
    ▼
对 final_reasoning_sft_train.jsonl 中的每个 question 重新采样
（温度 0.7，采样 3 条，保留 <think> 模式）
    │
    ▼
筛选：与原 gold answer 的 \boxed{} 答案一致 → 认为模型正确推理
（一致性检查：extract_boxed(generated) == extract_boxed(original)）
    │
    ▼
选出推理链更长、结构更完整的生成结果
替换原 gold answer 中质量较低的条目
    │
    ▼
第二轮 SFT 训练集 = 自我蒸馏选出的高质量集合
    │
    ▼
第二轮 SFT 模型 (checkpoint_v2)
```

### 7.2 自我蒸馏的数据格式要求

自我蒸馏不改变训练格式，仍然是：
```json
{"messages": [system, user, assistant(with <think>)]}
```

关键：推理时必须用 `enable_thinking=True`（Qwen3 特有），
确保模型在推理时真正进入 thinking 模式输出 `<think>` 块。

### 7.3 自我蒸馏对数据的要求

为支持自我蒸馏，route_a/b.raw.jsonl 中保留 `source_unit` 字段，
这样蒸馏时可以用原文做 groundedness 验证，确保蒸馏后的答案没有幻觉。

---

## 八、GPU 预算规划

### 8.1 LLM 调用量估算

| 阶段 | 处理条数 | 调用次数/条 | 预估总 tokens | 预估卡时 |
|------|---------|-----------|-------------|---------|
| Route A（A+B+D 类） | 47,467 | 最多 4 次 | ~160M | ~16 |
| Route B（C 类概念） | ~18,000 | 4 次 | ~80M | ~8 |
| Route A（C 类含Example） | ~9,682 | 4 次 | ~40M | ~4 |
| **Stage 5 合计** | | | **~280M** | **~28** |
| 第一轮 SFT 训练 | 40,000~55,000 条 | — | — | ~50 |
| 自我蒸馏采样 | 40,000~55,000 条 | 3 条/题 | — | ~20 |
| 第二轮 SFT 训练 | ~40,000 条 | — | — | ~50 |
| 评测提交 | — | — | — | ~5 |
| **总计** | | | | **~153** |
| 剩余预算（当前已用 ~0.5） | | | | **499.5 卡时** |

> 消耗约 153 卡时，剩余 346 卡时作为超参搜索和应急储备。

### 8.2 分阶段执行建议

```
阶段一（Stage 5 数据生成）：~28 卡时
  先跑 100 条小样本，人工检查 <think> 和 \boxed{} 质量
  确认后全量运行（建议分批：每批 5,000 条，方便断点续传）

阶段二（第一轮 SFT）：~50 卡时
  参数：lr=2e-5, epochs=2-3, batch_size=8
  用 lora 或全量微调（0.6B 模型全量微调可接受）

阶段三（自我蒸馏）：~20 卡时
  对训练集重采样，筛选高质量生成

阶段四（第二轮 SFT）：~50 卡时
  在蒸馏数据上继续训练

阶段五（最终评测）：~5 卡时
```

---

## 九、脚本设计

### 9.1 新建脚本

```
src/data/stage5_pipeline.py
├── ── vLLM 配置 ────────────────────────────────────
│   VLLM_BASE_URL = "http://localhost:8888/v1"  # 本地启动
│   VLLM_MODEL    = "Qwen3-0.6B"               # 本地模型名
│
├── ── 工具函数（直接从 Chapter7 复用）──────────────
│   chat(), save_jsonl(), d_clean_text(), d_title()
│   d_normalize(), d_jaccard(), d_tokens()
│   d_extract_problem_solution_unit()
│   d_extract_explicit_question(), d_clean_question()
│   d_looks_like_concept_doc(), d_extract_concept_excerpt()
│   d_clean_answer(), d_clean_reasoning()
│   d_rule_flags(), d_reasoning_flags(), d_parse_quality()
│   d_dedup_rows(), d_with_reasoning(), d_messages_only_rows()
│   ensure_boxed()
│
├── ── Prompt 函数（直接从 Chapter7 复用）──────────
│   prompt_d_backtranslate_question()
│   prompt_d_answer_from_doc()
│   prompt_d_self_instruct()
│   prompt_d_answer_from_knowledge()
│   prompt_d_long_reasoning()
│   prompt_d_quality_judge()
│
├── ── Route A 处理函数 ──────────────────────────────
│   process_route_a(doc) → Optional[dict]
│     1. d_extract_problem_solution_unit
│     2. d_extract_explicit_question / backtranslate
│     3. prompt_d_answer_from_doc
│     4. d_rule_flags
│     5. prompt_d_quality_judge
│     6. prompt_d_long_reasoning
│     7. d_reasoning_flags
│     8. 组装 + 返回 audit dict
│
├── ── Route B 处理函数 ──────────────────────────────
│   process_route_b(doc) → List[dict]
│     1. d_looks_like_concept_doc
│     2. d_extract_concept_excerpt
│     3. prompt_d_self_instruct（产出 2 个问题）
│     4. 每个问题：prompt_d_answer_from_knowledge
│     5. d_rule_flags → prompt_d_quality_judge
│     6. prompt_d_long_reasoning → d_reasoning_flags
│     7. 返回通过的条目列表
│
├── ── 主流程 ────────────────────────────────────────
│   main()
│   argparse：
│     --input {A,B,C,D,all}  指定处理哪类 stage3 数据
│     --limit N               小样本测试（默认 None=全量）
│     --workers N             并发线程数（默认 4）
│     --dry-run               只检查 vLLM 连通性
│     --resume                从上次断点续传
│
└── ── Token 长度过滤 + 最终合并 ────────────────────
    filter_by_token_length(rows, max_tokens=9012)
    merge_and_dedup(route_a_rows, route_b_rows)
    save 最终文件
```

### 9.2 本地 vLLM 启动命令

```bash
# 在 tmux/screen 中启动，不占满 GPU（留余量给数据生成脚本本身不需要 GPU）
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Qwen3-0.6B \
    --served-model-name Qwen3-0.6B \
    --port 8888 \
    --max-model-len 9012 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 \
    --enable-chunked-prefill

# 验证服务就绪
curl http://localhost:8888/v1/models
```

### 9.3 执行命令

```bash
# 小样本测试（先跑 100 条 A 类）
python src/data/stage5_pipeline.py --input A --limit 100

# 人工检查后全量
python src/data/stage5_pipeline.py --input A --workers 8
python src/data/stage5_pipeline.py --input B --workers 8
python src/data/stage5_pipeline.py --input C --workers 8
python src/data/stage5_pipeline.py --input D --workers 8

# 查看当前进度（断点续传时）
python src/data/stage5_pipeline.py --input A --resume
```

---

## 十、验证清单

### 10.1 小样本验证（正式运行前必做）

- [ ] 每条 assistant content 均包含 `<think>` 和 `</think>`
- [ ] 每条 assistant content 均包含 `\boxed{`
- [ ] system.content 为空字符串
- [ ] 无 prompt 泄漏词（instruction、source unit、request、assistant: 等）
- [ ] 推理链长度 ≥ 220 chars
- [ ] 问题文本有意义（不以小写字母开头、不是 "Example" 等词）
- [ ] token 总长度 ≤ 9,012

### 10.2 全量运行后统计

- [ ] 各路线输出条数
- [ ] 各质量门控的通过率
- [ ] 推理链平均长度
- [ ] Token 长度分布
- [ ] 更新 pipeline_progress.md

---

## 十一、关键设计决策备注

### 为什么 source_unit 要截断到 1,400 chars？

LLM 的 prompt 总长度有限制（vLLM max_model_len=9012），
source_unit + question + prompt 模板已经占用约 1,800 tokens，
再加上答案生成的 max_tokens=240，需要给推理链 max_tokens=1200，
因此 source_unit 建议控制在 1,400 chars（约 500 tokens）以内。

### 为什么 Route B 每文档只生成 2 个问题？

避免同一文档过度采样导致训练集的主题分布偏斜。
去重环节（Jaccard ≥ 0.82）会进一步过滤同文档内相似度过高的问题对。

### 关于 C 类文档的路由判断

`d_looks_like_concept_doc()` 判断为"概念文章"的走 Route B；
判断为"含结构"的（含 ## Problem、## Solution、## Example 节）走 Route A。
实际上 C 类约 60% 会走 Route B，40% 走 Route A。

### 自我蒸馏中的答案一致性检查

提取 `\boxed{}` 内容后做字符串规范化比较：
- 数字：去除空格、统一小数点格式
- 分数：统一到最简形式（可选）
- 文字答案：小写 + 去标点后比较

一致 = 模型推理正确，其推理链可作为高质量 demonstration。
