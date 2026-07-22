# Stage 3 分类日志

**脚本**：`src/data/stage3_classify.py`
**执行日期**：2026-04-08
**运行环境**：GPU（Qwen3-0.6B，float16，device_map="auto"）
**模型路径**：`/home/ubuntu/models/Qwen3-0.6B`

---

## 输入/输出统计

| 类别 | 名称 | 记录数 | 占比 | 去向 |
|------|------|--------|------|------|
| A | PROBLEM_SOLUTION | 46,675 | 51.1% | ✅ 保留 → Stage 4A |
| B | FORUM_QA | 641 | 0.7% | ✅ 保留 → Stage 4B |
| C | ARTICLE_TUTORIAL | 27,682 | 30.3% | ✅ 保留 → Stage 4C |
| D | TEXTBOOK | 151 | 0.2% | ✅ 保留 → Stage 4D |
| E | CALCULATOR_TOOL | — | — | ❌ 丢弃 |
| F | OTHER | — | — | ❌ 丢弃 |
| E+F 合计 | 丢弃 | 16,182 | 17.7% | ❌ stage3_discard.jsonl |
| **合计保留** | | **75,149** | **82.3%** | |
| **总输入** | | **91,331** | 100% | |

---

## 分类策略：三阶段混合

### Pass 1 — 规则预分类（CPU，即时）

高置信度直接分类，跳过 LLM：

| 规则函数 | 判断逻辑 | 分类结果 |
|---------|---------|---------|
| `is_calculator_page()` | ≥8 条纯算术行，或单位换算表，或 >50% 纯数字行 | → E（丢弃）|
| `is_forum()` | ≥2 处命中 `Posted by / Reply #N / Thread starter / Member Posts` 等 | → B |
| `is_textbook()` | ≥2 处编号章节（`1.1 Introduction`）**且** ≥2 处正式块（`Theorem 2.1`）| → D |
| `is_problem_solution()` | 同时有 Problem/Question/Exercise 标题 **且** Solution/Answer 标题 | → A |

### Pass 2 — LLM 批量推理（GPU）

对 Pass 1 未能确定的记录，使用 Qwen3-0.6B 批量分类。

**配置参数**：

| 参数 | 值 | 说明 |
|------|----|------|
| BATCH_SIZE | 32 | 批推理大小 |
| SNIPPET_LEN | 600 chars | 送入模型的文本截断长度 |
| MAX_IN_TOKS | 900 tokens | 含 prompt 的最大输入长度 |
| MAX_NEW_TOKS | 4 tokens | 只需生成1个字母 |
| do_sample | False | 贪心解码 |
| padding_side | left | 批推理必须左填充 |

**Prompt 设计**（Few-shot Completion 风格）：
- 不使用 chat template / system prompt（Qwen3-0.6B 是 base 模型，指令跟随能力弱）
- 5 个 few-shot 示例，覆盖 A/B/C/D/E 全类别
- 格式：`Text: "..." \nCategory: X`（让模型补全最后一个字母）

**推理速度**：77.5 rec/s → 91,331 条约 20 分钟

> **踩坑记录**：
> - 初版使用 `apply_chat_template` + 长 system prompt → 全部输出 "F"
>   原因：base 模型不具备指令跟随能力，长提示词无效
>   解决：改为 few-shot completion 直接续写模式
> - 初版单条推理速度 0.78 s/条 → 估计 19.8 小时
>   解决：batch_size=32 + padding_side="left" → 77.5 rec/s，约 20 分钟

### Pass 3 — F 标签回退处理

LLM 输出 "F"（OTHER）时，按以下顺序重新判断：

```
F 标签
  ├─ 1. 重跑规则分类器 → 如有命中则采用规则标签
  ├─ 2. 词问题启发式：文本 <5000 chars 且包含
  │       find / calculate / solve / prove that / given that 等
  │      → 归为 A
  ├─ 3. 定义/定理启发式：包含
  │       definition / theorem / lemma / corollary / proof 等
  │      → 归为 C
  └─ 4. 以上均不符合 → 保留 F（最终丢弃）
```

---

## 输出文件

| 文件 | 记录数 | 说明 |
|------|--------|------|
| `stage3_output/stage3_A.jsonl` | 46,675 | PROBLEM_SOLUTION — 题目+解答型 |
| `stage3_output/stage3_B.jsonl` | 641 | FORUM_QA — 论坛问答型 |
| `stage3_output/stage3_C.jsonl` | 27,682 | ARTICLE_TUTORIAL — 文章教程型 |
| `stage3_output/stage3_D.jsonl` | 151 | TEXTBOOK — 教科书型 |
| `stage3_output/stage3_discard.jsonl` | 16,182 | E（计算工具页）+ F（其他）丢弃 |

每条记录新增字段：`"stage3_label": "A"/"B"/"C"/"D"/"E"/"F"`
