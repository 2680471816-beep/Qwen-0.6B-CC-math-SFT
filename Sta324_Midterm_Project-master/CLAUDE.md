# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 项目背景

**课程**：2025-2026 Spring Semester（讲师：Linyi Yang）
**目标**：通过后训练方法提升 `Qwen3-0.6B-Base` 在数学推理任务上的表现。

**重要规则**：每次回答时须说一句：**seeker喵喵喵我来啦**，以作为参考过本文件的证据。

---

## 核心约束（必须严格遵守）

### 模型
- 唯一允许使用的基础模型：**`Qwen/Qwen3-0.6B-Base`**（本地路径：`/home/ubuntu/models/Qwen3-0.6B-Base`）
- 合成数据只能用该模型生成，**禁止使用任何其他预训练模型**

### 数据
- 训练数据**只能**来源于 **Nemotron-CC-Math-v1 的 10 万条子集**
  - 原始文件：`nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl`
  - 备用压缩包：`data/MidTerm_Project_TrainingData_100000.zip`
- **允许的操作**：过滤、清洗、去重、切分、格式整理、构建 QA pair / CoT / thinking 格式、偏好数据
- **禁止**：使用子集以外的任何外部数据

### 推理（评测时严格执行）
- 最大生成长度：**9,012 tokens**
- 仅允许单模型直接生成，**禁止** beam search、MCTS 等搜索算法
- **禁止**调用外部模型、检索、计算器、代码执行器等任何外部工具
- **必须**使用官方提供的 Chat 模板（`tokenizer.apply_chat_template()`）

### 训练
- **允许**：SFT 类算法（Supervised Fine-Tuning），允许修改 Loss Function
- 谨慎使用强化学习等非 SFT 类算法

---

## 项目目录结构

```
/home/ubuntu/Midterm_Project/
├── CLAUDE.md                          # 本文件（项目总指导）
├── New_demand.md                      # 原始需求文档
├── data_pipeline.md                   # 数据管道说明文档
├── nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl  # 原始100K数据
│
├── src/                               # ★ 所有代码统一放这里
│   ├── data/                          # 数据处理模块
│   │   ├── __init__.py
│   │   ├── stage1_filter.py           # Stage1: 规则过滤（CPU）
│   │   ├── stage1_split_long.py       # Stage1: 长文档切分（CPU）
│   │   ├── stage2_clean.py            # Stage2: 文本清洗（CPU）
│   │   ├── stage3_classify.py         # Stage3: 内容分类（GPU，Qwen3-0.6B）
│   │   ├── stage4A_format_conversion.py   # Stage4A: A类格式转换（已完成但废弃）
│   │   ├── stage4B_conversion.py          # Stage4B: B类格式转换（已完成但废弃）
│   │   ├── stage4C_conversion.py          # Stage4C: C类格式转换（已完成但废弃）
│   │   ├── stage4D_conversion.py          # Stage4D: D类格式转换（已完成但废弃）
│   │   └── stage5_pipeline.py            # Stage5: 完整重洗 + reasoning-SFT（✅ 脚本已完成）
│   ├── training/                      # 训练模块
│   │   ├── __init__.py
│   │   ├── train_sft.py               # SFT训练脚本（待开发）
│   │   ├── config.yaml                # 训练配置（待开发）
│   │   └── utils.py                   # 工具函数（待开发）
│   ├── evaluation/                    # 评测模块
│   │   ├── __init__.py
│   │   └── evaluate.py                # 本地评测脚本（待开发）
│   └── experiments/                   # 实验模块
│       ├── __init__.py
│       ├── hyperparam_search.py       # 超参搜索（待开发）
│       ├── data_ratio.py              # 数据配比实验（待开发）
│       └── configs/                   # 实验配置目录
│
├── stage1_output/                     # Stage1 输出（Git LFS）
│   ├── stage1_normal.jsonl            # 86,011 条正常长度记录
│   ├── stage1_long.jsonl              # 253 条超长文档（原始）
│   ├── stage1_long_chunks.jsonl       # 5,320 条切分后语义块
│   └── stage1_discarded.jsonl        # 13,736 条丢弃记录
├── stage2_output/                     # Stage2 输出（Git LFS）
│   ├── stage2_clean.jsonl             # 91,331 条清洗后记录
│   └── stage2_discarded.jsonl        # 0 条（post-clean过短，保留用于审计）
├── stage3_output/                     # Stage3 输出（Git LFS）
│   ├── stage3_A.jsonl                 # 46,675 条 PROBLEM_SOLUTION
│   ├── stage3_B.jsonl                 # 641 条 FORUM_QA
│   ├── stage3_C.jsonl                 # 27,682 条 ARTICLE_TUTORIAL
│   ├── stage3_D.jsonl                 # 151 条 TEXTBOOK
│   └── stage3_discard.jsonl          # 16,182 条（E+F 类，丢弃）
│
├── outputs/                           # 训练输出
│   ├── checkpoints/                   # 训练检查点
│   ├── final_model/                   # 最终模型权重
│   ├── experiments/                   # 各实验输出
│   └── full_pipeline/                 # Stage5 输出
│       ├── route_a.raw.jsonl          # Route A 全量审计（含质量评分）
│       ├── route_a_from_c.raw.jsonl   # C类中走Route A的审计记录
│       ├── route_a.sft_train.jsonl    # Route A 通过筛选的训练样本
│       ├── route_b.raw.jsonl          # Route B 全量审计
│       ├── route_b.sft_train.jsonl    # Route B 通过筛选的训练样本
│       └── final_reasoning_sft_train.jsonl  # ★ 最终训练集
└── reports/                           # 实验报告与日志
```

> **注意**：`stage*_output/` 数据目录保留在项目根部（已配置 Git LFS），避免大文件迁移风险。

---

## 团队分工（4 人）

| 成员 | 模块 | 职责 |
|------|------|------|
| A | `src/data/` | 数据清洗、格式化、合成数据生成 |
| B | `src/training/` | SFT 训练脚本、Loss 设计、配置管理 |
| C | `src/evaluation/` | 本地评测、榜单提交、实验记录 |
| D | `src/experiments/` | 超参扫描、结果汇总、GPU 调度 |

---

## 数据管道进度

| 阶段 | 脚本 | 输入 | 输出 | 状态 |
|------|------|------|------|------|
| Stage1 过滤 | `src/data/stage1_filter.py` | 100,000 条原始 | 86,011 normal + 253 long | ✅ 完成 |
| Stage1 切分 | `src/data/stage1_split_long.py` | 253 长文档 | 5,320 语义块 | ✅ 完成 |
| Stage2 清洗 | `src/data/stage2_clean.py` | 91,331 条 | 91,331 条（0 丢弃） | ✅ 完成 |
| Stage3 分类 | `src/data/stage3_classify.py` | 91,331 条 | 75,149 保留 / 16,182 丢弃 | ✅ 完成 |
| Stage4 格式化 | `src/data/stage4*_conversion.py` | 75,149 条（A/B/C/D） | 37,643 条 SFT（已废弃） | ✅ 完成（但质量不足） |
| **Stage5 重洗** | `src/data/stage5_pipeline.py` | 75,149 条（stage3） | **reasoning-SFT 含 \<think\>/\boxed{}** | ✅ **脚本完成，待运行** |
| SFT 训练 | `src/training/train_sft.py` | Stage5 输出 | checkpoint | ⏳ 待开发 |
| 评测提交 | `src/evaluation/evaluate.py` | checkpoint | 验证集得分 | ⏳ 待开发 |

### Stage4 vs Stage5 决策说明

**Stage4 问题**：
- 仅 3.4% 的答案包含 `\boxed{}`，导致评测无法提取答案
- 无 `</think>` 块，无法训练 thinking 模式
- 部分答案含噪声（Comments、References、截断句）

**Stage5 方案**（完整重洗）：
- 直接从 stage3_output/ 出发，使用本地 vLLM（Qwen3-0.6B）
- Route A（A/B/D 类）：规则切 unit → LLM 生成问题/答案/推理链
- Route B（C 类概念）：知识点扩展 → LLM 生成 2 个问题 → 答案/推理链
- 三层质量控制：规则过滤 → LLM 评分 → 推理链验证
- 预期产出：40,000~55,000 条高质量 reasoning-SFT

---

## 算力与费用预算

| 项目 | 限制 |
|------|------|
| 总算力预算 | 500 GPU 卡时 |
| 服务器费用上限 | 2,500 元（超出可能扣分） |

### GPU 预算阶段分配

| 阶段 | 内容 | 预算 | 实际 | 状态 |
|------|------|------|------|------|
| Phase 1（数据） | Stage5 重洗 + reasoning 合成 | 170 卡时 | ~30 | ✅ 脚本完成，待启动 GPU |
| Phase 2（Baseline） | 首轮 SFT + 评测 | 115 卡时 | - | ⏳ |
| Phase 3（蒸馏） | 自我蒸馏 + 第二轮 SFT | 100 卡时 | - | ⏳ |
| Phase 4（迭代） | 超参搜索、数据配比 | 100 卡时 | - | ⏳ |
| Phase 5（收尾） | 最终训练、提交 | 15 卡时 | - | ⏳ |
| **合计** | | **500 卡时** | | |

### Stage5 GPU 消耗细分

| 步骤 | 条数 | LLM 调用 | 预估卡时 |
|------|------|---------|---------|
| Route A（A+B+D 类） | 47,467 | 4 次/条 | ~16 |
| Route B（C 类概念） | ~18,000 | 4 次/条 | ~8 |
| Route A（C 类含 Example） | ~9,682 | 4 次/条 | ~4 |
| **Stage5 合计** | | | **~30** |
| 第一轮 SFT | 40K-55K 条 | — | ~50 |
| 自我蒸馏采样 | 40K-55K 条 | 3 条/题 | ~20 |
| 第二轮 SFT | ~40K 条 | — | ~50 |
| **总计（含蒸馏）** | | | **~148** |

---

## 评测与提交

- **Baseline（验证集准确率）**：38.2%（须超越此值方为有效成绩）
- **成绩依据**：最终测试集排名（不计报告分、过程分、验证集排名）
- 提交地址：`http://10.20.96.21:10808/`（仅校内网络）
- 提交格式：**`.safetensors`** 模型权重文件
- 默认以**最后一次提交**为正式版本，可手动指定某次 checkpoint
- 提交后约 **10 分钟**完成验证集评分

---

## 关键数字速查

| 项目 | 数值 |
|------|------|
| 基础模型参数量 | 0.6B |
| 训练数据量 | 100,000 条 |
| 验证集大小 | 1,000 条 |
| 测试集大小 | 100 条 |
| 最大生成长度 | 9,012 tokens |
| Baseline 准确率 | 38.2% |
| Stage5 预期产出 | 40,000~55,000 条 reasoning-SFT |

---

## Stage5 训练数据格式

### 最终 SFT 格式（严格遵守）

```json
{
  "messages": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "题目（自然语言 + LaTeX）"},
    {"role": "assistant", "content": "<think>\n详细推理过程...\n关键步骤...\n验证...\n</think>\n简洁正式解答。\n\n\\boxed{最终答案}"}
  ]
}
```

### 关键约束

- `system.content` **必须为空字符串** `""`（与 Qwen3 官方推荐对齐）
- `</think>...</think>` 块：至少 8 句有效推理，覆盖列式、化简、验证
- `\boxed{}` 必须在 assistant content 末尾，且只能出现一次
- 每条 token 总长度 ≤ 9,012（模型推理上限）

### 三层质量控制

1. **规则层**（零 GPU）：检查 `\boxed{}`、prompt 泄漏、长度、稀薄度
2. **LLM 评分层**：`prompt_d_quality_judge` 评分 groundedness/reasoning/pedagogy/format
3. **推理链验证层**：`d_reasoning_flags` 检查 reasoning 长度、泄漏、稀薄度

---

## 自我蒸馏（Self-Distillation）方案

### 目标

第一轮 SFT 完成后，对训练集问题重新采样，用模型自己的高质量生成替换原始数据中的低质量答案。

### 流程

```
第一轮 SFT 模型 (checkpoint_v1)
    ↓ 对训练集问题重新生成（temperature=0.7，采样 3 条）
    ↓ 保留 </think> 模式
    ↓ 筛选：\boxed{} 答案一致 → 推理正确
    ↓ 选出推理链更长、更完整的生成结果
    ↓ 替换原 gold answer 中质量较低的条目
第二轮 SFT 训练集 = 自我蒸馏选出的高质量集合
    ↓ 第二轮 SFT 模型 (checkpoint_v2)
```

### 一致性检查

提取 `\boxed{}` 内容后做字符串规范化比较：
- 数字：去除空格、统一小数点格式
- 分数：统一到最简形式
- 文字答案：小写 + 去标点后比较

### 自我蒸馏对数据的要求

- audit 文件保留 `source_unit` 和 `quality_score`
- 支持用原文做 groundedness 验证
- 避免蒸馏后的答案产生幻觉

---

## 数据集结构

`nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl` 每条记录格式：
```json
{
  "id": "<uuid>",
  "text": "<数学相关网页正文，Markdown格式>",
  "metadata": {
    "warc_filename": "<CC爬虫来源>",
    "warc_id": "<记录ID>",
    "finemath_int_scores": 4,
    "finemath_scores": 4.19,
    "nemocurator_int_scores": 2,
    "nemocurator_scores": 2.23,
    "category": "math",
    "models_used": "Phi-4"
  }
}
```

---

## 开发规范

### 代码风格
- 使用类型注解
- 函数添加 docstring
- 遵循 PEP 8

### Git 提交
- 小步提交，每完成一个功能点提交一次
- 提交信息格式：`type: description`（如 `feat: add SFT training script`）
- 模型权重**不提交**到 git（加入 `.gitignore`）
- 大数据文件通过 Git LFS 管理

### 实验管理
- 每次实验记录配置和结果到 `reports/`
- 使用唯一名称标识实验（如 `lr_5e-5_epoch3`）
- 留存所有训练日志和 checkpoint 路径

---

## 关键命令速查

```bash
# === 数据准备（已完成） ===
# Stage1 过滤 + 切分
python src/data/stage1_filter.py
python src/data/stage1_split_long.py

# Stage2 清洗
python src/data/stage2_clean.py

# Stage3 分类（GPU密集）
python src/data/stage3_classify.py

# === Stage5 数据重洗（脚本已完成，启动前先开 vLLM）===
# 1. 启动本地 vLLM 服务（在 tmux/screen 中）
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Qwen3-0.6B-Base \
    --served-model-name Qwen3-0.6B \
    --port 8888 \
    --max-model-len 9012 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16

# 2. 小样本测试（先跑 100 条验证质量）
python src/data/stage5_pipeline.py --input A --limit 100
python src/data/stage5_pipeline.py --input B --limit 100
python src/data/stage5_pipeline.py --input C --limit 100

# 3. 全量运行（各输入类型分开跑，或一次性全跑）
python src/data/stage5_pipeline.py --input A --workers 8
python src/data/stage5_pipeline.py --input B --workers 8
python src/data/stage5_pipeline.py --input C --workers 8
python src/data/stage5_pipeline.py --input D --workers 8
# 或一次性运行所有路线并自动合并：
# python src/data/stage5_pipeline.py --input all --workers 8

# 4. 仅合并/续传
python src/data/stage5_pipeline.py --merge
python src/data/stage5_pipeline.py --input A --resume

# === SFT 训练（待开发） ===
# 第一轮 SFT
python src/training/train_sft.py \
    --data outputs/full_pipeline/final_reasoning_sft_train.jsonl \
    --output outputs/checkpoints/round1 \
    --epochs 2 --lr 2e-5 --batch_size 8

# === 自我蒸馏（待开发） ===
# 用第一轮模型对训练集重采样
python src/training/self_distillation.py \
    --model outputs/checkpoints/round1 \
    --data outputs/full_pipeline/final_reasoning_sft_train.jsonl \
    --output outputs/full_pipeline/distilled_sft_train.jsonl \
    --samples_per_question 3

# 第二轮 SFT
python src/training/train_sft.py \
    --data outputs/full_pipeline/distilled_sft_train.jsonl \
    --output outputs/checkpoints/round2 \
    --epochs 2 --lr 2e-5 --batch_size 8

# === 评测提交（待开发） ===
python src/evaluation/evaluate.py --model outputs/checkpoints/round2
```

---

## 违规红线

以下行为导致**全组成绩记 0 分**：
- 探测测试集结果
- 提交与训练过程不一致的模型权重
- 破坏其他组进展或服务器设备
- 使用服务器挖矿、出租等
- 使用子集以外的任何外部数据或外部模型

---

## 开始前检查清单

- [ ] 修改服务器登录密码（默认为学号）
- [ ] 确认已获取 Chat 模板文件
- [ ] 留存所有实验记录（训练日志、checkpoint 路径等）
- [ ] 监控服务器费用，避免超过 2,500 元
- [ ] 合理分配 500 GPU 卡时，预留余量给最终调优
- [ ] 确认提交格式为 `.safetensors`
- [ ] 启动本地 vLLM 服务（`http://localhost:8888/v1`）后再运行 Stage5
- [ ] Stage5 小样本测试后人工检查 </think> 质量

---

## Stage5 参考文档

- `reports/stage5_pipeline_design.md` — 完整设计文档（路线、质量控制、GPU 预算）
- `reports/stage5_pipeline_report.md` — 技术说明报告（输入数据、分类原则、Route A/B 详细处理方法、去重合并、模型调用、多线程机制）
- `Chapter7_Data.ipynb` — vLLM 调用、prompt 函数、质量门控函数来源（参考用，不提交）

---

## Stage5 输出文件说明

```
outputs/full_pipeline/
├── route_a.raw.jsonl              # Route A 全量审计（含质量评分、推理标志）
├── route_a_from_c.raw.jsonl       # C类中走Route A的审计记录
├── route_a.sft_train.jsonl        # Route A 通过筛选的训练样本
├── route_b.raw.jsonl              # Route B 全量审计
├── route_b.sft_train.jsonl        # Route B 通过筛选的训练样本
├── final_reasoning_sft_train.jsonl  # ★ 最终训练集
└── （stage5_traces.jsonl 暂未实现，审计信息已并入 raw.jsonl）
```

### audit 记录（raw.jsonl）额外字段

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
    "judge": {"groundedness": 4, "reasoning": 5, "pedagogy": 4, "format": 5}
  },
  "reasoning_quality": {
    "keep": true,
    "flags": [],
    "think_length": 420
  }
}
```

### 最终训练集格式

只保留 `messages` 字段：
```json
{
  "messages": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "<think>\n推理...\n\n答案\n\n\\boxed{X}"}
  ]
}
```
