# Qwen3-0.6B-Base 数学推理 SFT 项目

## 项目约束（违反会影响成绩）

### 模型约束
- **唯一允许的基础模型**：`Qwen/Qwen3-0.6B-Base`
- 允许用 Base 模型生成合成数据
- 禁止使用任何其他预训练模型

### 数据约束
- **唯一允许的数据源**：Nemotron-CC-Math-v1 的 10 万条指定子集 (`data/MidTerm_Project_TrainingData_100000.zip`)
- **允许的操作**：过滤、清洗、去重、切分、格式整理
- **允许派生的数据**：基于子集构造 QA pair、CoT、thinking 格式、偏好数据
- **禁止**：使用子集以外的任何外部数据

### 算力约束
- **GPU 预算**：总计 ≤500 卡时（含数据合成 + SFT + 重复实验）
- 预算耗尽前必须完成所有实验

### 推理约束
- **最大生成长度**：9012 tokens
- **推理模式**：单模型直接生成
- **禁止**：搜索算法、计算器、外部工具

### 训练约束
- **允许**：SFT 及修改 SFT 的 Loss Function
- **必须**：使用官方提供的 Chat 模版

### 评分基准
- **验证集基线**：Base 模型准确率约 38.2%
- **有效成绩**：必须超过基线
- **最终提交**：上传 `.safetensors` 格式模型权重

---

## 项目目录结构

**期中项目代码位于 `midterm/` 目录**

```
D:/college/llm/
├── midterm/              # 期中项目目录
│   ├── project_plan.md        # 项目总计划
│   ├── task_member_A.md       # 成员 A 任务说明
│   ├── task_member_B.md       # 成员 B 任务说明
│   ├── task_member_C.md       # 成员 C 任务说明
│   └── task_member_D.md       # 成员 D 任务说明
├── src/
│   ├── data/               # 数据处理模块
│   │   ├── exploratory_analysis.py   # 数据探索
│   │   ├── cleaning.py               # 去重清洗
│   │   ├── difficulty_stratification.py  # 难度分层
│   │   ├── format_conversion.py      # 格式转换
│   │   └── synthetic_data.py         # 合成数据生成
│   ├── training/             # 训练模块
│   │   ├── train_sft.py      # SFT 训练脚本
│   │   ├── config.yaml       # 训练配置
│   │   └── utils.py          # 工具函数
│   ├── evaluation/           # 评测模块
│   │   └── evaluate.py       # 本地评测脚本
│   ├── experiments/          # 实验模块
│   │   ├── hyperparam_search.py  # 超参搜索
│   │   ├── data_ratio.py     # 数据配比实验
│   │   └── configs/          # 实验配置
│   └── submission/           # 提交模块
│       └── submit.py         # 提交脚本
├── data/
│   ├── raw_nemotron.json     # 原始数据
│   ├── cleaned_nemotron.json # 清洗后数据
│   ├── sampled_nemotron.json # 采样后数据
│   ├── sft_train.jsonl       # SFT 训练集
│   ├── synthetic_cot.jsonl   # CoT 合成数据
│   ├── synthetic_variants.jsonl  # 答案变体
│   ├── final_sft_dataset.jsonl   # 最终训练集
│   └── valid_data_1000.jsonl # 验证集
├── outputs/
│   ├── checkpoints/          # 训练检查点
│   ├── final_model/          # 最终模型
│   └── experiments/          # 实验输出
├── reports/
│   ├── data_exploration_stats.json
│   ├── difficulty_analysis.log
│   ├── baseline_training.log
│   ├── baseline_accuracy.log
│   ├── hyperparam_search_results.json
│   ├── data_ratio_results.json
│   └── final_experiment_summary.md
├── submission_package/       # 提交包
├── docs/
│   ├── superpowers/
│   │   ├── plans/            # 实施计划
│   │   └── specs/            # 设计文档
│   └── team_assignments.md   # 团队分工
├── scripts/
│   ├── prepare_submission.sh
│   └── final_submission.sh
├── tests/
│   ├── data/
│   └── training/
└── CLAUDE.md                 # 本文件
```

---

## 团队分工（4 人）

| 成员 | 模块 | 职责 |
|------|------|------|
| A | `src/data/` | 数据清洗、格式化、合成数据生成 |
| B | `src/training/` | SFT 训练脚本、Loss 设计、配置管理 |
| C | `src/evaluation/` | 本地评测、榜单提交、实验记录 |
| D | `src/experiments/` | 超参扫描、结果汇总、GPU 调度 |

---

## 开发规范

### 代码风格
- 使用类型注解
- 函数添加 docstring
- 遵循 PEP 8

### Git 提交
- 小步提交，每完成一个功能点提交一次
- 提交信息格式：`type: description`
- 模型权重不提交到 git（加入 `.gitignore`）

### 实验管理
- 每次实验记录配置和结果到 `reports/`
- 使用唯一名称标识实验（如 `lr_5e-5_epoch3`）

---

## 关键命令速查

```bash
# 数据探索
python src/data/exploratory_analysis.py

# 数据清洗
python src/data/cleaning.py

# 难度分层（GPU 密集）
python src/data/difficulty_stratification.py

# 格式转换
python src/data/format_conversion.py

# 合成数据生成（GPU 密集）
python src/data/synthetic_data.py

# SFT 训练（GPU 密集）
python src/training/train_sft.py

# 本地评测（GPU 密集）
python src/evaluation/evaluate.py --model outputs/final_model

# 超参搜索（GPU 密集）
python src/experiments/hyperparam_search.py

# 准备提交
bash scripts/prepare_submission.sh outputs/final_model
```

---

## GPU 预算追踪

当前预算：500 卡时

| 阶段 | 预算 | 实际 | 状态 |
|------|------|------|------|
| Phase 1（数据） | 170 | - | ⏳ |
| Phase 2（Baseline） | 115 | - | ⏳ |
| Phase 3（迭代） | 150 | - | ⏳ |
| Phase 4（收尾） | 50 | - | ⏳ |
| 预留 | 15 | - | ⏳ |

---

## 重要提醒

1. **不要使用外部数据**：所有训练数据必须来源于 Nemotron-CC-Math-v1 10 万条子集
2. **不要超过 500 卡时**：在运行任何 GPU 任务前确认预算
3. **必须使用 Chat 模版**：`tokenizer.apply_chat_template()` 是官方推荐方式
4. **验证准确率必须 >38.2%**：否则成绩无效
5. **提交格式必须是 .safetensors**：pytorch 模型需要转换
