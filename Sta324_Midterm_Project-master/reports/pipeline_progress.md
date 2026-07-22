# 数据管道进度总览

**项目**：Qwen3-0.6B-Base 数学推理 SFT — 期中项目
**课程**：2025-2026 Spring Semester（讲师：Linyi Yang）
**更新日期**：2026-04-10

---

## 整体进度

| 阶段 | 脚本 | 状态 | 输入 | 输出（保留） | 丢弃 |
|------|------|------|------|-------------|------|
| Stage 1A 过滤 | `src/data/stage1_filter.py` | ✅ 完成 | 100,000 | 86,011 normal + 253 long | 13,736 |
| Stage 1B 切分 | `src/data/stage1_split_long.py` | ✅ 完成 | 253 长文档 | 5,320 语义块 | 若干非数学块 |
| Stage 2 清洗 | `src/data/stage2_clean.py` | ✅ 完成 | 91,331 | 91,331 | 0 |
| Stage 3 分类 | `src/data/stage3_classify.py` | ✅ 完成 | 91,331 | 75,149 | 16,182 |
| Stage 4A 格式化 | `src/data/stage4A_format_conversion.py` | ✅ 完成 | 46,675（A类） | 23,102（49.5%） | 23,573 |
| Stage 4B 格式化 | `src/data/stage4B_conversion.py` | ✅ 完成 | 641（B类） | 422（65.8%） | 219 |
| Stage 4C 格式化 | `src/data/stage4C_conversion.py` | ✅ 完成 | 27,682（C类） | 14,064（50.8%） | 13,618 |
| Stage 4D 格式化 | `src/data/stage4D_conversion.py` | ✅ 完成 | 151（D类） | 55（36.4%） | 96 |
| SFT 训练 | `src/training/train_sft.py` | ⏳ 待开发 | - | - | - |
| 评测提交 | `src/evaluation/evaluate.py` | ⏳ 待开发 | - | - | - |

---

## 数据流汇总

```
原始数据 100,000 条
    │
    ▼ Stage 1A：规则过滤（CPU）
    ├─ 保留 normal：86,011 条
    ├─ 进入 long 队列：253 条
    └─ 丢弃：13,736 条
         │
         ▼ Stage 1B：长文档语义切分（CPU）
         └─ 253 条 → 5,320 语义块
                │
                ▼ 合并（normal + long_chunks）
                91,331 条
                    │
                    ▼ Stage 2：文本清洗（CPU）
                    91,331 条（无丢弃）
                        │
                        ▼ Stage 3：内容分类（GPU，Qwen3-0.6B，~20min）
                        ├─ A 类 PROBLEM_SOLUTION：46,675 条（51.1%）
                        ├─ B 类 FORUM_QA：641 条（0.7%）
                        ├─ C 类 ARTICLE_TUTORIAL：27,682 条（30.3%）
                        ├─ D 类 TEXTBOOK：151 条（0.2%）
                        └─ 丢弃 (E+F)：16,182 条（17.7%）

最终可用于 Stage 4：75,149 条（82.3%）

                        ▼ Stage 4（格式转换，CPU）
                        ├─ A 类 → 23,102 条 SFT（49.5%）
                        ├─ B 类 →    422 条 SFT（65.8%）
                        ├─ C 类 → 14,064 条 SFT（50.8%）
                        └─ D 类 →     55 条 SFT（36.4%）

当前 SFT 数据集（A+B+C+D）：37,643 条
```

---

## GPU 预算使用

| 阶段 | 预算 | 实际消耗 | 备注 |
|------|------|---------|------|
| Stage 3 分类 | Phase 1 内 | 约 0.5 卡时 | Qwen3-0.6B，batch_size=32，77.5 rec/s |
| Phase 1 数据合成 | 170 卡时 | 0 | 待开发 |
| Phase 2 Baseline SFT | 115 卡时 | 0 | 待开发 |
| Phase 3 迭代优化 | 150 卡时 | 0 | 待开发 |
| Phase 4 收尾提交 | 50 卡时 | 0 | 待开发 |
| 预留 | 15 卡时 | 0 | - |
| **合计** | **500 卡时** | **~0.5 卡时** | 剩余 499.5 卡时 |

---

## 各阶段详细日志

- [stage1_filter_log.md](stage1_filter_log.md) — Stage 1 过滤详情
- [stage2_clean_log.md](stage2_clean_log.md) — Stage 2 清洗详情
- [stage3_classify_log.md](stage3_classify_log.md) — Stage 3 分类详情
- [stage4A_method_report.md](stage4A_method_report.md) — Stage 4A A类处理方法报告
- [stage4B_method_report.md](stage4B_method_report.md) — Stage 4B B类处理方法报告
- [stage4D_method_report.md](stage4D_method_report.md) — Stage 4D D类处理方法报告
