# Stage 2 清洗日志

**脚本**：`src/data/stage2_clean.py`
**执行日期**：2026-04-08
**运行环境**：CPU only

---

## 输入/输出统计

| 项目 | 数值 |
|------|------|
| 输入总量 | 91,331 条（stage1_normal + stage1_long_chunks）|
| 保留 | **91,331 条**（100%）|
| 丢弃（post-clean < 600 chars）| **0 条** |

---

## 清洗规则详情

### Rule 1 — 导航 / UI 无用行删除

**策略**：仅删除整行为纯导航噪声的行（保守策略，不影响正文）

**匹配关键词**：
- 社交图标：`Facebook / Twitter / LinkedIn / Pinterest / Instagram / YouTube`
- 分享按钮：`Share this / Tweet`
- 面包屑：`Home > / You are here:`
- 分页：`Page 1 of 5`
- 订阅/通讯：`Subscribe now / Newsletter Sign-up`
- 法律链接：`Copyright © 2023 / All Rights Reserved / Privacy Policy`

**激活量**：1,332 条文档（1.5%）

> **重要修复**：原正则末尾为 `[\s\S]*$`（跨行贪婪匹配），会误删多行内容。
> 已修复为 `[^\n]*$`（限制单行），通过单元测试验证。

---

### Rule 2 — CMS 编辑标记处理

| 标记 | 处理方式 |
|------|---------|
| `[INS: content :INS]` | 保留内容，去掉标记 |
| `[DEL: content :DEL]` | 整体删除 |
| 未闭合的 `[INS:` | 删除标记本身 |
| 未闭合的 `[DEL:...]` | 删除 |

**激活量**：85 条文档（0.1%）

---

### Rule 3 — 论坛用户元数据行删除

**匹配字段**：`Joined / Posts / Thanks / Reputation / Location / Member since / Likes received / Trophy points / Gender / Online Status / Last Seen / Registered / Messages`

格式：`Joined: February 2009` / `Posts: 81`（仅删除这类纯元数据行）

**激活量**：252 条文档（0.3%）

---

### Rule 4 — 错误答案段落删除

**触发条件**（保守策略，仅删除有明确标注的错误段落）：
- Markdown 标题：`## Incorrect Attempts` / `### Wrong Answers` / `## Common Mistakes`
- 加粗标签：`**WRONG**: ...` / `**Incorrect**: ...`
- 列表项标签：`1. **WRONG**: ...`

**不触发**：对话中的 "where did I go wrong?" 等上下文用法

**激活量**：39 条文档（0.0%）

---

### Rule 5 — LaTeX 定界符规范化

| 原格式 | 规范化后 |
|--------|---------|
| `\( ... \)` | `$ ... $` |
| `\[ ... \]` | `$$ ... $$` |

**激活量**：81,680 条文档（**89.4%**）— 影响最广的规则

---

### Rule 6 — 多余空行压缩

连续 3 行及以上空行 → 压缩为 2 行空行

**激活量**：1,010 条文档（1.1%）

---

### Rule 7（后处理）— 清洗后长度检查

清洗后文本长度 < 600 chars → 丢弃

**触发量**：0 条（所有文档清洗后仍 ≥ 600 chars）

---

## 输出文件

| 文件 | 记录数 | 说明 |
|------|--------|------|
| `stage2_output/stage2_clean.jsonl` | 91,331 | 清洗后全量数据，送入 Stage 3 |
| `stage2_output/stage2_discarded.jsonl` | 0 | 保留文件用于审计，当前为空 |
