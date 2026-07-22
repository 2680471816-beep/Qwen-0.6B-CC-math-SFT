# Stage 1 过滤日志

**脚本**：`src/data/stage1_filter.py` + `src/data/stage1_split_long.py`
**执行日期**：2026-04-08
**运行环境**：CPU only

---

## Stage 1A — 规则过滤

### 过滤规则

| 规则 | 条件 | 动作 |
|------|------|------|
| 1 | 文本长度 < 800 chars | 丢弃 |
| 2 | 文本长度 > 50,000 chars | 进入 long 队列 |
| 3 | `nemocurator_scores` < 1.5 | 丢弃 |
| 4 | 噪声密度 > 0.15 / 100 chars | 丢弃 |
| 5 | 数学内容密度 < 0.5 / 500 chars | 丢弃 |

### 输入/输出统计

| 项目 | 数值 |
|------|------|
| 输入总量 | 100,000 条 |
| 输出 normal | **86,011 条**（86.0%）|
| 输出 long 队列 | **253 条**（0.3%）|
| 丢弃总量 | **13,736 条**（13.7%）|

### 丢弃原因分布（估算）

| 原因 | 说明 |
|------|------|
| `too_short` | 文本长度 < 800 chars |
| `low_nemo` | nemocurator_scores < 1.5 |
| `noise_dense` | 噪声密度过高（URL、CTA、社交词等） |
| `low_math` | 数学关键词/表达式密度不足 |

### 关键正则表达式

**噪声指标**（NOISE_PATTERNS）：
- CMS 标记：`[INS:...]` `[DEL:...]`
- 网页 chrome 词：`cookie / advertisement / navbar / menu / sidebar / footer / header`
- 时间戳：`12:30 pm`
- 裸 URL：`https://...`
- CTA 短语：`click here / read more / subscribe / sign up`
- 社交统计：`views: 123 / likes: 45`

**数学指标**（MATH_PATTERNS）：
- 行内 LaTeX：`$...$`
- 展示 LaTeX：`$$...$$`
- LaTeX 命令：`\frac \int \sum` 等
- 数学关键词：`equation / theorem / proof / lemma / integral / derivative` 等
- 算术表达式：`3 + 5 = 8`
- 数学函数：`sin / cos / log / sqrt / lim`

---

## Stage 1B — 长文档语义切分

### 切分策略

1. 按 Markdown 标题（H1–H4）划分段落
2. 将 < 800 chars 的微小段落合并到前一段
3. 对 > 15,000 chars 的段落按空行强制切分（目标 8,000 chars）
4. 过滤掉数学密度 < 0.3 的块（目录页、纯导航等）

### 配置参数

| 参数 | 值 |
|------|----|
| MIN_CHUNK | 800 chars |
| TARGET_CHUNK | 8,000 chars |
| MAX_CHUNK | 15,000 chars |
| MIN_MATH_DENSITY | 0.3 / 500 chars |

### 输入/输出统计

| 项目 | 数值 |
|------|------|
| 输入文档数 | 253 条超长文档 |
| 输出语义块数 | **5,320 块** |
| 平均块/文档 | 21.0 块 |
| 平均块长度 | 3,880 chars |
| 块长分布 p10/p50/p90 | 1,028 / 2,603 / 7,966 chars |

### 输出文件

| 文件 | 记录数 | 说明 |
|------|--------|------|
| `stage1_output/stage1_normal.jsonl` | 86,011 | 通过过滤的正常长度文档 |
| `stage1_output/stage1_long.jsonl` | 253 | 原始超长文档（切分前备份）|
| `stage1_output/stage1_long_chunks.jsonl` | 5,320 | 切分后的语义块 |
| `stage1_output/stage1_discarded.jsonl` | 13,736 | 丢弃记录（含 filter_reason 字段）|

**Stage 1 合并后送入 Stage 2 的记录数：86,011 + 5,320 = 91,331 条**
