# EComCareLM AI Review Guide

> 本文件定义了项目的 invariants、审查优先级和输出格式标准。
> 所有 AI 代码审查（diff review / 全仓库审查 / 红队演练）都必须引用本文件。

---

## 项目 Invariants

以下条件是项目正确性的基础。任何代码变更不得破坏这些 invariants：

### 通用（所有项目适用）

#### 安全
- API Key / Token 不从代码读取，必须走环境变量或 secret store
- 用户输入必须做校验/转义/参数化查询
- 文件路径拼接必须使用 `Path` 而非字符串 `+`

#### 幂等
- 所有外部 webhook handler 必须幂等（idempotency key）
- 异步 job 必须安全地执行多次（at-least-once 语义）
- 数据库写操作必须有重试保护，避免重复扣款/重复创建

#### 错误处理
- 异常不得被 `except: pass` 吞掉
- 外部 API 调用必须设置 timeout，失败必须 fallback 或报错
- 错误消息不得暴露内部实现细节（堆栈、路径、SQL）

### 项目特定

#### 数据格式
- **JSONL 样本必含字段**：所有 SFT/DPO/PGDM 样本必须有 `id`、`prompt`、`chosen`/`output` 字段
- **评测集字段完整性**：`eval_set.jsonl` 每条必须包含 `case_id`、`scenario`、`question`、`gold_answer`、`must_include`、`must_not_include`
- **预测结果对齐**：`predictions` 必须与 `eval_set` 按 `case_id` 一一对应，不多不少
- **Policy 标注完整性**：`label-policy` 的输出必须覆盖 eval_set 中所有 case

#### 评测指标方向
- **指标方向一致性**：`forbidden_content` 和 `off_topic` 在顶层 metric 中是错误率（1.0=坏），在 `dimensions` 子 dict 中反转（1.0=好）
- **HRPVR**：只计 `over_promise_or_unsafe`、`policy_conflict`、`forbidden_content` 三类 badcase 为违规；`incomplete`、`impolite`、`off_topic`、`missing_prediction` 不计入
- **维度对齐**：`DIMENSION_KEYS` 在 `disagreement.py` 和 `normalize_*` 函数之间必须一致
- **分歧挖掘只比较双方都有的维度**：单方独有的维度（如 judge 的 `hallucination`、rule 的 `forbidden_content`）静默跳过

#### 数据安全
- **脱敏不可逆**：`clean` 命令后不得保留原始手机号、身份证、银行卡号、地址
- **API Key 不硬编码**：LLM Judge / DeepSeek 调用必须从环境变量读取
- **训练用的 JSONL 不得包含原始 PII**：`cleaning.py` 的 `sanitize_value` 必须覆盖所有文本字段

#### 训练语义
- **SFT loss**：只对 assistant response 计算，prompt 部分 labels 必须 mask 为 `-100`
- **DPO 数据集**：`chosen` 和 `rejected` 不能相同文本；metadata 必须标注 `source` 来源
- **GRPO reward**：返回值必须在 `[0.0, 1.0]` 范围内

#### 幂等
- **所有 CLI 命令**：对同一输入多次运行，在相同 seed 下必须产生相同的输出
- **评测打分**：`evaluate_cases` 对相同的 eval_set + predictions 每次输出一致

#### 导入与打包
- `train/` 内的脚本通过 `from train.xxx import` 显式导入，不使用隐式相对导入
- 所有 `case_id` 缺失时必须使用一致的 fallback 逻辑（`f"{prefix}_{idx:05d}"`）

---

## 审查优先级

1. **P0 — Invariant 破坏**：任何违反上述 invariants 的变更
2. **P0 — 数据损坏**：导致 JSONL 格式不对、字段缺失、类型错误
3. **P1 — 指标失真**：HRPVR / disagreement 统计逻辑错误，metric 方向混淆
4. **P1 — 安全/隐私**：PII 泄露、API Key 暴露、越权数据访问
5. **P2 — 测试缺口**：高风险路径（退款/投诉/隐私场景）缺少测试覆盖
6. **P2 — 竞态条件/幂等性**：重复执行导致数据不一致
7. **P3 — 边界条件**：空 JSONL、重复 case_id、超过 max_length 截断
8. **P3 — 文档/注释与代码不一致

---

## 不报告的项

- 格式/缩进问题（由 formatter 处理）
- 命名偏好（除非严重影响可读性）
- 微小的重构建议（不影响正确性）
- 无法从代码或文档推断的行为假设

---

## 输出格式

每个问题必须包含：

```yaml
severity: critical | high | medium | low
location: 文件:行号 (函数名)
trigger: 如何触发
impact: 会造成什么后果
evidence: 从代码推断的依据
reproduction: 最小复现步骤或测试思路
fix: 最小修复建议
regression_test: 应该新增什么测试
```

不确定的问题标注 `needs_confirmation: true`。

---

## 参考文件

- [BUG_LEDGER.md](BUG_LEDGER.md) — 已发现和已修复的问题记录
- [docs/公开数据集.md](docs/公开数据集.md) — 数据源说明
- [ecomcarelm/policy.py](ecomcarelm/policy.py) — Policy taxonomy 定义
- [pyproject.toml](pyproject.toml) — 测试配置和可选的 extra
