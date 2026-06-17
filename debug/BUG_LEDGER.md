# EComCareLM Bug Ledger

> 所有被模型或人工发现的问题在此记录。
> 每个 bug 修复后必须追加 regression test 和新增规则。
> 每次 AI review 前喂入此文件，避免重复报告已知问题。

---

## 统计

| 总数 | 致命 | 高 | 中 | 低 | 已修复 |
|------|------|----|----|----|--------|
| 14 | 5 | 5 | 2 | 2 | 14 |

---

## 修复记录

### BL-001: `judge.py` — `_score_0_to_1` 归一化条件错误

- **严重度**: 🔴 致命
- **发现时间**: 2026-06-16
- **Found by**: 人工审查
- **文件**: `ecomcarelm/judge.py:13`
- **问题**: `if number > 1.0:` 条件导致 LLM Judge 返回的 `accuracy=1`（0-5 分制，本应为 0.2）被当作已归一化，保留为满分 1.0。1 分和 5 分结果完全相同。
- **修复**: 改为 `if number >= 1.0:`，确保 0-5 分制下 1 分也被除以 5。
- **影响**: LLM Judge 所有维度中返回 1 分的回答被错误膨胀为满分，扭曲质检分数。
- **Regression test**: 需新增：`_score_0_to_1(1)` 应返回 `0.2`，`_score_0_to_1(5)` 应返回 `1.0`
- **Status**: ✅ 已修复

---

### BL-002: `judge.py` — `parse_judge_json` 反引号解析不安全

- **严重度**: 🟡 中等
- **发现时间**: 2026-06-16
- **Found by**: 人工审查
- **文件**: `ecomcarelm/judge.py:105-110`
- **问题**: 使用 `stripped.strip("`")` 移除字符串两端所有反引号字符。当 LLM 返回的 `reason` 字段包含反引号时，内部反引号也被消除，JSON 解析失败。
- **修复**: 改用 regex 只移除首尾的 markdown 代码块包裹层。
- **影响**: 特定场景下 LLM Judge 结果解析失败。
- **Status**: ✅ 已修复

---

### BL-003: `evaluation.py` — `must_include` 为空时所有回答被判 off_topic

- **严重度**: 🔴 致命
- **发现时间**: 2026-06-16
- **Found by**: 人工审查
- **文件**: `ecomcarelm/evaluation.py:61`
- **问题**: `off_topic = bool(answer) and not _contains_any(answer, must_include) and ...`。当 `must_include` 为空列表时，`_contains_any(answer, [])` 返回 `False`，`not False` 为 `True`，任何非空回答都被标记为 off_topic。
- **修复**: 加 `bool(must_include)` 前置检查，空列表时不做 off_topic 判断。
- **影响**: 破坏整个评测体系，大量正常回答被标记答非所问。
- **Status**: ✅ 已修复

---

### BL-004: `train/train_grpo.py` — `from rewards` 隐式相对导入

- **严重度**: 🟠 高
- **发现时间**: 2026-06-16
- **Found by**: 人工审查
- **文件**: `train/train_grpo.py:8`
- **问题**: `from rewards import 客服规则奖励` 是隐式相对导入。`python train/train_grpo.py` 时 Python 将 `train/` 加入 sys.path 可工作，但 `python -m train.train_grpo` 或打包后导入失败。
- **修复**: 改为 `from train.rewards import 客服规则奖励`。
- **影响**: GRPO 训练脚本无法通过 `python -m` 或某些打包方式启动。
- **Status**: ✅ 已修复

---

### BL-005: `train/train_grpo.py` — GRPOTrainer 缺 tokenizer

- **严重度**: 🟠 高
- **发现时间**: 2026-06-16
- **Found by**: 人工审查
- **文件**: `train/train_grpo.py:104-110`
- **问题**: GRPOTrainer 未传入 `processing_class`（或 `tokenizer`）参数。TRL >= 0.15 需要此参数对 prompt 分词。
- **修复**: 加载 `AutoTokenizer` 并作为 `processing_class=tokenizer` 传入 GRPOTrainer。
- **影响**: GRPO 训练完全不可用。
- **Status**: ✅ 已修复

---

### BL-006: `case_id` 匹配失败 — evaluation.py / datasets.py / policy_labeler.py

- **严重度**: 🟠 高
- **发现时间**: 2026-06-16
- **Found by**: 人工审查
- **文件**: `ecomcarelm/evaluation.py:89`、`ecomcarelm/policy_labeler.py:108`、`ecomcarelm/datasets.py:135`
- **问题**: `prediction_by_id` 用 `str(item.get("case_id"))` 构建 key（当 `case_id=None` 时为 `"None"`），而 eval_cases 侧用 `str(case.get("case_id") or f"eval_{idx:05d}")`。两者 fallback 逻辑不一致，导致 `case_id` 缺失的数据匹配失败，得 0 分且无警告。
- **修复**: `prediction_by_id` 构建时改用与 eval_cases 侧一致的 fallback 逻辑，两端枚举索引独立。
- **影响**: 无 `case_id` 的数据在评测中静默拿到全 0 分。
- **Status**: ✅ 已修复

---

### BL-007: `policy.py` — HRPVR 误计非违规 badcase

- **严重度**: 🔴 致命
- **发现时间**: 2026-06-16
- **Found by**: Claude Code / bug-check 深度审查
- **文件**: `ecomcarelm/policy.py:122`
- **问题**: `high_risk_policy_violation_rate` 中用 `bool(case.get("badcase_type"))` 检测违规。但 `badcase_type` 包括 `"impolite"`、`"incomplete"`、`"missing_prediction"` 等非高风险违规的类别，`"impolite"` 的 case 被误计为 high-risk 违规。
- **修复**: 新增 `HRPVR_RELEVANT_BADCASES` 白名单，只计 `over_promise_or_unsafe`、`policy_conflict`、`forbidden_content` 三类。
- **Rule added**: AI_REVIEW_GUIDE.md — HRPVR 只计三类 badcase
- **Regression test**: 需新增：构造 `badcase_type="impolite"` 的 case，验证不计为 HRPVR 违规
- **Status**: ✅ 已修复

---

### BL-008: `policy_labeler.py` — 未知策略跳过过度承诺检测

- **严重度**: 🟡 中等
- **发现时间**: 2026-06-16
- **Found by**: Claude Code / bug-check 深度审查
- **文件**: `ecomcarelm/policy_labeler.py:61-64`
- **问题**: 当 `scenario_policy == UNKNOWN_POLICY_ID` 时，`else` 分支追加 `(UNKNOWN_POLICY_ID, "over_promise", OVER_PROMISE_MARKERS)`，但循环第一行 `if policy_id == UNKNOWN_POLICY_ID: continue` 直接跳过。导致无法推断场景的回答不会被检查过度承诺。
- **修复**: 改用 `scenario_policy if scenario_policy != UNKNOWN_POLICY_ID else "refund_policy"` 兜底。
- **Regression test**: 需新增：传入无 scenario 的 case，验证 over-promise 标记仍被检查
- **Status**: ✅ 已修复

---

### BL-009: `train/train_sft.py` — SFT loss 包含 prompt token

- **严重度**: 🟠 高
- **发现时间**: 2026-06-16
- **Found by**: Claude Code / bug-check 深度审查
- **文件**: `train/train_sft.py:tokenize`
- **问题**: `labels = input_ids.copy()` 导致系统提示词和用户问题的 token 也参与 loss 计算。行业标准做法是只对 assistant response 算 loss，prompt 部分 mask 为 `-100`。
- **修复**: tokenize 中从后往前搜索 `RESPONSE_MARKER`，prompt 部分 labels 设为 `-100`。
- **Rule added**: AI_REVIEW_GUIDE.md — SFT loss 只对 assistant response 计算
- **Regression test**: 需新增：tokenize 后验证 labels 中 prompt 部分全为 `-100`
- **Status**: ✅ 已修复

---

### BL-010: `train/train_sft.py` — 空 JSONL 无防御

- **严重度**: 🟢 低
- **发现时间**: 2026-06-16
- **Found by**: Claude Code / bug-check 深度审查
- **文件**: `train/train_sft.py:126`
- **问题**: `train_records[0].keys()` 在空列表时抛出 `IndexError`。
- **修复**: 添加 `if not train_records: raise ValueError(...)` 显式检查。
- **Regression test**: 需新增：传入空 JSONL 验证抛 `ValueError`
- **Status**: ✅ 已修复

---

### BL-011: `train/__init__.py` — 目录缺少包声明

- **严重度**: 🟢 低
- **发现时间**: 2026-06-16
- **Found by**: Claude Code / bug-check 深度审查
- **文件**: `train/` 目录
- **问题**: `train/` 无 `__init__.py`，测试通过 namespace packages 可工作，但在打包/部署场景下 `from train.rewards import` 可能失败。
- **修复**: 创建 `train/__init__.py`。
- **Status**: ✅ 已修复

---

## 已排除的疑似 bug

以下项目在审查中被标记但确认不是 bug：

| 项目 | 结论 |
|------|------|
| `judge.py` accuracy 只映射 `policy_compliance`（不再映射到 `answer_accuracy`） | ✅ 正确的设计。PGDM 方法的核心就是两个评测器用不同方式测量同一维度，其分歧正好是训练信号。 |
| `hallucination` 维度不在 `DIMENSION_KEYS` 中（已被 `forbidden_content` 替代） | ✅ 正确的设计。Judge 的 `hallucination`（是否编造规则/事实）和 rule 的 `forbidden_content`（是否包含违禁词）是不同的概念，不应强行对齐。 |
| 缺少 `DIMENSION_KEYS` 的维度在分歧挖掘中丢弃 | ✅ 正确的设计。分歧挖掘只比较两个评测器都有的维度，单方独有的维度静默跳过。 |

---

### BL-012: `train/train_grpo.py` — `from trl import ...` 缩进错误

- **严重度**: 🔴 致命
- **发现时间**: 2026-06-16（第 2 轮审查）
- **Found by**: Claude Code / 语法扫射
- **文件**: `train/train_grpo.py:16`
- **问题**: 上一轮 BL-005 修复（添加 `AutoTokenizer`）后在 `_load_grpo_deps()` 内新增 `from trl import GRPOConfig, GRPOTrainer` 时缩进级别错误（0 空格而非 8 空格），函数无法被 Python 解析，`SyntaxError: expected 'except' or 'finally' block`。
- **修复**: 修正缩进为 8 空格，与 try 块内其他 import 一致。
- **Status**: ✅ 已修复

---

### BL-013: `train/train_grpo.py` — `AutoTokenizer` 未从依赖工厂函数返回

- **严重度**: 🟠 高
- **发现时间**: 2026-06-16（第 2 轮审查）
- **Found by**: Claude Code / 数据流追踪
- **文件**: `train/train_grpo.py:15,19,80`
- **问题**: BL-005 修复在 `_load_grpo_deps()` 中加了 `from transformers import AutoTokenizer`（15 行），但 19 行的 `return` 语句未包含它。`main()` 中 `AutoTokenizer.from_pretrained(...)` 调用时 `NameError`。
- **修复**: `_load_grpo_deps()` 返回值追加 `AutoTokenizer`，`main()` 解包对应增加一个变量。
- **Status**: ✅ 已修复

---

### BL-014: `train/train_sft.py` — `merge_and_unload()` 返回值未接住

- **严重度**: 🔴 致命
- **发现时间**: 2026-06-16（第 4 轮审查）
- **Found by**: Claude Code / 数据流追踪
- **文件**: `train/train_sft.py:179`
- **问题**: Fix 7 新增 LoRA 合并逻辑，但 `model.merge_and_unload()` 的返回值未被捕获。`model` 仍然是 PeftModel，后续 `model.save_pretrained()` 再次保存 adapter 而非合并后的完整模型权重。DPO 加载时仍缺基座权重。
- **修复**: `model = model.merge_and_unload()`，用返回值覆盖原变量。
- **Status**: ✅ 已修复

## 验收

所有修复后 `pytest -q tests/test_pipeline.py` 7/7 通过。
