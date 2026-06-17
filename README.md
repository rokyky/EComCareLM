# EComCareLM

EComCareLM 是一个中文电商客服垂域模型对齐项目，核心方法是 **Policy-Grounded Disagreement Mining (PGDM)**——通过规则评测与 LLM Judge 的维度级分歧，自动挖掘高价值 hard negatives，构造策略加权的 DPO 偏好对，降低高风险客服场景的违规率。

与 [bitalab](https://github.com/leinao/bitalab) 的区分：
- **bitalab**：Agent 平台工程，MCP、工具编排、Memory/RAG、Runner、Session 和部署。
- **EComCareLM**：模型能力工程，垂域数据构建、SFT、DPO/PGDM-DPO、自动评测、LLM Judge 和 badcase 驱动的数据迭代。

## PGDM 方法概述

```
JDDC Baseline 对话数据         自构 SOP / 规则
        │                            │
        └─────┬──────────────────────┘
              │
        SFT 训练（Qwen2.5 + LoRA）
              │
        模型预测 → 评测集
              │
    ┌─────────┼────────────┐
    │         │            │
    规则评测  LLM Judge   Policy Labeler
    │         │            │
    └────┬────┘            │
         │                 │
   维度级分歧挖掘           │
    (disagreement.py)      │
         │                 │
    ┌────┴─────────────────┘
    │
    PGDM-DPO 数据构建
    (severity × disagreement_strength × confidence 加权)
    │
    6 组消融对比:
    Base → SFT → Template-DPO → Badcase-DPO → PGDM-DPO (full)
    │
    HRPVR 指标验证（高风险场景违规率）
```

## 代码结构

```
ecomcarelm/
├── policy.py              Policy taxonomy（6 类 + severity + high-risk）
├── policy_labeler.py      Policy violation labeler（规则 + LLM）
├── disagreement.py        维度级分歧挖掘（rule vs judge）
├── evaluation.py          规则评测（7 维度）
├── judge.py               LLM Judge + 维度对齐
├── builders.py             SFT / DPO / PGDM-DPO 数据构建
├── datasets.py             HuggingFace 数据集导入 + badcase DPO
├── cleaning.py            脱敏 + 去重
├── baseline.py            规则 baseline 生成
├── report.py              Markdown 报告
└── cli.py                 13 个子命令（含 PGDM 相关 4 个）
```

## 已实现能力

- **JDDC Baseline 数据导入**：`scripts/import_jddc_baseline.py` — 从 [JDDC-Baseline-Seq2Seq](https://github.com/SimonJYang/JDDC-Baseline-Seq2Seq) 导入约 1 万 session 京东真实客服脱敏对话，自动解析 QA 对、推断场景、输出 SFT + Eval 格式。
- **Policy Taxonomy**：refund_policy / privacy_policy / escalation_policy（高风险）/ logistics_policy / evidence_policy / tone_policy。
- **Policy Labeler**：规则匹配 + LLM 标注，输出 `policy_id / violation_type / confidence / evidence`。
- **维度级分歧挖掘**：比较规则评测和 LLM Judge 在 7 个维度上的分数差异，输出 aggregate + dimension 两级分歧。
- **PGDM-DPO 数据构建**：按 `severity × disagreement_strength × label_confidence` 加权采样，per-policy quota cap 防止大类垄断。
- **HRPVR 指标**：High-Risk Policy Violation Rate，主指标聚焦 refund / privacy / escalation 三类高风险场景。
- **SFT 训练**：`train/train_sft.py`，基于 Transformers Trainer，LoRA/QLoRA。
- **DPO 训练**：`train/train_dpo.py`，基于 TRL DPOTrainer，LoRA/QLoRA。
- **GRPO 训练**：`train/train_grpo.py`，基于 TRL GRPOTrainer + 自定义规则 reward。
- **规则评测**：7 维度规则评测（可离线复现，快速回归）。
- **LLM Judge**：支持 OpenAI-compatible API（含 DeepSeek），输出逐样本质检 JSONL。
- **报告生成**：整体指标 + 分场景 + badcase 统计 + top-5 示例。

## CLI 命令（13 个）

| 命令 | 功能 | 新增 |
|------|------|------|
| `import-hf` | 从 HuggingFace 导入数据集 | — |
| `clean` | 脱敏 + 去重 | — |
| `build-sft` | 构建 SFT 格式 | — |
| `build-dpo` | 模板负样本 DPO 构建 | — |
| `build-dpo-from-predictions` | badcase 驱动 DPO 构建 | — |
| **`label-policy`** | Policy violation 标注 | ✅ |
| **`mine-disagreements`** | 规则 vs LLM Judge 分歧挖掘 | ✅ |
| **`build-pgdm-dpo`** | PGDM-DPO 偏好对构建 | ✅ |
| **`hrpvr`** | 高风险场景违规率计算 | ✅ |
| `demo` | 规则 baseline 预测 | — |
| `eval` | 规则评测打分 | — |
| `judge` | LLM Judge 质检 | — |
| `report` | Markdown 报告 | — |

## 快速开始

### 前置条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) 包管理器
- （可选）NVIDIA GPU + CUDA，或 AutoDL / 阿里云等 GPU 云实例
- （可选）DeepSeek / OpenAI API Key（用于 LLM Judge）

### 安装

```bash
git clone <repo-url> && cd EComCareLM
uv sync --extra train    # 训练依赖（torch, transformers, trl, peft）
uv sync --extra judge    # 评测依赖（openai）
```

### 第 1 步：获取数据

使用 `scripts/import_jddc_baseline.py` 从 [JDDC-Baseline-Seq2Seq](https://github.com/SimonJYang/JDDC-Baseline-Seq2Seq) 导入约 1 万 session 京东真实客服对话：

```bash
# 手动下载 chat.txt：
#   https://raw.githubusercontent.com/SimonJYang/JDDC-Baseline-Seq2Seq/master/data/chat.txt
# 放到 data/generated/jddc_chat.txt

uv run python scripts/import_jddc_baseline.py --sft-count 3000 --eval-count 500
```

产出：
- `data/generated/sft_train.jsonl` — 3000 条 SFT 训练数据
- `data/generated/eval_set.jsonl` — 500 条评测数据（含场景分类）

> ⚠️ **JDDC 数据限制**
> JDDC 是真实客服对话，没有人工精标的标准答案和平台规则标注。
> - `gold_answer` 取自客服原始回复，**不是**标准答案，而是"弱标注"参考
> - `policy` 字段为空字符串（JDDC 没有规则标注）
> - `must_include` / `must_not_include` 为空数组 `[]`（无关键词标注），`_coverage(answer, [])` 返回 1.0 满分
> - 评测指标依赖 `policy` 的场景（如 `policy_compliance`）在 JDDC 数据上精度有限
>
> **建议**：HRPVR 等指标的绝对值仅供参考，应关注**不同方法之间的相对趋势**（如 PGDM-DPO vs Template-DPO 的 HRPVR 差距）。

### 第 2 步：SFT 训练（0.5B 验证链路）

```bash
uv run python train/train_sft.py \
  --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct \
  --train-file data/generated/sft_train.jsonl \
  --output-dir outputs/sft-0.5b \
  --use-lora --load-in-4bit --num-train-epochs 3
```

### 第 3 步：基线评测 + Policy 标注

```bash
# 规则 baseline 预测
uv run python -m ecomcarelm demo \
  --eval-set data/generated/eval_set.jsonl \
  --output outputs/pred-baseline.jsonl

# 规则评测
uv run python -m ecomcarelm eval \
  --eval-set data/generated/eval_set.jsonl \
  --predictions outputs/pred-baseline.jsonl \
  --output outputs/rule-baseline.jsonl

# Policy 标注
uv run python -m ecomcarelm label-policy \
  --eval-set data/generated/eval_set.jsonl \
  --predictions outputs/pred-baseline.jsonl \
  --output outputs/policy-labels.jsonl

# HRPVR 基线
uv run python -m ecomcarelm hrpvr \
  --rule-results outputs/rule-baseline.jsonl \
  --policy-labels outputs/policy-labels.jsonl \
  --output outputs/hrpvr-baseline.json
```

### 第 4 步：LLM Judge + 分歧挖掘 + PGDM-DPO

```bash
# LLM Judge（需 API Key）
export OPENAI_API_KEY="sk-xxx"
# 或 DeepSeek：
# export OPENAI_BASE_URL=https://api.deepseek.com

uv run python -m ecomcarelm judge \
  --eval-set data/generated/eval_set.jsonl \
  --predictions outputs/pred-baseline.jsonl \
  --output outputs/judge-baseline.jsonl \
  --model deepseek-chat

# 维度级分歧挖掘
uv run python -m ecomcarelm mine-disagreements \
  --rule-results outputs/rule-baseline.jsonl \
  --judge-results outputs/judge-baseline.jsonl \
  --output outputs/disagreements.jsonl

# PGDM-DPO 数据构建
uv run python -m ecomcarelm build-pgdm-dpo \
  --eval-set data/generated/eval_set.jsonl \
  --predictions outputs/pred-baseline.jsonl \
  --policy-labels outputs/policy-labels.jsonl \
  --disagreements outputs/disagreements.jsonl \
  --output data/dpo_pgdm.jsonl \
  --max-items 1000
```

### 第 5 步：6 组 DPO 消融实验

```bash
# ① Template-DPO（基线）
uv run python -m ecomcarelm build-dpo \
  --input data/generated/sft_train.jsonl \
  --output data/dpo_template.jsonl
uv run python train/train_dpo.py \
  --model-name-or-path outputs/sft-0.5b \
  --train-file data/dpo_template.jsonl \
  --output-dir outputs/dpo-template \
  --use-lora --load-in-4bit --num-train-epochs 3

# ② Badcase-DPO
uv run python -m ecomcarelm build-dpo-from-predictions \
  --eval-set data/generated/eval_set.jsonl \
  --predictions outputs/pred-baseline.jsonl \
  --output data/dpo_badcase.jsonl
uv run python train/train_dpo.py \
  --model-name-or-path outputs/sft-0.5b \
  --train-file data/dpo_badcase.jsonl \
  --output-dir outputs/dpo-badcase \
  --use-lora --load-in-4bit --num-train-epochs 3

# ③ PGDM-DPO
uv run python train/train_dpo.py \
  --model-name-or-path outputs/sft-0.5b \
  --train-file data/dpo_pgdm.jsonl \
  --output-dir outputs/dpo-pgdm \
  --use-lora --load-in-4bit --num-train-epochs 3
```

对每组 DPO 输出重新跑 `eval` + `hrpvr`，对比 HRPVR 指标。

### 第 6 步：换 7B 出正式结论

0.5B 验证趋势正确后，将 `--model-name-or-path` 替换为 `Qwen/Qwen2.5-7B-Instruct`，其余参数不变。7B QLoRA 在单卡 24GB 上 SFT 约 75 分钟，DPO 约 45 分钟/组。

建议实验组：

| 组别 | 说明 |
|------|------|
| Base | 原始 Qwen2.5-7B-Instruct |
| SFT | 普通客服 SFT |
| Template-DPO | 模板负样本基线 |
| Badcase-DPO | 模型错误回答驱动 |
| PGDM-DPO (aggregate only) | 只用整体分歧 |
| PGDM-DPO (no policy) | 去掉 policy attribution |
| PGDM-DPO (no weighting) | 去掉策略加权 |
| PGDM-DPO (full) | **完整方法** |

## 数据源

本项目主力数据源为 [JDDC-Baseline-Seq2Seq](https://github.com/SimonJYang/JDDC-Baseline-Seq2Seq) 中约 1 万 session 的京东客服脱敏对话数据，辅以自构电商 SOP / FAQ 问答。详细说明见 [docs/公开数据集.md](docs/公开数据集.md)。

## 面试准备

面试前请重点阅读 [interview/QA.md](interview/QA.md)（153 题完整回答）和 [interview/onlyQ.md](interview/onlyQ.md）（问题清单快速自测）。两文件按"面试回答→面试官视角→项目结合"三段式组织，覆盖项目总览、数据构建、SFT/LoRA、DPO、PGDM、GRPO、PPO、评测、模型架构、推理部署、Agent 面试等 13 个分类。

## 引用

- JDDC-Baseline-Seq2Seq: https://github.com/SimonJYang/JDDC-Baseline-Seq2Seq
- JDDC Corpus, LREC 2020: https://aclanthology.org/2020.lrec-1.58/
- EcomGPT / EcomInstruct, AAAI 2024: https://ojs.aaai.org/index.php/AAAI/article/view/29820
