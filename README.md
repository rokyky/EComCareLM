# EComCareLM

EComCareLM 是一个中文电商客服垂域模型对齐项目。它现在不只是数据处理 demo，而是包含公开数据导入、SFT 训练、DPO 训练、规则评测、LLM Judge 和 badcase 迭代的完整链路。

它和 bitalab 的区分是：

- bitalab：Agent 平台工程，重点是 MCP、工具编排、Memory/RAG、Runner、Session 和部署。
- EComCareLM：模型能力工程，重点是垂域数据构建、SFT、DPO、自动评测、LLM Judge 和失败案例驱动的数据迭代。

## 已实现能力

- 公开数据导入：支持 Hugging Face datasets，支持字段映射、限制样本数和 train/dev/test 切分。
- 数据清洗：手机号、订单号、快递单号、银行卡号、地址等脱敏，近重复样本过滤。
- SFT 数据构建：统一生成 instruction/input/output 格式。
- DPO 数据构建：支持模板负样本，也支持从模型预测 badcase 构造 chosen/rejected。
- 真实 SFT 训练：`train/train_sft.py`，基于 Transformers Trainer，支持 LoRA/QLoRA。
- 真实 DPO 训练：`train/train_dpo.py`，基于 TRL DPOTrainer，支持 LoRA/QLoRA。
- GRPO 训练：`train/train_grpo.py`，基于 TRL GRPOTrainer 和电商客服规则 reward。
- PPO 路线检查：`train/train_ppo.py`，用于说明 PPO 前置条件和 reward model 依赖。
- 规则评测：可离线复现，适合快速回归。
- LLM Judge：支持 OpenAI-compatible API，输出逐样本质检 JSONL。
- 报告生成：输出整体指标、分场景指标、badcase 统计和示例。

## 快速验证

```powershell
cd D:\my-projects\EComCareLM
$env:UV_LINK_MODE = "copy"
uv run --with pytest pytest -q
.\scripts\run_sample_pipeline.ps1
```

生成结果在 `data/processed/`。

Windows PowerShell 直接 `Get-Content` 读取无 BOM 的 JSONL 时可能显示乱码，但文件本身是标准 UTF-8，训练框架可正常读取。人工查看样本建议用：

```powershell
uv run python -m ecomcarelm peek --input data/processed/dpo_from_badcase.jsonl -n 2
```

## 公开数据集

数据源说明见 [docs/公开数据集.md](docs/公开数据集.md)。

示例：从 Hugging Face 数据集导入并做字段映射：

```powershell
uv sync --extra data
uv run python -m ecomcarelm import-hf `
  --dataset DATASET_NAME `
  --split train `
  --question-field question `
  --answer-field answer `
  --scenario public_ecommerce `
  --limit 10000 `
  --train-output data/processed/public_train.jsonl `
  --dev-output data/processed/public_dev.jsonl `
  --test-output data/processed/public_test.jsonl
```

不同公开数据集字段名不一样，所以这里不写死字段，而是通过 `--question-field`、`--answer-field`、`--context-field`、`--policy-field` 映射。

## 数据构建

```powershell
uv run python -m ecomcarelm clean --input data/raw/ecommerce_faq_sample.jsonl --output data/processed/cleaned.jsonl
uv run python -m ecomcarelm build-sft --input data/processed/cleaned.jsonl --output data/processed/sft_train.jsonl
uv run python -m ecomcarelm build-dpo --input data/processed/sft_train.jsonl --output data/processed/dpo_train_template.jsonl
```

从真实模型 badcase 构造 DPO：

```powershell
uv run python -m ecomcarelm build-dpo-from-predictions `
  --eval-set data/samples/eval_set_v1.jsonl `
  --predictions data/samples/bad_predictions_sample.jsonl `
  --output data/processed/dpo_from_badcase.jsonl
```

面试时建议强调：DPO 的 rejected 优先来自模型真实错误回答；模板负样本只用于冷启动。

## 真实训练

安装训练依赖：

```powershell
uv sync --extra train
```

SFT smoke test：

```powershell
uv run python train/train_sft.py `
  --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct `
  --train-file data/processed/sft_train.jsonl `
  --output-dir outputs/ecomcarelm-sft-smoke `
  --use-lora `
  --load-in-4bit `
  --max-steps 5
```

DPO smoke test：

```powershell
uv run python train/train_dpo.py `
  --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct `
  --train-file data/processed/dpo_from_badcase.jsonl `
  --output-dir outputs/ecomcarelm-dpo-smoke `
  --use-lora `
  --load-in-4bit `
  --max-steps 5
```

GRPO smoke test：

```powershell
uv run python train/train_grpo.py `
  --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct `
  --train-file data/samples/eval_set_v1.jsonl `
  --output-dir outputs/ecomcarelm-grpo-smoke `
  --use-lora `
  --max-steps 5
```

PPO 路线检查：

```powershell
uv run python train/train_ppo.py `
  --sft-model-path outputs/ecomcarelm-sft `
  --reward-model-path outputs/ecomcarelm-reward-model `
  --train-file data/processed/sft_train.jsonl
```

PPO 不建议作为第一阶段主线。它需要稳定 reward model/value 训练，工程成本比 DPO/GRPO 高；如果没有 reward model，硬写 PPO 只是在堆名词。这个项目更合理的顺序是：

1. LoRA/QLoRA SFT：学会客服话术和规则格式。
2. DPO：用模型真实 badcase 做 chosen/rejected 偏好对齐。
3. GRPO：在有可执行规则 reward 后做在线强化。
4. PPO：只有当 reward model 足够可靠时再考虑。

正式训练建议：

- 小成本实验：Qwen2.5-0.5B/1.5B + LoRA。
- 简历强版本：Qwen2.5-7B 或 Qwen3-8B + QLoRA。
- 进阶亮点：DPO 后接 GRPO，用规则 reward 优化规则遵循、完整性和安全边界。
- 指标必须来自真实训练结果，不要提前编造。

## 评测

规则评测：

```powershell
uv run python -m ecomcarelm demo --eval-set data/samples/eval_set_v1.jsonl --output data/processed/predictions.jsonl
uv run python -m ecomcarelm eval --eval-set data/samples/eval_set_v1.jsonl --predictions data/processed/predictions.jsonl --output data/processed/eval_results.jsonl
uv run python -m ecomcarelm report --results data/processed/eval_results.jsonl --output data/processed/eval_report.md
```

LLM Judge：

```powershell
uv sync --extra judge
$env:OPENAI_API_KEY = "你的 key"
uv run python -m ecomcarelm judge `
  --eval-set data/samples/eval_set_v1.jsonl `
  --predictions data/processed/predictions.jsonl `
  --output data/processed/judge_results.jsonl `
  --model gpt-4o-mini
```

## 面试讲法

可以这样讲：

> 这个项目我做的是电商客服垂域模型对齐，不是再搭一个客服 RAG。数据层面，我实现了公开数据集导入和字段映射，把 JDDC/EcomInstruct/Hugging Face 电商 QA 类数据统一成 SFT 和 DPO 格式；训练层面，我用 LoRA/QLoRA 做 SFT，用 TRL DPOTrainer 做偏好对齐，还提供了基于规则 reward 的 GRPO 训练入口；评测层面，先用规则评测做可复现回归，再用 LLM Judge 做更接近质检的打分。DPO 的 rejected 不是只写死模板，而是优先来自模型在评测集上的真实 badcase，然后把这些错误回答和 gold answer 组成偏好对，形成 badcase-driven 的数据迭代闭环。

这个项目的价值是补上 bitalab 没有覆盖的“模型训练和对齐能力”。bitalab 证明我能做 Agent 平台工程，EComCareLM 证明我能做垂域数据、训练、偏好对齐和评测。
