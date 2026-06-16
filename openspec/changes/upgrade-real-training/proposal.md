# 变更提案：升级为可真实训练项目

## 目标

- 把 EComCareLM 从“数据与评测 MVP”升级为“可直接训练的垂域模型对齐项目”。
- 增加公开数据集导入能力，支持从 Hugging Face datasets、本地 JSON/JSONL、JDDC/EcomGPT 类数据转换为统一格式。
- 增加项目内自有 SFT 训练脚本，不再只依赖 LLaMA-Factory YAML。
- 增加 DPO 训练入口，支持基于真实模型输出 badcase 构造 chosen/rejected。
- 增加 LLM Judge 评测入口，让 keyword matching 只作为轻量 baseline。

## 非目标

- 不在仓库中内置大规模公开数据原文，避免许可证和体积问题。
- 不承诺在无 GPU 环境下完成 7B/8B 模型训练。
- 不伪造训练结果；README 中只提供运行方法和结果记录模板。

## 影响范围

- 新增公开数据集说明：`docs/公开数据集.md`。
- 新增数据导入模块：`ecomcarelm/datasets.py`。
- 新增训练脚本：`train/train_sft.py`、`train/train_dpo.py`。
- 新增 LLM Judge：`ecomcarelm/judge.py`。
- 扩展 CLI：公开数据导入、基于预测构造 DPO、LLM Judge。
- 扩展测试，保证无训练依赖时基础链路仍然可跑。
