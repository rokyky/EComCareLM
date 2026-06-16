# 设计说明：真实训练升级

## 总体方案

升级后的项目分成四层：

- 数据层：公开数据集导入、本地数据转换、字段映射、抽样、切分。
- 训练层：项目内 SFT/DPO/GRPO 训练脚本，支持 Qwen 等 Hugging Face causal LM。
- 评测层：规则评测作为可复现 baseline，LLM Judge 作为更接近真实质检的评测方式。
- 迭代层：从模型预测 badcase 中生成 DPO rejected，不再只依赖硬编码模板。

## 数据策略

公开数据源按优先级分为：

1. JDDC / JDDC 2.1：真实中文电商客服多轮对话，适合构建客服回复 SFT 数据。
2. EcomGPT / EcomInstruct：电商指令数据，适合补充商品理解、评论理解、标题生成、属性问答等任务。
3. Hugging Face 上的电商客服/电商 QA 数据：作为可快速复现实验的数据入口。
4. 自构 SOP 与规则样本：用于补齐退货、物流、发票、投诉、安全拒答等业务边界。

## 训练策略

SFT：

- 输入：统一 SFT JSONL。
- 模型：默认 `Qwen/Qwen2.5-0.5B-Instruct`，便于低成本 smoke test；正式实验可改为 1.5B/7B。
- 方法：Transformers Trainer + LoRA/QLoRA。
- 输出：adapter 或完整模型目录。

DPO：

- 输入：统一 DPO JSONL。
- chosen：标准答案或人工改写答案。
- rejected：真实模型 badcase 优先；没有真实 badcase 时才使用模板负样本。
- 方法：优先使用 TRL DPOTrainer，并支持 LoRA/QLoRA；环境没有 TRL 时给出清晰错误。

GRPO：

- 输入：带 `must_include` 和 `must_not_include` 的评测/规则样本。
- reward：规则覆盖、礼貌性、长度完整性加分；过度承诺、违规协助、禁用表达扣分。
- 方法：TRL GRPOTrainer + 可选 LoRA。

PPO：

- PPO 需要稳定 reward model/value 结构，不适合作为第一阶段主线。
- 项目保留 PPO 路线检查脚本，用于说明前置条件，避免伪造“已完成 PPO”。

## 评测策略

- 规则评测：保证可复现，适合 CI 和快速回归。
- LLM Judge：按固定 rubrics 输出 JSON，评估准确性、完整性、礼貌性、安全性、幻觉和答非所问。
- 人工抽检：最终简历指标建议至少抽样 100 条复核。
