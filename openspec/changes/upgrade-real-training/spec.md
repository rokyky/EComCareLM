# 行为规格：真实训练升级

## 公开数据导入

- WHEN 用户提供 Hugging Face dataset 名称和字段映射，THEN `import-hf` 必须输出统一 JSONL。
- WHEN 字段不存在，THEN 命令必须列出可用字段并失败。
- WHEN 用户设置 `--limit`，THEN 导入样本数不得超过该限制。
- WHEN 用户设置 `--split-ratio`，THEN 命令必须生成 train/dev/test 三份文件。

## SFT 训练

- WHEN 用户提供 SFT JSONL 和模型名，THEN `train/train_sft.py` 必须能够启动 Transformers 训练。
- WHEN 启用 LoRA，THEN 脚本必须保存 adapter。
- WHEN 启用 `--load-in-4bit`，THEN 脚本必须使用量化加载，形成 QLoRA 训练路径。
- WHEN 训练依赖缺失，THEN 脚本必须明确提示需要安装 `.[train]`。
- WHEN 只做 smoke test，THEN 脚本必须支持 `--max-steps` 限制训练步数。

## DPO 训练

- WHEN 用户提供 DPO JSONL，THEN `train/train_dpo.py` 必须能基于 prompt/chosen/rejected 启动 DPO。
- WHEN rejected 来自模型预测，THEN metadata 中必须记录来源。
- WHEN 用户启用 LoRA/QLoRA，THEN DPO 训练必须能接收对应参数。
- WHEN TRL 依赖缺失，THEN 脚本必须明确报错并给出安装方式。

## GRPO 训练

- WHEN 用户提供带规则字段的数据，THEN `train/train_grpo.py` 必须能够基于规则 reward 启动 GRPO。
- WHEN completion 覆盖 must_include，THEN reward 应该上升。
- WHEN completion 命中 must_not_include 或风险表达，THEN reward 应该下降。

## PPO 路线

- WHEN 用户尝试 PPO，THEN 脚本必须明确检查 SFT 模型、reward model 和训练数据路径。
- WHEN 没有 reward model，THEN 文档必须说明不建议把 PPO 作为第一阶段主线。

## LLM Judge

- WHEN 用户提供 eval set 和 predictions，THEN `judge` 必须输出逐样本 JSONL 评分。
- WHEN judge 输出不是合法 JSON，THEN 必须记录 parse_error，不能静默丢弃。
- WHEN 未提供 API key，THEN 命令必须明确提示需要配置。
