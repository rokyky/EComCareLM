# 变更提案：初始化 EComCareLM

## 目标

- 创建一个可复现的中文电商客服垂域模型对齐 MVP。
- 覆盖面试中最容易被追问的闭环：数据清洗、SFT 样本构建、DPO 偏好样本构建、自动评测、badcase 挖掘和报告生成。
- 保证项目在普通笔记本上也能跑通，不强依赖 GPU 或外部大模型下载。
- 为后续接入 LLaMA-Factory 或 ms-swift 训练留下清晰扩展点。

## 非目标

- 不内置任何私有电商数据，也不虚构生产指标。
- 不从零实现完整 LoRA/QLoRA 训练循环。
- 不做 RAG 客服机器人；该项目定位是模型后训练和评测。
- MVP 不依赖付费 API。

## 影响范围

- 新增独立项目目录：`EComCareLM`。
- 新增 Python 包：`ecomcarelm`。
- 新增用于数据处理和评测的 CLI 命令。
- 新增 `data/` 下的样例数据和 schema。
- 新增 `train/` 下的训练配置占位文件。
- 新增 `openspec/changes/bootstrap-ecomcarelm/` 下的 OpenSpec 变更文件。
