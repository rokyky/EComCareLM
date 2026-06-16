# 行为规格：EComCareLM MVP

## 数据清洗

- WHEN 原始记录包含手机号、身份证样式号码、订单号、快递单号、银行卡号或地址，THEN 清洗器必须用稳定占位符进行脱敏。
- WHEN 两条记录归一化后的相似度超过配置阈值，THEN 清洗器必须只保留第一条。
- WHEN 某一行不是合法 JSONL，THEN 命令必须明确报错并指出文件和行号。
- WHEN 输入文件不存在，THEN 命令必须非零退出并打印缺失路径。

## SFT 构建

- WHEN FAQ 记录包含 `question` 和 `answer`，THEN `build-sft` 必须生成 instruction/input/output 样本。
- WHEN 样本包含 `scenario`、`difficulty`、`requires_policy` 等元数据，THEN 必须保留到 `metadata`。
- WHEN 样本包含上下文或平台规则，THEN 必须写入模型输入，不能丢弃。

## DPO 构建

- WHEN 存在 SFT 样本，THEN `build-dpo` 必须创建 `prompt`、`chosen` 和 `rejected` 字段。
- WHEN 配置了负样本类型，THEN 每条 DPO 样本必须包含 `metadata.negative_type`。
- WHEN 样本没有可用 chosen answer，THEN 必须跳过该样本并在命令输出中统计数量。

## Baseline 演示

- WHEN 传入 eval set，THEN `demo` 必须为每个评测样本写出一条预测。
- WHEN 样本包含平台规则和必答点，THEN baseline 回答应该尽量覆盖规则相关处理路径。
- WHEN eval set 为空，THEN 命令必须创建空预测文件并报告 0 条样本。

## 自动评测

- WHEN predictions 和 eval cases 共享 `case_id`，THEN `eval` 必须计算逐样本指标。
- WHEN 预测回答包含 `must_not_include` 中的禁用表达，THEN 安全性和幻觉信号必须反映该失败。
- WHEN 预测回答遗漏 `must_include` 中的关键点，THEN 完整性得分必须下降，并且该样本应进入 badcase 候选。
- WHEN 某个评测样本没有预测结果，THEN 评测器必须标记 missing prediction，不能静默忽略。

## 报告生成

- WHEN 存在评测结果，THEN `report` 必须汇总整体指标、分场景指标和 badcase 数量。
- WHEN 没有 badcase，THEN 报告必须明确写出没有发现 badcase。
- WHEN 输入结果文件不存在，THEN 报告命令必须明确失败。
