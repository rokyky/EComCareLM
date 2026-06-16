# 设计说明：EComCareLM MVP

## 架构

EComCareLM 由一个轻量 Python 包和一组 CLI 命令组成。核心模块保持简单，方便面试解释和后续扩展：

- `cleaning`：文本归一化、敏感信息脱敏、近重复过滤。
- `builders`：把 FAQ、对话、规则记录转换成 SFT 样本，并构造 DPO chosen/rejected 偏好样本。
- `baseline`：本地可复现的规则客服回答生成器，用来验证评测链路。
- `evaluation`：基于规则的评测指标，包括规则遵循、完整性、礼貌性、安全性、幻觉和答非所问风险。
- `report`：生成适合项目复盘和面试展示的 markdown 报告。

## 数据流

1. 原始记录以 JSONL 形式放在 `data/raw`。
2. `clean` 把脱敏、去重后的记录写入 `data/processed`。
3. `build-sft` 生成 instruction/input/output 格式的 SFT 样本。
4. `build-dpo` 基于标准答案和可控负样本生成偏好对。
5. `demo` 为评测集生成规则 baseline 回答。
6. `eval` 对预测结果逐样本打分，并产出 badcase 类型。
7. `report` 汇总整体指标、分场景指标和 badcase 统计。

## 数据结构

SFT 样本：

```json
{
  "instruction": "你是中文电商平台客服，请根据订单状态和平台规则回答用户。",
  "input": "用户问题、订单状态、平台规则等上下文",
  "output": "标准客服回复",
  "metadata": {"scenario": "return_refund", "difficulty": "medium"}
}
```

DPO 样本：

```json
{
  "prompt": "用户问题和平台规则",
  "chosen": "更准确、完整、礼貌、合规的回答",
  "rejected": "存在规则错误、过度承诺或答非所问的回答",
  "metadata": {"negative_type": "policy_conflict"}
}
```

评测样本：

```json
{
  "case_id": "eval_001",
  "question": "用户问题",
  "context": "订单或商品上下文",
  "policy": "平台规则",
  "gold_answer": "参考答案",
  "scenario": "return_refund",
  "must_include": ["申请售后", "上传凭证"],
  "must_not_include": ["一定赔偿", "无需审核"]
}
```

## 风险与权衡

- 规则评测透明、可复现，但不能完全替代人工评审；最终报告建议抽样 100 条做人工复核。
- 合成 rejected answer 适合作为 MVP 的 DPO 数据来源，但真实项目后期应混入模型实际 badcase。
- 规则 baseline 只能证明数据和评测链路跑通，不能代表最终模型质量；真实微调需要接入训练框架。
