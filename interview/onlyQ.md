# EComCareLM 面试问题清单（完整版）

> 本文件只保留问题，用于面试前快速自测。每题都可展开为面试回答 + 面试官视角 + 项目结合三段式回答。
> 顺序模拟真实面试流程：**先开门见山讲项目，再问数据来源和训练方法，然后深入 PGDM 创新点和评测体系，最后考察基础功和扩展知识**。
> 共 **13 个分类，153 题**。★高频必问 ▲ PGDM 特色题。

---

## 一、项目总览（开场定位）

> 面试官开场必问："介绍一下你的项目"。这一节回答要简洁有力，说清楚做什么、为什么、和别的项目什么关系。

1.  ★ EComCareLM 一句话怎么介绍？
2.  ★ 为什么选电商客服做对齐？
3.  ★ 和 bitalab 有什么区别？
4.  ★ 为什么不是 RAG 方案？
5.  ★ 被问"是不是调包"怎么回应？
6.  ★ 项目当前最大不足是什么？
7.    项目的核心贡献是工程还是方法？
8.  ▲ PGDM 是什么意思？一句话说清
9.    项目从 MVP 到 PGDM 经历了哪几个阶段？
10.   你觉得这个项目在面试中能打几分？

---

## 二、数据构建（数据来源与清洗）

> 面试官会接着问数据——"数据哪来的？怎么处理的？有什么坑？"数据工程意识是加分项。

11. ★ 公开数据怎么接入？字段映射怎么设计的？
12. ★ JDDC 和 EcomInstruct 各适合什么场景？
13.    JDDC 数据有哪些已知限制？（弱标注、无 policy、无 must_include）
14.    JDDC 多轮对话怎么拆成单轮 QA 对的？
15.    import_jddc_baseline.py 里的场景推断是怎么做的？
16. ★ PII 脱敏为什么重要？项目做了哪 6 类脱敏？
17.    正则脱敏的主要局限是什么？生产环境怎么补？
18. ★ 去重阈值为什么用 0.92？怎么调的？
19.    近重复去重和精确去重有什么区别？大规模场景用什么？
20. ★ SFT 数据的 instruction/input/output 格式各是什么？
21.    DPO 数据的 chosen/rejected 有几种构造方式？
22.    build_dpo_from_predictions 是怎么基于模型预测构造 DPO 对的？
23.    chosen 和 rejected 文本相同时为什么要跳过？
24. ★ SFT 数据太模板化有什么风险？怎么解？
25.    train/dev/test 的切分比例和随机种子怎么设？
26. ★ JDDC 数据为什么没有 policy 和 must_include？对评测有什么影响？

---

## 三、SFT & LoRA 训练（基础训练方法）

> 面试官开始问训练细节——"用什么方法训练的？为什么选 LoRA 不是全参？"考察实践能力。

27. ★ SFT 解决什么问题？它的根本局限是什么？
28. ★ SFT 的 loss 怎么理解？cross-entropy 在做什么？
29. ★ SFT 为什么只对 assistant response 算 loss？prompt 部分怎么 mask？
30.    response marker 匹配为什么要从后往前找？
31. ★ LoRA 原理是什么？为什么不一上来用全参微调？
32. ★ target_modules 为什么选 q/k/v/o 和 gate/up/down？
33. ★ LoRA 的 rank/alpha/dropout 怎么选？直觉是什么？
34. ★ QLoRA 和 LoRA 有什么区别？（4bit NF4、double quantization）
35. ★ merge_and_unload() 为什么重要？不 merge 会导致什么问题？
36.    gradient accumulation 解决什么问题？和 batch size 什么关系？
37.    gradient checkpointing 的原理是什么？计算换显存？
38.    mixed precision（bf16/fp16）有什么风险？怎么选？
39.    为什么训练前要做 smoke test？--max-steps 5 在测什么？

---

## 四、DPO & 偏好对齐（偏好优化基础）

> 面试官深入问对齐方法——"只用 SFT 有什么问题？DPO 怎么工作的？"这是项目主线之一，需要讲清楚。

40. ★ DPO 的原理直觉是什么？和 PPO 的核心区别？
41. ★ DPO loss 怎么理解？policy 和 reference 的 logprob 差在做什么？
42. ★ beta 的作用是什么？beta 太大/太小会怎样？
43. ★ 为什么 rejected 要优先来自模型真实 badcase？
44.    模板负样本（Template-DPO）的根本问题是什么？
45.    DPO 训练后模型变啰嗦/变拒答怎么办？
46. ★ DPO 数据量太小怎么办？最少需要多少条？
47.    DPO 有哪些已知的失败模式？
48.    DPO 训练时 chosen 比 rejected 长很多会有什么问题？
49.    DPO 的 eval 怎么设计？只看 loss 够吗？
50.    build-dpo 和 build-dpo-from-predictions 各在什么时候用？

---

## 五、PGDM 核心方法（项目创新点）

> 面试官已经理解了 DPO 基础，现在问你的创新——"你在 DPO 上做了什么改进？"这是展示方法论深度的最佳时机。

51. ★▲ PGDM 的全称和核心思路是什么？
52. ★▲ 为什么规则评测和 LLM Judge 的分歧是很有价值的训练信号？
53. ▲  维度级分歧挖掘是怎么做的？规则和 Judge 的维度怎么对齐？
54. ▲  分歧挖掘输出 aggregate 和 dimension 两级，各是什么含义？
55. ★▲ PGDM-DPO 的 sample_weight = severity × disagreement_strength × confidence，为什么这样设计？
56. ▲  policy_cap 是什么？为什么需要 per-policy quota cap？
57. ▲  PGDM-DPO 和 Template-DPO 的 rejected 构造本质区别是什么？
58. ▲  PGDM-DPO 和 Badcase-DPO 的 rejected 构造本质区别是什么？
59. ▲  PGDM 为什么需要 policy labeler？规则 labeler 和 LLM labeler 各有什么优劣？
60. ▲  disagreement_strength 的 floor 为什么设计为 1.0？
61. ▲  如果规则评测和 LLM Judge 结果完全一致，PGDM 还有用吗？
62. ★▲ HRPVR 是什么？为什么选它做主指标？
63. ▲  HRPVR 只计哪三类 badcase？为什么排除 incomplete、impolite？
64. ▲  6 组消融实验分别对比什么？结论预期是什么？

---

## 六、GRPO & 规则奖励（规则约束方法）

> 面试官问另一条优化路径——"除了 DPO，还用了什么方法？GRPO 怎么做的？"

65. ★ GRPO 的原理是什么？和 PPO 的核心区别在哪里？
66. ★▲ GRPO 为什么特别适合客服规则的场景？
67. ★▲ 客服规则奖励（rewards.py）的设计思路？几个 score 维度怎么组合？
68. ★  reward hacking 是什么？项目的多维奖励怎么防止 hacking？
69.    规则 reward 返回全 1.0 为什么不是好事？
70.    GRPO 训多了会模板化吗？怎么解？
71.    GRPO 的 num_generations 设多少合理？为什么？
72.    什么情况下应该训练独立的 reward model 而不是写规则 reward？

---

## 七、PPO & RLHF 全流程（理论深度）

> 面试官考察你对 RLHF 完整流程的理解——"DPO/GRPO 和标准 PPO 比有什么优劣？什么时候应该上 PPO？"

73. ★ RLHF 标准三步骤（SFT → RM → PPO）分别做什么？
74. ★ 为什么项目第一阶段不把 PPO 作为主线？
75.    PPO 的 clipping 参数（epsilon）起什么作用？
76.    KL penalty 在 RLHF 里起什么作用？beta 和 KL 什么关系？
77.    训练 reward model 需要什么样的数据？多少条？
78.    SFT → DPO → GRPO → PPO 的合理推进顺序是什么？
79.    train_ppo.py 为什么目前只是一个 route checker？

---

## 八、评测体系（效果验证）

> 面试官关心你怎么证明自己的方法有效——"怎么评测的？规则评测和 LLM Judge 各有什么优劣？"

80. ★▲ 项目的双层评测体系（规则评测 + LLM Judge）各有什么定位？
81. ★ 规则评测的 7 个维度分别是哪些？怎么打分的？
82. ★ 评测指标中为什么 forbidden_content 和 off_topic 的方向和其他维度不一致？
83. ★▲ LLM Judge 的 rubric 是怎么设计的？6 个评分维度分别是什么？
84. ★▲ LLM Judge 的 accuracy 为什么映射到 policy_compliance 而不是 answer_accuracy？
85. ★▲ hallucination 维度为什么只在 Judge 里有、不在规则评测里？
86. ★  LLM Judge 有什么已知偏差？怎么缓解？
87. ★  LLM Judge 的可信度怎么保证？人工一致率是什么？
88. ★▲ DIMENSION_KEYS 是怎么定义和对齐的？
89. ▲  分歧挖掘时单方独有的维度（hallucination、forbidden_content）怎么处理？
90. ▲  mine-disagreements 的 threshold（0.35）和 pass_threshold（0.75）各控制什么？
91. ★  客服场景的幻觉怎么定义？和通用 NLP 的幻觉有什么不同？
92. ★  Preference Win Rate 怎么算？为什么比绝对分数更有意义？
93.    规则评测全 1.0 说明什么？是好事吗？
94.    人工抽检怎么做？样本量怎么定？
95.    指标提升但人工抽检觉得差，听谁的？
96.    评测报告（report.py）包含哪些内容？

---

## 九、Policy Taxonomy & Labeling（支撑体系）

> 面试官问 PGDM 的支撑系统——"你们定义了几类策略？怎么标注违规的？"

97. ★▲ Policy taxonomy 定义了哪 6 类策略？各自的 severity 是多少？
98. ★▲ refund_policy / privacy_policy / escalation_policy 为什么划为高风险？
99. ▲  tone_policy 的 violation_hints 举例？为什么 severity 只有 0.8？
100.★▲ policy_labeler 的 rule-based 标注流程是什么？
101.▲  过度承诺检查为什么在 unknown policy 时也要兜底？
102.▲  policy label 的 confidence 怎么定的？为什么 unknown 是 0.2？
103.▲  LLM-based policy labeler 和 rule-based 比各有什么优劣？
104.▲  evidence 字段有什么用？怎么回传给开发者？
105.▲  infer_policy_from_scenario 是怎么根据场景推断策略的？

---

## 十、模型架构（基础功考察）

> 面试官考察你的基础功——"用的什么模型？Transformer/Attention 了解多少？"这一节是很多算法岗面试的必考内容。

106.★ Qwen2.5 是什么结构？（RoPE + SwiGLU + RMSNorm + GQA）
107.★ Qwen2.5 和 Qwen3 的核心架构差异有哪些？
108.★ Attention 公式？Q/K/V 分别做什么？
109.★ MHA / MQA / GQA 的区别？为什么 GQA 成为主流？
110.★ GQA 为什么能减少 KV Cache？具体省多少？
111.★ KV Cache 的显存怎么估算？公式是什么？
112.★ RoPE 是什么？和绝对位置编码的区别？
113.   RoPE 与长上下文扩展的关系？（NTK-aware / YaRN）
114.★ SwiGLU 是什么？为什么比 ReLU/GELU 好？
115.★ RMSNorm vs LayerNorm？Pre-norm vs Post-norm？
116.   QK-Norm 是什么？Qwen3 为什么加上它？
117.★ Tokenizer 对中文电商客服场景有什么影响？
118.★ Causal LM 的训练目标是什么？和 Encoder-only 的区别？
119.   Transformer block 包含哪些核心组件？
120.   为什么 LoRA 注入 Attention 和 MLP 的线性层，而不注入 embedding 或 lm_head？

---

## 十一、推理与部署（工程落地）

> 面试官问工程落地的能力——"模型训练完了怎么部署？推理效率怎么优化？"

121.★ vLLM 的 PagedAttention 解决什么问题？为什么比原生 Attention 省显存？
122.★ 推理显存大头在哪？多并发场景 KV Cache 怎么管理？
123.   量化（int4/int8）会损失效果吗？怎么评估量化损失？
124.★ adapter 推理怎么部署？分开加载 vs 合并权重的优劣？
125.★ 怎么减少生成延迟？模型层/输入层/解码层/系统层四层优化？
126.★ 解码策略怎么选？为什么客服场景 temperature 要低？
127.   top-k 和 top-p 采样的区别？什么时候用哪个？
128.   长上下文会带来哪些显存、延迟和效果挑战？
129.   如果业务规则每天变，微调还合适吗？怎么和 RAG 配合？
130.   项目如果要上线成一个客服服务，整体架构怎么设计？

---

## 十二、Agent 面试题（扩展知识）

> 如果面 Agent 开发岗或业务涉及 Agent，面试官会问这部分——"对 Agent 框架了解多少？MCP 是什么？"

131.★ 什么是 LLM Agent？核心组件（Brain/Planning/Memory/Tools）？
132.★ ReAct 框架的原理？ReAct vs CoT vs Act-Only 的区别？
133.★ Function Calling 的工作原理？工具定义格式？
134.★▲ MCP 协议是什么？和 Function Calling 比有什么优势？
135.★ MCP 的三类能力（Tools / Resources / Prompts）？
136.★ Agent 记忆系统怎么设计？L1-L4 四层各存什么？
137.★ 多 Agent 协作有哪些模式？（中心化 vs 去中心化 vs 混合）
138.▲ bitalab 的 Quick Session 免登录是怎么设计的？
139.▲ bitalab 的 Runner 隔离是怎么做的？
140.   Tool Calling 的幻觉怎么处理？（参数幻觉、工具选错）
141.   Agent 评测怎么做？（Tool Call 准确率、任务完成率等）
142.★ EComCareLM 和 bitalab 如果结合，会做成什么？
143.   2025-2026 年 Agent 技术趋势有哪些值得关注？（MCP、A2A、Computer Use）

---

## 十三、进阶对齐与 MoE

130. ★ GRPO 应用在 MoE 混合专家架构上会存在什么问题？
131. GSPO 相比 GRPO 做了哪些优化？为什么更适配 MoE 架构？
132. GRPO 中 Token 级别重要性采样的实现逻辑与作用是什么？


---

## 附：高频代码位置速查

| 问题领域 | 代码位置 |
|----------|----------|
| CLI 入口 & 13 个子命令 | `ecomcarelm/cli.py` |
| 数据导入 + 字段映射 + 切分 | `ecomcarelm/datasets.py` |
| SFT / DPO / PGDM-DPO 数据构建 | `ecomcarelm/builders.py` |
| PII 脱敏 + 近重复去重 | `ecomcarelm/cleaning.py` |
| 规则评测（7 维度打分） | `ecomcarelm/evaluation.py` |
| LLM Judge + 维度对齐 | `ecomcarelm/judge.py` |
| Policy Taxonomy（6 类） | `ecomcarelm/policy.py` |
| Policy 违规标注 | `ecomcarelm/policy_labeler.py` |
| 维度级分歧挖掘 | `ecomcarelm/disagreement.py` |
| 评测 Markdown 报告 | `ecomcarelm/report.py` |
| 规则 Baseline 预测 | `ecomcarelm/baseline.py` |
| GRPO 客服规则奖励 | `train/rewards.py` |
| SFT 训练（Transformers Trainer） | `train/train_sft.py` |
| DPO 训练（TRL DPOTrainer） | `train/train_dpo.py` |
| GRPO 训练（TRL GRPOTrainer） | `train/train_grpo.py` |
| PPO 路线检查器 | `train/train_ppo.py` |
| 全链路 Smoke Test | `scripts/smoke_test.py` |
| JDDC 多轮对话导入 | `scripts/import_jddc_baseline.py` |
| 评测集 JSONL 样例 | `data/samples/eval_set_v1.jsonl` |
| Pipeline 单元测试（7 个） | `tests/test_pipeline.py` |
| 项目 Invariants 定义 | `AI_REVIEW_GUIDE.md` |
| 已发现/已修复 Bug 记录 | `BUG_LEDGER.md` |
