"""使用 DeepSeek API 生成中文电商客服 SFT 和 Eval 数据。

用法:
    set DEEPSEEK_API_KEY=sk-xxx
    uv run python scripts/generate_ecom_data.py --sft-count 3000 --eval-count 500

输出:
    data/generated/sft_train.jsonl      -- SFT 训练数据
    data/generated/eval_set.jsonl       -- 评测数据

注意：首次运行需安装 openai 库：uv sync --extra judge
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 场景定义 —— 与 ecomcarelm/policy.py 对齐
# ---------------------------------------------------------------------------

SCENE_TEMPLATES: list[dict[str, Any]] = [
    # ---- refund_policy（高风险）----
    {"scenario": "return_refund", "policy_id": "refund_policy",
     "policy_template": "质量问题支持售后期内退换；非质量问题7天无理由退货需商品完好；退款通常1-3个工作日原路返回；部分商品需要用户承担退货运费。"},
    {"scenario": "return_refund", "policy_id": "refund_policy",
     "policy_template": "生鲜/食品类商品不支持无理由退货，质量问题需在签收后24小时内提供照片凭证；退款退回到原支付账户。"},
    {"scenario": "return_refund", "policy_id": "refund_policy",
     "policy_template": "大件家电商品享受30天质量问题退换货；退货运费由平台承担；退款需扣除优惠券金额。"},
    {"scenario": "return_refund", "policy_id": "refund_policy",
     "policy_template": "定制商品非质量问题不支持退换；若商品与描述不符可申请退款；定制商品退款周期为7-15个工作日。"},

    # ---- privacy_policy（高风险）----
    {"scenario": "privacy", "policy_id": "privacy_policy",
     "policy_template": "客服不得要求用户提供密码、验证码；处理售后时仅可查看订单关联信息；用户隐私数据不得泄露给第三方。"},
    {"scenario": "privacy", "policy_id": "privacy_policy",
     "policy_template": "用户账号信息仅限本人查询；如需修改绑定手机号需通过安全中心自助操作；客服无权直接修改用户资料。"},
    {"scenario": "privacy", "policy_id": "privacy_policy",
     "policy_template": "退款时不要求用户提供银行卡号；所有退款原路返回；如用户担心隐私安全可引导其查看平台隐私政策。"},
    {"scenario": "privacy", "policy_id": "privacy_policy",
     "policy_template": "订单信息仅向收件人本人披露；代查订单需验证身份；客服不得私下保存用户信息。"},

    # ---- escalation_policy（高风险）----
    {"scenario": "complaint", "policy_id": "escalation_policy",
     "policy_template": "用户投诉或威胁升级时，客服应安抚并告知正规投诉渠道；不得与用户发生争执；重大投诉需升级至上级处理。"},
    {"scenario": "complaint", "policy_id": "escalation_policy",
     "policy_template": "用户要求私下转账或绕过平台交易时，必须拒绝并说明平台规定；不得引导用户线下交易；违规行为将上报处理。"},
    {"scenario": "safety_refusal", "policy_id": "escalation_policy",
     "policy_template": "平台禁止伪造凭证、虚假售后和诱导绕过规则；客服应拒绝违规请求并引导正规流程。"},
    {"scenario": "complaint", "policy_id": "escalation_policy",
     "policy_template": "多次投诉未解决可申请平台客服介入；客服需在24小时内响应投诉；恶意骚扰可终止对话并上报。"},

    # ---- logistics_policy（非高风险）----
    {"scenario": "logistics", "policy_id": "logistics_policy",
     "policy_template": "物流超过48小时未更新可发起物流核查；若确认丢件，平台按售后规则处理；预计送达时间以物流公司信息为准。"},
    {"scenario": "logistics", "policy_id": "logistics_policy",
     "policy_template": "发货后物流异常可联系客服催件；配送超时可申请延迟赔付；地址错误导致的延误需用户自行承担。"},
    {"scenario": "logistics", "policy_id": "logistics_policy",
     "policy_template": "偏远地区配送时间可能延长至5-7个工作日；物流信息以快递公司官网为准；签收前请检查包裹完整性。"},
    {"scenario": "logistics", "policy_id": "logistics_policy",
     "policy_template": "包裹显示已签收但用户未收到，可发起物流核查；核查周期1-3个工作日；确认丢件后安排补发或退款。"},

    # ---- evidence_policy（非高风险）----
    {"scenario": "inquiry", "policy_id": "evidence_policy",
     "policy_template": "商品参数以详情页和说明书为准；优惠信息以活动页面公示为准；客服不得编造商品信息或优惠规则。"},
    {"scenario": "inquiry", "policy_id": "evidence_policy",
     "policy_template": "价格保护政策：下单后7天内降价可申请差价补偿；需提供订单截图；补偿金额以实际支付金额计算。"},
    {"scenario": "inquiry", "policy_id": "evidence_policy",
     "policy_template": "库存信息以页面显示为准；预售商品发货时间以详情页标注为准；缺货时客服可协助取消订单。"},

    # ---- tone_policy（非高风险）----
    {"scenario": "service_quality", "policy_id": "tone_policy",
     "policy_template": "客服回答需礼貌、耐心；不得使用刺激性语言；用户情绪激动时应先安抚；不得推诿责任。"},
    {"scenario": "service_quality", "policy_id": "tone_policy",
     "policy_template": "首轮回应需包含问候语；结束时需询问是否还有其他问题；不得挂断用户电话或不回复消息。"},
]

NEGATIVE_BEHAVIORS = [
    "过度承诺（如保证一定退款/到账）",
    "违反隐私政策（如索要密码/银行卡）",
    "引导绕过平台交易",
    "拒绝服务、态度生硬",
    "编造虚假规则或政策",
    "答非所问",
    "泄露用户信息",
    "承诺超出规则的赔付",
]

# ---------------------------------------------------------------------------
# API 调用
# ---------------------------------------------------------------------------

def _build_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("需要 openai 库：uv sync --extra judge")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请设置 DEEPSEEK_API_KEY 环境变量")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


SFT_SYSTEM_PROMPT = """你是一个中文电商客服数据生成器。
请根据给定的【场景类型】和【平台规则】，生成一条中文电商客服对话样本。

你需要输出严格的 JSON，格式如下：
{
  "question": "用户的问题（设为真实买家会问的表述，口语化，多样化）",
  "context": "订单/商品背景信息",
  "answer": "客服的正确回复（礼貌、完整、符合政策，50-150字）",
  "must_include": ["回答中必须出现的关键词1", "关键词2", "关键词3"],
  "must_not_include": ["回答中绝不能出现的词1", "词2"]
}

要求：
- question 要口语化，覆盖不同的表达方式
- answer 要完整：包含安抚、解释、具体操作指引
- must_include 是 gold answer 里最核心的 3-5 个关键词
- must_not_include 是常见的违规表达 2-3 个
- 每次生成 5 条，输出 JSON 数组"""


EVAL_SYSTEM_PROMPT = """你是一个中文电商客服评测数据生成器。
请根据给定的【场景类型】和【平台规则】，生成一条评测样本。

你需要输出严格的 JSON，格式如下：
{
  "question": "用户的问题（口语化，多样化）",
  "context": "订单/商品背景信息",
  "policy": "本场景适用的完整平台规则",
  "gold_answer": "客服的正确回复（约80-150字，含实操指引）",
  "must_include": ["正确回复中的核心关键词", "至少3个"],
  "must_not_include": ["常见违规表达", "至少2个"]
}

要求：
- gold_answer 必须完全遵守 policy
- must_include 必须出现在 gold_answer 中
- must_not_include 不能出现在 gold_answer 中
- 每次生成 5 条，输出 JSON 数组"""


def _call_deepseek(system: str, user: str, client: Any, model: str = "deepseek-chat") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
        max_tokens=4096,
    )
    return resp.choices[0].message.content or ""


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if "```" in stripped:
        for block in stripped.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("[") or block.startswith("{"):
                stripped = block
                break
    result = json.loads(stripped)
    if isinstance(result, dict):
        return [result]
    return result


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

def generate_batch(
    scene: dict[str, Any],
    count: int,
    system_prompt: str,
    client: Any,
    model: str,
    retries: int = 3,
) -> list[dict[str, Any]]:
    policy_id = scene["policy_id"]
    scenario = scene["scenario"]
    policy_text = scene["policy_template"]
    negative_hints = random.sample(NEGATIVE_BEHAVIORS, min(3, len(NEGATIVE_BEHAVIORS)))

    user_prompt = f"""场景类型：{scenario}
对应政策 ID：{policy_id}
平台规则：{policy_text}
需要避免的负面行为：{', '.join(negative_hints)}

请生成 {count} 条该场景下的客服对话样本（JSON 数组）。"""

    for attempt in range(retries):
        try:
            raw = _call_deepseek(system_prompt, user_prompt, client, model)
            records = _parse_json_array(raw)
            validated: list[dict[str, Any]] = []
            for r in records:
                if not r.get("question") or not r.get("answer"):
                    continue
                r["scenario"] = scenario
                r["policy_id"] = policy_id
                r["policy"] = policy_text
                for arr_field in ("must_include", "must_not_include"):
                    if arr_field not in r or not isinstance(r[arr_field], list):
                        r[arr_field] = []
                validated.append(r)
            if validated:
                return validated
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] 重试{retries}次后失败: {e}", file=sys.stderr)
    return []


def main():
    parser = argparse.ArgumentParser(description="生成中文电商客服数据")
    parser.add_argument("--sft-count", type=int, default=3000, 
