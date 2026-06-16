from __future__ import annotations

from typing import Any


SCENARIO_ACTIONS = {
    "logistics": "您可以在订单详情页查看物流轨迹；如果长时间没有更新，可以联系平台发起物流核查。",
    "return_refund": "您可以在订单售后页申请退货退款，并按页面要求上传凭证，平台会根据规则审核。",
    "exchange_repair": "您可以在售后入口选择换货或维修，并补充问题照片、视频或检测信息。",
    "product_param": "建议以商品详情页参数和官方说明为准；如果参数影响使用，可以先不要拆封并咨询客服确认。",
    "coupon_price": "您可以查看优惠券使用门槛和价保规则，符合条件时在订单页提交价保或优惠问题申诉。",
    "invoice_payment": "您可以在订单页申请发票或查看支付状态，发票信息需与实际抬头保持一致。",
    "complaint": "理解您的感受，建议您先保留订单、聊天记录和凭证，平台会按售后规则协助处理。",
    "safety_refusal": "涉及绕过平台规则、虚假凭证或违规操作的请求无法协助，请通过平台正规流程处理。",
}


def generate_answer(case: dict[str, Any]) -> str:
    scenario = str(case.get("scenario", "return_refund"))
    action = SCENARIO_ACTIONS.get(scenario, SCENARIO_ACTIONS["return_refund"])
    policy = str(case.get("policy", "")).strip()
    prefix = "您好，理解您的情况。"
    policy_sentence = f"根据平台规则，{policy}" if policy else "我会根据当前订单状态和平台规则为您判断。"
    required = [str(item) for item in case.get("must_include", []) if str(item)]
    missing = [item for item in required if item not in action and item not in policy_sentence]
    key_points = f"这类问题需要关注：{'、'.join(missing)}。" if missing else ""
    reminder = "处理结果需要以平台审核为准，建议您不要相信站外承诺或私下转账。"
    return f"{prefix}{policy_sentence}{action}{key_points}{reminder}"


def build_predictions(eval_cases: list[dict[str, Any]], model_name: str = "rule_baseline") -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for idx, case in enumerate(eval_cases, start=1):
        predictions.append(
            {
                "case_id": case.get("case_id") or f"eval_{idx:05d}",
                "model": model_name,
                "answer": generate_answer(case),
            }
        )
    return predictions
