from __future__ import annotations

from typing import Any


礼貌词 = ("您好", "理解", "建议", "请", "感谢")
风险表达 = ("一定今天", "保证到账", "无需审核", "全额赔偿", "帮你编", "可以绕过平台", "建议私下转账")


def _转成文本(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _转成列表(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        if not value:
            return []
        return [item.strip() for item in value.split("|") if item.strip()]
    return [str(value)]


def 客服规则奖励(completions: list[Any], must_include: Any = None, must_not_include: Any = None, **_: Any) -> list[float]:
    """给 GRPO 使用的电商客服规则 reward。

    分数范围大致在 0 到 1 之间：
    - 覆盖必答点会加分。
    - 礼貌安抚会加分。
    - 过度承诺、违规协助、命中禁用表达会扣分。
    """

    include_lists = must_include if isinstance(must_include, list) else [must_include] * len(completions)
    forbid_lists = must_not_include if isinstance(must_not_include, list) else [must_not_include] * len(completions)
    rewards: list[float] = []

    for completion, include_value, forbid_value in zip(completions, include_lists, forbid_lists):
        text = _转成文本(completion)
        required = _转成列表(include_value)
        forbidden = _转成列表(forbid_value)

        score = 0.2
        if required:
            hit = sum(1 for phrase in required if phrase in text)
            score += 0.5 * (hit / len(required))
        else:
            score += 0.3

        if any(word in text for word in 礼貌词):
            score += 0.15
        if len(text) >= 20:
            score += 0.15

        if any(phrase in text for phrase in forbidden):
            score -= 0.5
        if any(phrase in text for phrase in 风险表达):
            score -= 0.4

        rewards.append(max(0.0, min(1.0, score)))
    return rewards
