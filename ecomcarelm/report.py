from __future__ import annotations

from collections import Counter
from typing import Any


METRIC_LABELS = {
    "answer_accuracy": "回答准确率",
    "policy_compliance": "规则遵循率",
    "completeness": "完整性",
    "politeness": "礼貌性",
    "safety": "安全性",
    "hallucination": "幻觉率",
    "off_topic": "答非所问率",
}


def _metric_table(rows: dict[str, dict[str, float]]) -> str:
    headers = ["分组", *METRIC_LABELS.values()]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for group, metrics in rows.items():
        values = [group, *[f"{metrics.get(key, 0.0):.4f}" for key in METRIC_LABELS]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown_report(scored_cases: list[dict[str, Any]], title: str = "EComCareLM 评测报告") -> str:
    count = len(scored_cases)
    if count == 0:
        return f"# {title}\n\n没有发现评测样本。\n"

    overall = {
        key: sum(float(case[key]) for case in scored_cases) / count
        for key in METRIC_LABELS
    }
    scenarios = sorted({str(case.get("scenario", "unknown")) for case in scored_cases})
    scenario_rows = {}
    for scenario in scenarios:
        cases = [case for case in scored_cases if str(case.get("scenario", "unknown")) == scenario]
        scenario_rows[scenario] = {
            key: sum(float(case[key]) for case in cases) / len(cases)
            for key in METRIC_LABELS
        }

    badcase_counts = Counter(str(case["badcase_type"]) for case in scored_cases if case.get("badcase_type"))
    lines = [
        f"# {title}",
        "",
        f"- 评测样本数：{count}",
        f"- badcase 数量：{sum(badcase_counts.values())}",
        "",
        "## 整体指标",
        "",
        _metric_table({"overall": overall}),
        "",
        "## 分场景指标",
        "",
        _metric_table(scenario_rows),
        "",
        "## Badcase 统计",
        "",
    ]
    if badcase_counts:
        lines.extend(f"- {kind}: {num}" for kind, num in sorted(badcase_counts.items()))
    else:
        lines.append("没有发现 badcase。")

    examples = [case for case in scored_cases if case.get("badcase_type")][:5]
    lines.extend(["", "## Badcase 示例", ""])
    if examples:
        for case in examples:
            lines.append(f"- `{case.get('case_id')}` {case.get('scenario')}: {case.get('badcase_type')}")
    else:
        lines.append("没有 badcase 示例。")
    lines.append("")
    return "\n".join(lines)
