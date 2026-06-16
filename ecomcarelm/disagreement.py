from __future__ import annotations

from typing import Any

from .judge import normalize_judge_dimensions


DIMENSION_KEYS = (
    "answer_accuracy",
    "policy_compliance",
    "completeness",
    "politeness",
    "safety",
    "hallucination",
    "off_topic",
)


def normalize_rule_dimensions(record: dict[str, Any]) -> dict[str, float]:
    existing = record.get("dimensions")
    if isinstance(existing, dict):
        return {
            key: round(float(value), 4)
            for key, value in existing.items()
            if key in DIMENSION_KEYS and _is_number(value)
        }

    dimensions: dict[str, float] = {}
    for key in ("answer_accuracy", "policy_compliance", "completeness", "politeness", "safety"):
        if _is_number(record.get(key)):
            dimensions[key] = round(float(record[key]), 4)
    if _is_number(record.get("hallucination")):
        dimensions["hallucination"] = round(1.0 - float(record["hallucination"]), 4)
    if _is_number(record.get("off_topic")):
        dimensions["off_topic"] = round(1.0 - float(record["off_topic"]), 4)
    return dimensions


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _overall_pass(dimensions: dict[str, float], pass_threshold: float) -> bool:
    if not dimensions:
        return False
    return sum(dimensions.values()) / len(dimensions) >= pass_threshold


def _direction(rule_score: float, judge_score: float, pass_threshold: float) -> str:
    rule_pass = rule_score >= pass_threshold
    judge_pass = judge_score >= pass_threshold
    if rule_pass and not judge_pass:
        return "rule_pass_judge_fail"
    if not rule_pass and judge_pass:
        return "rule_fail_judge_pass"
    if not rule_pass and not judge_pass:
        return "both_fail"
    return "both_pass"


def _aggregate_disagreement(
    rule_dimensions: dict[str, float],
    judge_dimensions: dict[str, float],
    pass_threshold: float,
) -> str:
    rule_pass = _overall_pass(rule_dimensions, pass_threshold)
    judge_pass = _overall_pass(judge_dimensions, pass_threshold)
    if rule_pass and not judge_pass:
        return "rule_pass_judge_fail"
    if not rule_pass and judge_pass:
        return "rule_fail_judge_pass"
    if not rule_pass and not judge_pass:
        return "both_fail"
    return "both_pass"


def compare_dimensions(
    rule_record: dict[str, Any],
    judge_record: dict[str, Any],
    threshold: float = 0.35,
    pass_threshold: float = 0.75,
) -> dict[str, Any]:
    rule_dimensions = normalize_rule_dimensions(rule_record)
    raw_judge_dimensions = judge_record.get("dimensions")
    if isinstance(raw_judge_dimensions, dict):
        judge_dimensions = {
            key: round(float(value), 4)
            for key, value in raw_judge_dimensions.items()
            if key in DIMENSION_KEYS and _is_number(value)
        }
    else:
        judge_dimensions = normalize_judge_dimensions(judge_record)

    dimension_disagreements: list[dict[str, Any]] = []
    for key in DIMENSION_KEYS:
        if key not in rule_dimensions or key not in judge_dimensions:
            continue
        rule_score = rule_dimensions[key]
        judge_score = judge_dimensions[key]
        delta = abs(rule_score - judge_score)
        if delta >= threshold:
            dimension_disagreements.append(
                {
                    "dimension": key,
                    "rule_score": rule_score,
                    "judge_score": judge_score,
                    "delta": round(delta, 4),
                    "direction": _direction(rule_score, judge_score, pass_threshold),
                }
            )

    return {
        "case_id": str(rule_record.get("case_id") or judge_record.get("case_id") or ""),
        "scenario": rule_record.get("scenario", judge_record.get("scenario", "unknown")),
        "aggregate_disagreement": _aggregate_disagreement(
            rule_dimensions,
            judge_dimensions,
            pass_threshold,
        ),
        "dimension_disagreements": dimension_disagreements,
        "rule_dimensions": rule_dimensions,
        "judge_dimensions": judge_dimensions,
    }


def mine_disagreements(
    rule_results: list[dict[str, Any]],
    judge_results: list[dict[str, Any]],
    threshold: float = 0.35,
    pass_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    judge_by_id = {str(item.get("case_id")): item for item in judge_results}
    records: list[dict[str, Any]] = []
    for rule_record in rule_results:
        case_id = str(rule_record.get("case_id"))
        judge_record = judge_by_id.get(case_id)
        if not judge_record:
            records.append(
                {
                    "case_id": case_id,
                    "scenario": rule_record.get("scenario", "unknown"),
                    "aggregate_disagreement": "missing_judge",
                    "dimension_disagreements": [],
                    "rule_dimensions": normalize_rule_dimensions(rule_record),
                    "judge_dimensions": {},
                }
            )
            continue
        records.append(compare_dimensions(rule_record, judge_record, threshold, pass_threshold))
    return records

