from __future__ import annotations

import math
from typing import Any

from .policy import UNKNOWN_POLICY_ID, policy_severity


DEFAULT_INSTRUCTION = "你是中文电商平台客服，请根据订单状态、商品信息和平台规则，给出准确、完整、礼貌且不过度承诺的回答。"


NEGATIVE_TEMPLATES = {
    "policy_conflict": "您好，这种情况平台无法处理，建议您自行和商家协商。",
    "off_topic": "您好，优惠券通常会在活动页展示，您可以关注后续促销。",
    "over_promise": "您好，这个问题我们一定今天给您全额赔偿并立即到账。",
    "incomplete": "您好，可以处理。",
    "impolite": "你自己去订单页看一下，按页面提示操作就行。",
}


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {}
    for key in ("scenario", "difficulty", "requires_policy", "requires_order_status", "safety_sensitive"):
        if key in record:
            metadata[key] = record[key]
    return metadata


def _compose_input(record: dict[str, Any]) -> str:
    chunks: list[str] = []
    if record.get("question"):
        chunks.append(f"用户问题：{record['question']}")
    if record.get("context"):
        chunks.append(f"上下文：{record['context']}")
    if record.get("policy"):
        chunks.append(f"平台规则：{record['policy']}")
    if record.get("product"):
        chunks.append(f"商品信息：{record['product']}")
    return "\n".join(chunks)


def build_sft_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        question = record.get("question")
        answer = record.get("answer") or record.get("gold_answer")
        if not question or not answer:
            continue
        sample = {
            "id": record.get("case_id") or f"sft_{idx:05d}",
            "instruction": record.get("instruction") or DEFAULT_INSTRUCTION,
            "input": _compose_input(record),
            "output": answer,
            "metadata": _metadata(record),
        }
        samples.append(sample)
    return samples


def prompt_from_sft(sample: dict[str, Any]) -> str:
    instruction = sample.get("instruction", DEFAULT_INSTRUCTION)
    sample_input = sample.get("input", "")
    return f"{instruction}\n\n{sample_input}".strip()


def build_dpo_records(
    sft_samples: list[dict[str, Any]],
    negative_type: str = "policy_conflict",
) -> tuple[list[dict[str, Any]], int]:
    rejected_template = NEGATIVE_TEMPLATES.get(negative_type, NEGATIVE_TEMPLATES["policy_conflict"])
    dpo_samples: list[dict[str, Any]] = []
    skipped = 0

    for idx, sample in enumerate(sft_samples, start=1):
        chosen = sample.get("output")
        if not chosen:
            skipped += 1
            continue
        metadata = dict(sample.get("metadata", {}))
        metadata["negative_type"] = negative_type
        dpo_samples.append(
            {
                "id": sample.get("id") or f"dpo_{idx:05d}",
                "prompt": prompt_from_sft(sample),
                "chosen": chosen,
                "rejected": rejected_template,
                "metadata": metadata,
            }
        )
    return dpo_samples, skipped


def _case_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("case_id")): item for item in records}


def _best_dimension(disagreement: dict[str, Any] | None) -> dict[str, Any] | None:
    if not disagreement:
        return None
    dimensions = disagreement.get("dimension_disagreements", [])
    if not isinstance(dimensions, list) or not dimensions:
        return None
    return max(
        (item for item in dimensions if isinstance(item, dict)),
        key=lambda item: float(item.get("delta", 0.0)),
        default=None,
    )


def _disagreement_strength(best_dimension: dict[str, Any] | None) -> float:
    if not best_dimension:
        return 1.0
    try:
        delta = float(best_dimension.get("delta", 0.0))
    except (TypeError, ValueError):
        delta = 0.0
    return round(1.0 + max(0.0, min(1.0, delta)), 4)


def _eval_prompt(case: dict[str, Any]) -> str:
    return f"{DEFAULT_INSTRUCTION}\n\n{_compose_input(case)}".strip()


def build_pgdm_dpo_records(
    eval_cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    policy_labels: list[dict[str, Any]],
    disagreements: list[dict[str, Any]],
    max_items: int | None = None,
    policy_cap: float = 0.5,
    min_label_confidence: float = 0.5,
    include_unknown: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prediction_by_id = _case_map(predictions)
    label_by_id = _case_map(policy_labels)
    disagreement_by_id = _case_map(disagreements)
    skipped = {
        "missing_prediction": 0,
        "missing_chosen": 0,
        "identical_pair": 0,
        "low_confidence": 0,
        "unknown_policy": 0,
        "quota_cap": 0,
    }
    candidates: list[dict[str, Any]] = []

    for case in eval_cases:
        case_id = str(case.get("case_id"))
        prediction = prediction_by_id.get(case_id)
        if not prediction:
            skipped["missing_prediction"] += 1
            continue

        chosen = case.get("gold_answer") or case.get("answer")
        rejected = prediction.get("answer")
        if not chosen:
            skipped["missing_chosen"] += 1
            continue
        if not rejected or str(chosen).strip() == str(rejected).strip():
            skipped["identical_pair"] += 1
            continue

        label = label_by_id.get(case_id, {})
        policy_id = str(label.get("policy_id", UNKNOWN_POLICY_ID))
        try:
            confidence = float(label.get("confidence", 0.2))
        except (TypeError, ValueError):
            confidence = 0.2
        if policy_id == UNKNOWN_POLICY_ID and not include_unknown:
            skipped["unknown_policy"] += 1
            continue
        if confidence < min_label_confidence:
            skipped["low_confidence"] += 1
            continue

        disagreement = disagreement_by_id.get(case_id, {})
        best_dimension = _best_dimension(disagreement)
        eval_dimension = None if best_dimension is None else best_dimension.get("dimension")
        disagreement_strength = _disagreement_strength(best_dimension)
        severity = policy_severity(policy_id)
        sample_weight = round(severity * disagreement_strength * confidence, 4)
        candidates.append(
            {
                "prompt": _eval_prompt(case),
                "chosen": str(chosen),
                "rejected": str(rejected),
                "sample_weight": sample_weight,
                "metadata": {
                    "source": "pgdm_dpo",
                    "case_id": case_id,
                    "scenario": case.get("scenario", "unknown"),
                    "policy_id": policy_id,
                    "violation_type": label.get("violation_type", "unknown"),
                    "eval_dimension": eval_dimension,
                    "aggregate_disagreement": disagreement.get("aggregate_disagreement", "missing_disagreement"),
                    "dimension_disagreement": None if best_dimension is None else best_dimension.get("direction"),
                    "label_confidence": confidence,
                    "disagreement_strength": disagreement_strength,
                    "policy_severity": severity,
                    "sample_weight": sample_weight,
                },
            }
        )

    candidates.sort(key=lambda item: float(item["sample_weight"]), reverse=True)
    target = max_items if max_items is not None else len(candidates)
    target = max(0, target)
    if target == 0:
        return [], skipped

    cap_count = math.ceil(target * policy_cap) if policy_cap > 0 else target
    cap_count = max(1, cap_count)
    selected: list[dict[str, Any]] = []
    counts_by_policy: dict[str, int] = {}

    for candidate in candidates:
        policy_id = str(candidate["metadata"].get("policy_id", UNKNOWN_POLICY_ID))
        current = counts_by_policy.get(policy_id, 0)
        if current >= cap_count:
            skipped["quota_cap"] += 1
            continue
        selected.append(candidate)
        counts_by_policy[policy_id] = current + 1
        if len(selected) >= target:
            break

    for idx, sample in enumerate(selected, start=1):
        sample["id"] = f"pgdm_dpo_{idx:08d}"
        sample["metadata"]["rank"] = idx
    return selected, skipped
