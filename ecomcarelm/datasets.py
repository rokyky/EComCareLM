from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SplitResult:
    train: list[dict[str, Any]]
    dev: list[dict[str, Any]]
    test: list[dict[str, Any]]


def _get_field(record: dict[str, Any], field: str | None, default: str = "") -> str:
    if not field:
        return default
    value = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(field)
        value = value[part]
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def available_fields(record: dict[str, Any], prefix: str = "") -> list[str]:
    fields: list[str] = []
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        fields.append(name)
        if isinstance(value, dict):
            fields.extend(available_fields(value, name))
    return fields


def convert_records(
    records: list[dict[str, Any]],
    question_field: str,
    answer_field: str,
    context_field: str | None = None,
    policy_field: str | None = None,
    scenario: str = "public_dataset",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        try:
            question = _get_field(record, question_field)
            answer = _get_field(record, answer_field)
            context = _get_field(record, context_field) if context_field else ""
            policy = _get_field(record, policy_field) if policy_field else ""
        except KeyError as exc:
            fields = available_fields(record)
            raise ValueError(f"字段不存在：{exc.args[0]}；可用字段：{', '.join(fields)}") from exc

        if not question.strip() or not answer.strip():
            continue
        converted.append(
            {
                "case_id": f"public_{idx + 1:08d}",
                "scenario": scenario,
                "question": question,
                "context": context,
                "policy": policy,
                "answer": answer,
                "must_include": [],
                "must_not_include": [],
            }
        )
        if limit is not None and len(converted) >= limit:
            break
    return converted


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.9,
    dev_ratio: float = 0.05,
    seed: int = 42,
) -> SplitResult:
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("切分比例不合法：需要 0 < train_ratio，0 <= dev_ratio，且 train_ratio + dev_ratio < 1")

    copied = list(records)
    random.Random(seed).shuffle(copied)
    train_end = int(len(copied) * train_ratio)
    dev_end = train_end + int(len(copied) * dev_ratio)
    return SplitResult(
        train=copied[:train_end],
        dev=copied[train_end:dev_end],
        test=copied[dev_end:],
    )


def import_hf_dataset(
    dataset_name: str,
    split: str,
    question_field: str,
    answer_field: str,
    context_field: str | None = None,
    policy_field: str | None = None,
    scenario: str = "public_dataset",
    limit: int | None = None,
    dataset_config: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("缺少 datasets 依赖，请先运行：uv sync --extra data") from exc

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    records = [dict(item) for item in dataset]
    return convert_records(
        records,
        question_field=question_field,
        answer_field=answer_field,
        context_field=context_field,
        policy_field=policy_field,
        scenario=scenario,
        limit=limit,
    )


def build_dpo_from_predictions(
    eval_cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    prediction_by_id = {str(item.get("case_id")): item for item in predictions}
    pairs: list[dict[str, Any]] = []
    for case in eval_cases:
        case_id = str(case.get("case_id"))
        prediction = prediction_by_id.get(case_id)
        chosen = case.get("gold_answer") or case.get("answer")
        rejected = prediction.get("answer") if prediction else None
        if not chosen or not rejected or str(chosen).strip() == str(rejected).strip():
            continue
        prompt_parts = [
            "你是中文电商平台客服，请根据上下文和平台规则回答用户。",
            f"用户问题：{case.get('question', '')}",
        ]
        if case.get("context"):
            prompt_parts.append(f"上下文：{case['context']}")
        if case.get("policy"):
            prompt_parts.append(f"平台规则：{case['policy']}")
        pairs.append(
            {
                "id": f"dpo_from_pred_{len(pairs) + 1:08d}",
                "prompt": "\n".join(prompt_parts),
                "chosen": str(chosen),
                "rejected": str(rejected),
                "metadata": {
                    "source": "model_prediction_badcase",
                    "case_id": case_id,
                    "scenario": case.get("scenario", "unknown"),
                },
            }
        )
        if max_items is not None and len(pairs) >= max_items:
            break
    return pairs
