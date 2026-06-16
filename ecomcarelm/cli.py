from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .baseline import build_predictions
from .builders import build_dpo_records, build_pgdm_dpo_records, build_sft_records
from .cleaning import clean_records
from .datasets import build_dpo_from_predictions, import_hf_dataset, split_records
from .disagreement import mine_disagreements
from .evaluation import evaluate_cases
from .io import read_jsonl, write_jsonl, write_text
from .judge import run_llm_judge
from .policy import high_risk_policy_violation_rate
from .policy_labeler import label_policy_violations
from .report import build_markdown_report


def cmd_clean(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    cleaned, stats = clean_records(records, dedup_threshold=args.dedup_threshold)
    write_jsonl(args.output, cleaned)
    print(f"cleaned input={stats.input_count} output={stats.output_count} duplicates={stats.duplicate_count}")
    return 0


def cmd_build_sft(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    samples = build_sft_records(records)
    write_jsonl(args.output, samples)
    print(f"sft_samples={len(samples)}")
    return 0


def cmd_build_dpo(args: argparse.Namespace) -> int:
    sft_samples = read_jsonl(args.input)
    samples, skipped = build_dpo_records(sft_samples, negative_type=args.negative_type)
    write_jsonl(args.output, samples)
    print(f"dpo_samples={len(samples)} skipped={skipped} negative_type={args.negative_type}")
    return 0


def cmd_build_dpo_from_predictions(args: argparse.Namespace) -> int:
    eval_cases = read_jsonl(args.eval_set)
    predictions = read_jsonl(args.predictions)
    samples = build_dpo_from_predictions(eval_cases, predictions, max_items=args.max_items)
    write_jsonl(args.output, samples)
    print(f"dpo_samples={len(samples)} source=model_prediction_badcase")
    return 0


def cmd_import_hf(args: argparse.Namespace) -> int:
    records = import_hf_dataset(
        dataset_name=args.dataset,
        dataset_config=args.config,
        split=args.split,
        question_field=args.question_field,
        answer_field=args.answer_field,
        context_field=args.context_field,
        policy_field=args.policy_field,
        scenario=args.scenario,
        limit=args.limit,
    )
    if args.train_output and args.dev_output and args.test_output:
        splits = split_records(records, train_ratio=args.train_ratio, dev_ratio=args.dev_ratio, seed=args.seed)
        write_jsonl(args.train_output, splits.train)
        write_jsonl(args.dev_output, splits.dev)
        write_jsonl(args.test_output, splits.test)
        print(f"train={len(splits.train)} dev={len(splits.dev)} test={len(splits.test)}")
    else:
        write_jsonl(args.output, records)
        print(f"imported={len(records)}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    eval_cases = read_jsonl(args.eval_set)
    predictions = build_predictions(eval_cases, model_name=args.model_name)
    write_jsonl(args.output, predictions)
    print(f"predictions={len(predictions)}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    eval_cases = read_jsonl(args.eval_set)
    predictions = read_jsonl(args.predictions)
    scored, summary = evaluate_cases(eval_cases, predictions)
    write_jsonl(args.output, scored)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"evaluated={summary.case_count} badcases={sum(summary.badcase_counts.values())}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    eval_cases = read_jsonl(args.eval_set)
    predictions = read_jsonl(args.predictions)
    results = run_llm_judge(eval_cases, predictions, model=args.model, temperature=args.temperature)
    write_jsonl(args.output, results)
    parse_errors = sum(1 for item in results if item.get("parse_error"))
    print(f"judged={len(results)} parse_errors={parse_errors}")
    return 0


def cmd_label_policy(args: argparse.Namespace) -> int:
    eval_cases = read_jsonl(args.eval_set)
    predictions = read_jsonl(args.predictions)
    labels = label_policy_violations(eval_cases, predictions)
    write_jsonl(args.output, labels)
    unknown = sum(1 for item in labels if item.get("policy_id") == "unknown_policy")
    print(f"policy_labels={len(labels)} unknown={unknown}")
    return 0


def cmd_mine_disagreements(args: argparse.Namespace) -> int:
    rule_results = read_jsonl(args.rule_results)
    judge_results = read_jsonl(args.judge_results)
    records = mine_disagreements(
        rule_results,
        judge_results,
        threshold=args.threshold,
        pass_threshold=args.pass_threshold,
    )
    write_jsonl(args.output, records)
    dimension_count = sum(len(item.get("dimension_disagreements", [])) for item in records)
    print(f"disagreements={len(records)} dimension_disagreements={dimension_count}")
    return 0


def cmd_build_pgdm_dpo(args: argparse.Namespace) -> int:
    eval_cases = read_jsonl(args.eval_set)
    predictions = read_jsonl(args.predictions)
    policy_labels = read_jsonl(args.policy_labels)
    disagreements = read_jsonl(args.disagreements)
    samples, skipped = build_pgdm_dpo_records(
        eval_cases,
        predictions,
        policy_labels,
        disagreements,
        max_items=args.max_items,
        policy_cap=args.policy_cap,
        min_label_confidence=args.min_label_confidence,
        include_unknown=args.include_unknown,
    )
    write_jsonl(args.output, samples)
    skipped_text = " ".join(f"{key}={value}" for key, value in sorted(skipped.items()))
    print(f"pgdm_dpo_samples={len(samples)} {skipped_text}")
    return 0


def cmd_hrpvr(args: argparse.Namespace) -> int:
    rule_results = read_jsonl(args.rule_results)
    policy_labels = read_jsonl(args.policy_labels)
    metric = high_risk_policy_violation_rate(
        rule_results,
        policy_labels,
        pass_threshold=args.pass_threshold,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"hrpvr={metric['rate']} total={metric['total']} violations={metric['violations']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    scored = read_jsonl(args.results)
    report = build_markdown_report(scored, title=args.title)
    write_text(args.output, report)
    print(f"report={args.output}")
    return 0


def cmd_peek(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    for record in records[: args.n]:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"shown={min(len(records), args.n)} total={len(records)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecomcarelm", description="EComCareLM 数据构建与评测工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean", help="清洗原始 JSONL，完成脱敏和去重")
    clean.add_argument("--input", required=True)
    clean.add_argument("--output", required=True)
    clean.add_argument("--dedup-threshold", type=float, default=0.92)
    clean.set_defaults(func=cmd_clean)

    sft = subparsers.add_parser("build-sft", help="构建 SFT JSONL 样本")
    sft.add_argument("--input", required=True)
    sft.add_argument("--output", required=True)
    sft.set_defaults(func=cmd_build_sft)

    dpo = subparsers.add_parser("build-dpo", help="从 SFT 样本构建 DPO 偏好对")
    dpo.add_argument("--input", required=True)
    dpo.add_argument("--output", required=True)
    dpo.add_argument("--negative-type", default="policy_conflict")
    dpo.set_defaults(func=cmd_build_dpo)

    dpo_pred = subparsers.add_parser("build-dpo-from-predictions", help="从模型预测 badcase 构建 DPO 偏好对")
    dpo_pred.add_argument("--eval-set", required=True)
    dpo_pred.add_argument("--predictions", required=True)
    dpo_pred.add_argument("--output", required=True)
    dpo_pred.add_argument("--max-items", type=int)
    dpo_pred.set_defaults(func=cmd_build_dpo_from_predictions)

    hf = subparsers.add_parser("import-hf", help="从 Hugging Face datasets 导入公开数据")
    hf.add_argument("--dataset", required=True)
    hf.add_argument("--config")
    hf.add_argument("--split", default="train")
    hf.add_argument("--question-field", required=True)
    hf.add_argument("--answer-field", required=True)
    hf.add_argument("--context-field")
    hf.add_argument("--policy-field")
    hf.add_argument("--scenario", default="public_dataset")
    hf.add_argument("--limit", type=int)
    hf.add_argument("--output", default="data/processed/public_import.jsonl")
    hf.add_argument("--train-output")
    hf.add_argument("--dev-output")
    hf.add_argument("--test-output")
    hf.add_argument("--train-ratio", type=float, default=0.9)
    hf.add_argument("--dev-ratio", type=float, default=0.05)
    hf.add_argument("--seed", type=int, default=42)
    hf.set_defaults(func=cmd_import_hf)

    demo = subparsers.add_parser("demo", help="生成可复现的规则 baseline 预测")
    demo.add_argument("--eval-set", required=True)
    demo.add_argument("--output", required=True)
    demo.add_argument("--model-name", default="rule_baseline")
    demo.set_defaults(func=cmd_demo)

    evaluate = subparsers.add_parser("eval", help="根据评测集为预测结果打分")
    evaluate.add_argument("--eval-set", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--summary")
    evaluate.set_defaults(func=cmd_eval)

    judge = subparsers.add_parser("judge", help="调用 LLM Judge 生成质检评分")
    judge.add_argument("--eval-set", required=True)
    judge.add_argument("--predictions", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--model", default="gpt-4o-mini")
    judge.add_argument("--temperature", type=float, default=0.0)
    judge.set_defaults(func=cmd_judge)

    label_policy = subparsers.add_parser("label-policy", help="Label policy violations for eval predictions")
    label_policy.add_argument("--eval-set", required=True)
    label_policy.add_argument("--predictions", required=True)
    label_policy.add_argument("--output", required=True)
    label_policy.set_defaults(func=cmd_label_policy)

    disagreements = subparsers.add_parser("mine-disagreements", help="Mine rule-vs-judge dimension disagreements")
    disagreements.add_argument("--rule-results", required=True)
    disagreements.add_argument("--judge-results", required=True)
    disagreements.add_argument("--output", required=True)
    disagreements.add_argument("--threshold", type=float, default=0.35)
    disagreements.add_argument("--pass-threshold", type=float, default=0.75)
    disagreements.set_defaults(func=cmd_mine_disagreements)

    pgdm = subparsers.add_parser("build-pgdm-dpo", help="Build PGDM-DPO preference pairs")
    pgdm.add_argument("--eval-set", required=True)
    pgdm.add_argument("--predictions", required=True)
    pgdm.add_argument("--policy-labels", required=True)
    pgdm.add_argument("--disagreements", required=True)
    pgdm.add_argument("--output", required=True)
    pgdm.add_argument("--max-items", type=int)
    pgdm.add_argument("--policy-cap", type=float, default=0.5)
    pgdm.add_argument("--min-label-confidence", type=float, default=0.5)
    pgdm.add_argument("--include-unknown", action="store_true")
    pgdm.set_defaults(func=cmd_build_pgdm_dpo)

    hrpvr = subparsers.add_parser("hrpvr", help="Compute High-Risk Policy Violation Rate")
    hrpvr.add_argument("--rule-results", required=True)
    hrpvr.add_argument("--policy-labels", required=True)
    hrpvr.add_argument("--output", required=True)
    hrpvr.add_argument("--pass-threshold", type=float, default=0.75)
    hrpvr.set_defaults(func=cmd_hrpvr)

    report = subparsers.add_parser("report", help="生成 markdown 评测报告")
    report.add_argument("--results", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--title", default="EComCareLM 评测报告")
    report.set_defaults(func=cmd_report)

    peek = subparsers.add_parser("peek", help="按 UTF-8 正确预览 JSONL 样本")
    peek.add_argument("--input", required=True)
    peek.add_argument("-n", type=int, default=3)
    peek.set_defaults(func=cmd_peek)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raise SystemExit(args.func(args))
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
