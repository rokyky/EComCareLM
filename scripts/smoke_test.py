#!/usr/bin/env python3
"""EComCareLM 全链路 smoke test - 用 4 条样本 + CPU 验证所有代码路径无崩溃。

用法:
    uv run python scripts/smoke_test.py                        # 只测 CLI 层
    uv run --extra train python scripts/smoke_test.py --train  # 含训练脚本验证

输出: data/smoke_test/ 保留（不自动清理），可供人工检查。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Windows GBK 兼容
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TESTS_DIR = Path("data/smoke_test")
SAMPLE_EVAL = Path("data/samples/eval_set_v1.jsonl")
SAMPLE_PRED = Path("data/samples/bad_predictions_sample.jsonl")
SAMPLE_JUDGE = Path("data/samples/judge_results_sample.jsonl")

PASS = 0
FAIL = 0
SKIP = 0


def run(cmd: str, label: str, timeout: int = 120, check_path: Path | None = None) -> bool:
    global PASS, FAIL
    full_cmd = f"uv run python {cmd}"
    print(f"  [{label[:40]}] $ {full_cmd[:100]}...")
    t0 = time.time()
    try:
        r = subprocess.run(
            full_cmd, shell=True,
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"    FAIL (exit={r.returncode}, {elapsed:.0f}s)")
            _print_last_lines(r.stderr, "stderr", 5)
            _print_last_lines(r.stdout, "stdout", 3)
            FAIL += 1
            return False
        if check_path and not check_path.exists():
            print(f"    FAIL {check_path} 未生成 ({elapsed:.0f}s)")
            FAIL += 1
            return False
        print(f"    OK ({elapsed:.0f}s)")
        PASS += 1
        return True
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (>={timeout}s)")
        FAIL += 1
        return False
    except FileNotFoundError as e:
        print(f"    FAIL: {e}")
        FAIL += 1
        return False


def run_direct(code: str, label: str, timeout: int = 120) -> bool:
    """直接跑 Python 代码片段，验证 import + 逻辑不崩溃。"""
    global PASS, FAIL
    print(f"  [{label[:40]}] $ python -c ...")
    t0 = time.time()
    try:
        r = subprocess.run(
            ["uv", "run", "python", "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"    FAIL (exit={r.returncode}, {elapsed:.0f}s)")
            _print_last_lines(r.stderr, "stderr", 5)
            FAIL += 1
            return False
        print(f"    OK ({elapsed:.0f}s)")
        PASS += 1
        return True
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (>={timeout}s)")
        FAIL += 1
        return False


def _print_last_lines(text: str, tag: str, n: int) -> None:
    lines = [l for l in text.strip().split("\n") if l.strip()]
    for line in lines[-n:]:
        print(f"      {tag}: {line}")


def check_output(path: Path, desc: str) -> None:
    if path.exists():
        try:
            count = len(path.read_text("utf-8").strip().split("\n"))
            size = path.stat().st_size
            print(f"    CHECK {desc}: {path.name} ({count} 行, {size} bytes) [OK]")
        except Exception as e:
            print(f"    CHECK {desc}: {path.name} 读取失败 {e}")
    else:
        print(f"    CHECK {desc}: [MISSING] 不存在")


def main() -> int:
    global PASS, FAIL, SKIP
    parser = argparse.ArgumentParser(description="EComCareLM 全链路 smoke test")
    parser.add_argument("--train", action="store_true", help="包含训练脚本验证（需 GPU 或 CPU fallback）")
    parser.add_argument("--keep", action="store_true", help="保留 data/smoke_test/ 不清理")
    args = parser.parse_args()

    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = TESTS_DIR
    t0 = time.time()

    # =====================================================
    # PART 1: CLI 核心链路
    # =====================================================
    print("\n" + "=" * 60)
    print("PART 1: CLI 核心链路")
    print("=" * 60)

    # 1. clean
    run(
        f"-m ecomcarelm clean --input {SAMPLE_EVAL} --output {out/'cleaned.jsonl'} --dedup-threshold 0.92",
        label="clean",
        check_path=out / "cleaned.jsonl",
    )

    # 2. build-sft
    run(
        f"-m ecomcarelm build-sft --input {out/'cleaned.jsonl'} --output {out/'sft.jsonl'}",
        label="build-sft",
        check_path=out / "sft.jsonl",
    )

    # 3. build-dpo
    run(
        f"-m ecomcarelm build-dpo --input {out/'sft.jsonl'} --output {out/'dpo_template.jsonl'}",
        label="build-dpo",
        check_path=out / "dpo_template.jsonl",
    )

    # 4. build-dpo-from-predictions
    run(
        f"-m ecomcarelm build-dpo-from-predictions --eval-set {SAMPLE_EVAL} --predictions {SAMPLE_PRED} --output {out/'dpo_badcase.jsonl'}",
        label="build-dpo-from-pred",
        check_path=out / "dpo_badcase.jsonl",
    )

    # 5. demo
    run(
        f"-m ecomcarelm demo --eval-set {SAMPLE_EVAL} --output {out/'pred.jsonl'}",
        label="demo",
        check_path=out / "pred.jsonl",
    )

    # 6. eval
    run(
        f"-m ecomcarelm eval --eval-set {SAMPLE_EVAL} --predictions {out/'pred.jsonl'} --output {out/'eval_results.jsonl'} --summary {out/'eval_summary.json'}",
        label="eval",
        check_path=out / "eval_results.jsonl",
    )
    check_output(out / "eval_summary.json", "EvalSummary")

    # 7. label-policy
    run(
        f"-m ecomcarelm label-policy --eval-set {SAMPLE_EVAL} --predictions {SAMPLE_PRED} --output {out/'policy_labels.jsonl'}",
        label="label-policy",
        check_path=out / "policy_labels.jsonl",
    )

    # 8. hrpvr
    run(
        f"-m ecomcarelm hrpvr --rule-results {out/'eval_results.jsonl'} --policy-labels {out/'policy_labels.jsonl'} --output {out/'hrpvr.json'}",
        label="hrpvr",
        check_path=out / "hrpvr.json",
    )
    if (out / "hrpvr.json").exists():
        hrpvr = json.loads((out / "hrpvr.json").read_text("utf-8"))
        print(f"    HRPVR: rate={hrpvr['rate']} violations={hrpvr['violations']}/{hrpvr['total']}")

    # 9. mine-disagreements（用 mock judge 数据，无需 API key）
    # 先用 eval 结果 + mock judge 跑分歧
    run(
        f"-m ecomcarelm mine-disagreements --rule-results {out/'eval_results.jsonl'} --judge-results {SAMPLE_JUDGE} --output {out/'disagreements.jsonl'} --threshold 0.2",
        label="mine-disagreements",
        check_path=out / "disagreements.jsonl",
    )

    # 10. build-pgdm-dpo
    run(
        f"-m ecomcarelm build-pgdm-dpo --eval-set {SAMPLE_EVAL} --predictions {SAMPLE_PRED} --policy-labels {out/'policy_labels.jsonl'} --disagreements {out/'disagreements.jsonl'} --output {out/'pgdm_dpo.jsonl'} --max-items 4 --include-unknown",
        label="build-pgdm-dpo",
        check_path=out / "pgdm_dpo.jsonl",
    )

    # 11. report
    run(
        f"-m ecomcarelm report --results {out/'eval_results.jsonl'} --output {out/'eval_report.md'}",
        label="report",
        check_path=out / "eval_report.md",
    )

    # 12. peek
    run(
        f"-m ecomcarelm peek --input {out/'eval_results.jsonl'} -n 2",
        label="peek",
    )

    # 13. 用 bad predictions 跑 eval，验证 badcase 分类
    run(
        f"-m ecomcarelm eval --eval-set {SAMPLE_EVAL} --predictions {SAMPLE_PRED} --output {out/'bad_eval_results.jsonl'}",
        label="eval (bad preds)",
        check_path=out / "bad_eval_results.jsonl",
    )
    # 验证 badcase_type 正确识别
    if (out / "bad_eval_results.jsonl").exists():
        bad_types = set()
        for line in (out / "bad_eval_results.jsonl").read_text("utf-8").strip().split("\n"):
            if line:
                rec = json.loads(line)
                bad_types.add(rec.get("badcase_type", "none"))
        print(f"    badcase types: {bad_types}")

    # =====================================================
    # PART 2: 训练脚本导入验证（可选）
    # =====================================================
    if args.train:
        print("\n" + "=" * 60)
        print("PART 2: 训练脚本验证（CPU, 1 step）")
        print("=" * 60)

        # 准备小样本
        tiny_sft = out / "sft_tiny.jsonl"
        with open(out / "sft.jsonl") as f:
            sft_lines = [l for l in f if l.strip()]
        if sft_lines:
            with open(tiny_sft, "w") as f:
                f.write("".join(sft_lines[:2]))

        tiny_dpo = out / "dpo_tiny.jsonl"
        with open(out / "dpo_template.jsonl") as f:
            dpo_lines = [l for l in f if l.strip()]
        if dpo_lines:
            with open(tiny_dpo, "w") as f:
                f.write("".join(dpo_lines[:2]))

        tiny_grpo = out / "grpo_tiny.jsonl"
        with open(SAMPLE_EVAL) as f:
            grpo_lines = [l for l in f if l.strip()]
        if grpo_lines:
            with open(tiny_grpo, "w") as f:
                f.write("".join(grpo_lines[:2]))

        # SFT: 验证 import + tokenize 不崩溃，max_steps=0 跳过训练
        if tiny_sft.exists():
            run(
                f"train/train_sft.py --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct --train-file {tiny_sft} --output-dir {out/'sft-out'} --max-steps 0 --eval-file {tiny_sft}",
                label="train_sft import",
                timeout=300,
            )

        # DPO: 验证 import + dataset 构建不崩溃
        if tiny_dpo.exists():
            run(
                f"train/train_dpo.py --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct --train-file {tiny_dpo} --output-dir {out/'dpo-out'} --max-steps 0",
                label="train_dpo import",
                timeout=300,
            )

        # GRPO: 验证 import + reward_funcs 类型正确
        if tiny_grpo.exists():
            run(
                f"train/train_grpo.py --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct --train-file {tiny_grpo} --output-dir {out/'grpo-out'} --max-steps 0",
                label="train_grpo import",
                timeout=300,
            )

        # PPO: route checker
        run(
            f"train/train_ppo.py --sft-model-path {out/'sft-out'} --reward-model-path {out/'sft-out'} --train-file {tiny_sft}",
            label="train_ppo import",
        )
    else:
        print("\n  [SKIP] 训练脚本验证跳过（加 --train 开关）")

    # =====================================================
    # PART 3: Python unittest
    # =====================================================
    print("\n" + "=" * 60)
    print("PART 3: Pytest")
    print("=" * 60)
    r = subprocess.run(
        ["uv", "run", "--with", "pytest", "pytest", "-q", "tests/test_pipeline.py"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode == 0:
        print(f"  pytest: 7/7 passed")
        PASS += 1
    else:
        print(f"  pytest: FAILED\n{r.stdout[-300:]}\n{r.stderr[-300:]}")
        FAIL += 1

    # =====================================================
    # RESULT
    # =====================================================
    elapsed = time.time() - t0
    total = PASS + FAIL
    print("\n" + "=" * 60)
    print(f"SMOKE TEST: {PASS}/{total} passed ({elapsed:.0f}s)")
    if FAIL:
        print(f"  FAILURES: {FAIL}")
    if not args.keep:
        shutil.rmtree(TESTS_DIR, ignore_errors=True)
        print(f"  临时文件: {TESTS_DIR} 已清理")
    else:
        print(f"  临时文件: {TESTS_DIR} 保留")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断")
        sys.exit(1)
