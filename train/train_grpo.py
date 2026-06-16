from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rewards import 客服规则奖励


def _load_grpo_deps() -> tuple[Any, ...]:
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("缺少 GRPO 训练依赖，请先运行：uv sync --extra train") from exc
    return Dataset, LoraConfig, GRPOConfig, GRPOTrainer


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt(record: dict[str, Any]) -> str:
    parts = [
        "你是中文电商平台客服，请根据用户问题、上下文和平台规则回答。要求准确、完整、礼貌，不能过度承诺。",
        f"用户问题：{record.get('question') or record.get('input') or record.get('prompt') or ''}",
    ]
    if record.get("context"):
        parts.append(f"上下文：{record['context']}")
    if record.get("policy"):
        parts.append(f"平台规则：{record['policy']}")
    return "\n".join(parts)


def to_grpo_dataset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        samples.append(
            {
                "prompt": build_prompt(record),
                "must_include": "|".join(str(item) for item in record.get("must_include", [])),
                "must_not_include": "|".join(str(item) for item in record.get("must_not_include", [])),
                "scenario": record.get("scenario", "unknown"),
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="EComCareLM GRPO 训练脚本")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="outputs/ecomcarelm-grpo")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    Dataset, LoraConfig, GRPOConfig, GRPOTrainer = _load_grpo_deps()
    records = read_jsonl(args.train_file)
    dataset = Dataset.from_list(to_grpo_dataset(records))

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_steps=20,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=config,
        reward_funcs=客服规则奖励,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
