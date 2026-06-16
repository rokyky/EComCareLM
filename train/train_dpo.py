from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_dpo_deps() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("缺少 DPO 训练依赖，请先运行：uv sync --extra train") from exc
    return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, DPOConfig, DPOTrainer


def _lora_config(args: argparse.Namespace) -> Any:
    if not args.use_lora:
        return None
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError("启用 LoRA 需要 peft，请先运行：uv sync --extra train") from exc
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def _quantization_config(args: argparse.Namespace, torch: Any) -> Any:
    if not args.load_in_4bit and not args.load_in_8bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("QLoRA/量化加载需要 bitsandbytes，请先运行：uv sync --extra train") from exc
    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="EComCareLM DPO 训练脚本")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="outputs/ecomcarelm-dpo")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", help="启用 4bit QLoRA 加载")
    parser.add_argument("--load-in-8bit", action="store_true", help="启用 8bit 量化加载")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    torch, Dataset, AutoModelForCausalLM, AutoTokenizer, DPOConfig, DPOTrainer = _load_dpo_deps()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = _quantization_config(args, torch)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=None if quantization_config is not None else (dtype if torch.cuda.is_available() else torch.float32),
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    records = read_jsonl(args.train_file)
    dataset = Dataset.from_list(
        [
            {
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            }
            for item in records
        ]
    )

    config = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        max_length=args.max_length,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=_lora_config(args),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
