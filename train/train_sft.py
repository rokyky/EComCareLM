from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _load_train_deps() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("缺少训练依赖，请先运行：uv sync --extra train") from exc

    return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments


def _load_lora_deps() -> tuple[Any, Any]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("启用 LoRA 需要 peft，请先运行：uv sync --extra train") from exc
    return LoraConfig, get_peft_model


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
    import json

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


RESPONSE_MARKER = "<|im_start|>assistant\n"


def format_sft_text(sample: dict[str, Any], eos_token: str) -> str:
    instruction = sample.get("instruction", "")
    sample_input = sample.get("input", "")
    output = sample.get("output", "")
    return f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{sample_input}<|im_end|>\n{RESPONSE_MARKER}{output}{eos_token}"


def main() -> None:
    parser = argparse.ArgumentParser(description="EComCareLM SFT 训练脚本")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file")
    parser.add_argument("--output-dir", default="outputs/ecomcarelm-sft")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
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

    torch, Dataset, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments = _load_train_deps()

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

    if args.use_lora:
        LoraConfig, get_peft_model = _load_lora_deps()
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, config)

    train_records = read_jsonl(args.train_file)
    if not train_records:
        raise ValueError(f"训练文件为空：{args.train_file}")
    eval_records = read_jsonl(args.eval_file) if args.eval_file else None

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [
            format_sft_text(
                {"instruction": ins, "input": inp, "output": out},
                tokenizer.eos_token or "",
            )
            for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
        ]
        tokenized = tokenizer(texts, truncation=True, max_length=args.max_length)

        # 只对 assistant response 部分计算 loss，prompt 部分 mask 为 -100
        marker_ids = tokenizer.encode(RESPONSE_MARKER, add_special_tokens=False)
        marker_len = len(marker_ids)
        labels = []
        for input_ids in tokenized["input_ids"]:
            label = [-100] * len(input_ids)
            # 从后往前找最后一个 response marker 的位置
            for i in range(len(input_ids) - marker_len, -1, -1):
                if input_ids[i : i + marker_len] == marker_ids:
                    label[i + marker_len :] = input_ids[i + marker_len :]
                    break
            labels.append(label)

        tokenized["labels"] = labels
        return tokenized

    train_dataset = Dataset.from_list(train_records).map(tokenize, batched=True, remove_columns=list(train_records[0].keys()))
    eval_dataset = None
    if eval_records:
        eval_dataset = Dataset.from_list(eval_records).map(tokenize, batched=True, remove_columns=list(eval_records[0].keys()))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        eval_steps=100 if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    if args.use_lora:
        # 合并 LoRA 权重，使输出目录可直接作为 --model-name-or-path 给 DPO 使用
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)

    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
