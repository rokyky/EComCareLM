$ErrorActionPreference = "Stop"
$env:UV_LINK_MODE = "copy"
$badTitle = "EComCareLM Badcase " + [char]0x6F14 + [char]0x793A + [char]0x62A5 + [char]0x544A

uv run python -m ecomcarelm clean --input data/raw/ecommerce_faq_sample.jsonl --output data/processed/cleaned.jsonl
uv run python -m ecomcarelm build-sft --input data/processed/cleaned.jsonl --output data/processed/sft_train.jsonl
uv run python -m ecomcarelm build-dpo --input data/processed/sft_train.jsonl --output data/processed/dpo_train.jsonl --negative-type policy_conflict
uv run python -m ecomcarelm demo --eval-set data/samples/eval_set_v1.jsonl --output data/processed/predictions.jsonl
uv run python -m ecomcarelm eval --eval-set data/samples/eval_set_v1.jsonl --predictions data/processed/predictions.jsonl --output data/processed/eval_results.jsonl --summary data/processed/eval_summary.json
uv run python -m ecomcarelm report --results data/processed/eval_results.jsonl --output data/processed/eval_report.md
uv run python -m ecomcarelm eval --eval-set data/samples/eval_set_v1.jsonl --predictions data/samples/bad_predictions_sample.jsonl --output data/processed/bad_eval_results.jsonl
uv run python -m ecomcarelm report --results data/processed/bad_eval_results.jsonl --output data/processed/bad_eval_report.md --title $badTitle
uv run python -m ecomcarelm build-dpo-from-predictions --eval-set data/samples/eval_set_v1.jsonl --predictions data/samples/bad_predictions_sample.jsonl --output data/processed/dpo_from_badcase.jsonl
