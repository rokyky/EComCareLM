$ErrorActionPreference = "Stop"
$env:UV_LINK_MODE = "copy"

Write-Host "=== EComCareLM PGDM 全链路验证（0.5B smoke test）===" -ForegroundColor Cyan
Write-Host ""

# 第 1 步：导入 JDDC 数据
Write-Host "=== Step 1: 导入 JDDC 数据 ===" -ForegroundColor Yellow
uv run python scripts/import_jddc_baseline.py --sft-count 3000 --eval-count 500
if ($LASTEXITCODE -ne 0) { throw "Step 1 失败" }

# 第 2 步：SFT 训练（0.5B smoke test）
Write-Host "=== Step 2: SFT 训练 ===" -ForegroundColor Yellow
uv run python train/train_sft.py `
    --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct `
    --train-file data/generated/sft_train.jsonl `
    --output-dir outputs/sft-0.5b `
    --use-lora --load-in-4bit `
    --num-train-epochs 3 `
    --max-steps 100
if ($LASTEXITCODE -ne 0) { throw "Step 2 失败" }

# 第 3 步：基线评测 + Policy 标注
Write-Host "=== Step 3: 基线评测 + Policy 标注 ===" -ForegroundColor Yellow
uv run python -m ecomcarelm demo --eval-set data/generated/eval_set.jsonl --output outputs/pred-baseline.jsonl
uv run python -m ecomcarelm eval --eval-set data/generated/eval_set.jsonl --predictions outputs/pred-baseline.jsonl --output outputs/rule-baseline.jsonl --summary outputs/rule-summary.json
uv run python -m ecomcarelm label-policy --eval-set data/generated/eval_set.jsonl --predictions outputs/pred-baseline.jsonl --output outputs/policy-labels.jsonl
uv run python -m ecomcarelm hrpvr --rule-results outputs/rule-baseline.jsonl --policy-labels outputs/policy-labels.jsonl --output outputs/hrpvr-baseline.json

Write-Host "=== 基线 HRPVR ===" -ForegroundColor Green
Get-Content outputs/hrpvr-baseline.json

Write-Host ""
Write-Host "=== 完成！接下来需要 API Key 运行 LLM Judge + PGDM-DPO 全链路 ===" -ForegroundColor Cyan
Write-Host "   请执行："
Write-Host "   `$env:OPENAI_API_KEY = `"sk-xxx`""
Write-Host "   uv run python -m ecomcarelm judge ..."
Write-Host "   uv run python -m ecomcarelm mine-disagreements ..."
Write-Host "   uv run python -m ecomcarelm build-pgdm-dpo ..."
Write-Host "   详见 README.md 第 4 步"
