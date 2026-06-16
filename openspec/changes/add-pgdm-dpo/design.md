# Design: PGDM-DPO MVP

## Architecture

PGDM adds a data-selection layer between evaluation outputs and DPO pair construction.

1. `policy.py` defines the policy taxonomy, high-risk policies, severity, and keyword hints.
2. `policy_labeler.py` assigns policy attribution records to model responses. The MVP supports a deterministic rule labeler and a confidence schema compatible with a later LLM labeler.
3. `evaluation.py` continues to produce rule-based metrics and adds normalized dimension fields.
4. `judge.py` continues to call an LLM judge and adds a local normalization helper that maps judge output into the same dimension schema.
5. `disagreement.py` compares normalized rule and judge dimensions and emits aggregate-level plus dimension-level disagreement records.
6. `builders.py` creates PGDM-DPO pairs from eval cases, model predictions, policy labels, and disagreement records.

## Data Model

Policy label record:

```json
{
  "case_id": "case_001",
  "policy_id": "refund_policy",
  "violation_type": "over_promise",
  "confidence": 1.0,
  "source": "rule",
  "evidence": ["guaranteed refund"]
}
```

Disagreement record:

```json
{
  "case_id": "case_001",
  "aggregate_disagreement": "rule_pass_judge_fail",
  "dimension_disagreements": [
    {
      "dimension": "policy_compliance",
      "rule_score": 0.0,
      "judge_score": 1.0,
      "delta": 1.0,
      "direction": "rule_fail_judge_pass"
    }
  ]
}
```

PGDM-DPO record:

```json
{
  "id": "pgdm_dpo_00000001",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "metadata": {
    "source": "pgdm_dpo",
    "case_id": "case_001",
    "policy_id": "refund_policy",
    "eval_dimension": "policy_compliance",
    "sample_weight": 1.7
  }
}
```

## Scoring and Weighting

The first-stage sample weight is:

```text
weight = severity * disagreement_strength * label_confidence
```

Where:

- `severity` comes from the policy taxonomy.
- `disagreement_strength` is the max absolute dimension delta, with a floor of 1.0 when a badcase has no judge output.
- `label_confidence` uses the frozen MVP scale:
  - 1.0 for rule hits
  - 0.8 for high-confidence LLM labels
  - 0.5 for low-confidence LLM labels
  - 0.2 for unknown attribution, normally filtered out

Quota caps are applied per policy during PGDM-DPO construction to prevent a single high-volume policy from consuming the full DPO pool.

## Risks and Tradeoffs

- The MVP rule labeler is high precision but low recall. The interface is intentionally compatible with a later LLM policy labeler.
- Judge normalization treats 0-5 scores as 0-1 floats. Boolean failure dimensions are converted to pass-style scores where 1.0 means good.
- HRPVR is rule-evaluator based in phase 1 to keep the main metric deterministic and reproducible.

