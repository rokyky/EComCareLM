# Change Proposal: Add PGDM-DPO MVP

## Goals

- Add a first-stage Policy-Grounded Dimensional Disagreement Mining (PGDM) pipeline for DPO data construction.
- Decide the first data route as A + C:
  - A: use public Hugging Face ecommerce QA/instruction data as the main SFT/DPO source.
  - C: expand the tiny local eval set for pipeline debugging before formal runs.
- Keep `chosen` fixed to human/reference customer-service answers in phase 1, so ablations compare only the `rejected` construction strategy.
- Define High-Risk Policy Violation Rate (HRPVR) as the primary experiment metric for `refund_policy`, `privacy_policy`, and `escalation_policy`.
- Add CLI support for policy labeling, dimensional disagreement mining, and PGDM-DPO pair generation.

## Non-Goals

- Do not run full SFT/DPO training in this change.
- Do not add GRPO reward changes in phase 1.
- Do not generate GPT-rewritten `chosen` answers in phase 1.
- Do not vendor large public datasets into the repository.
- Do not claim experimental improvements until real training and ablation runs are completed.

## Impact Scope

- New modules:
  - `ecomcarelm/policy.py`
  - `ecomcarelm/policy_labeler.py`
  - `ecomcarelm/disagreement.py`
- Updated modules:
  - `ecomcarelm/evaluation.py`
  - `ecomcarelm/judge.py`
  - `ecomcarelm/builders.py`
  - `ecomcarelm/cli.py`
- Tests:
  - Extend `tests/test_pipeline.py` with policy attribution, disagreement mining, and PGDM-DPO construction coverage.

