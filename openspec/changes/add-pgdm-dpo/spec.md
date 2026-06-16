# Spec: PGDM-DPO MVP

## Policy Taxonomy

- WHEN policy metadata is requested, THEN the system SHALL return stable `policy_id`, severity, high-risk status, and keyword hints.
- WHEN an unknown `policy_id` is requested, THEN the system SHALL return a default low-severity policy profile instead of crashing.

## Policy Labeling

- WHEN a response contains a high-confidence policy violation phrase, THEN the rule labeler SHALL emit a policy label with `confidence=1.0` and `source=rule`.
- WHEN no policy violation is detected, THEN the rule labeler SHALL emit `unknown_policy` with `confidence=0.2`.
- WHEN case or prediction inputs are missing optional fields, THEN policy labeling SHALL still return a valid label record.

## Dimension Normalization

- WHEN rule evaluation scores are produced, THEN they SHALL include normalized dimensions for `answer_accuracy`, `policy_compliance`, `completeness`, `politeness`, `safety`, `hallucination`, and `off_topic`.
- WHEN judge scores are normalized, THEN numeric 0-5 dimensions SHALL be mapped to 0-1 and boolean failure dimensions SHALL be mapped so 1.0 means pass.
- WHEN judge output has parse errors or missing dimensions, THEN missing dimensions SHALL be omitted without failing the whole comparison.

## Disagreement Mining

- WHEN both rule and judge scores exist for a case, THEN the system SHALL emit aggregate disagreement and dimension disagreement records.
- WHEN a dimension delta is greater than or equal to the configured threshold, THEN the system SHALL include that dimension in `dimension_disagreements`.
- WHEN rule passes and judge fails overall, THEN `aggregate_disagreement` SHALL be `rule_pass_judge_fail`.
- WHEN rule fails and judge passes overall, THEN `aggregate_disagreement` SHALL be `rule_fail_judge_pass`.
- WHEN both fail, THEN `aggregate_disagreement` SHALL be `both_fail`.
- WHEN both pass, THEN `aggregate_disagreement` SHALL be `both_pass`.

## PGDM-DPO Construction

- WHEN a case has a reference answer and a model prediction, THEN PGDM-DPO SHALL use the reference as `chosen` and the model answer as `rejected`.
- WHEN `chosen` and `rejected` are identical after trimming, THEN the case SHALL be skipped.
- WHEN policy labels and disagreements are available, THEN PGDM-DPO metadata SHALL include `policy_id`, `violation_type`, `eval_dimension`, `aggregate_disagreement`, `sample_weight`, and `label_confidence`.
- WHEN `max_items` is set, THEN output size SHALL not exceed it.
- WHEN `policy_cap` is set, THEN no policy SHALL exceed the configured fraction of the output pool unless the pool has fewer eligible records than the cap allows.

## CLI

- WHEN `label-policy` is run, THEN it SHALL read eval cases and predictions and write policy labels as JSONL.
- WHEN `mine-disagreements` is run, THEN it SHALL read rule results and judge results and write disagreement records as JSONL.
- WHEN `build-pgdm-dpo` is run, THEN it SHALL read eval cases, predictions, policy labels, and disagreements and write DPO JSONL.
- WHEN `hrpvr` is run, THEN it SHALL read rule results and policy labels and write the High-Risk Policy Violation Rate as JSON.
