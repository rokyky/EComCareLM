from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
ORDER_RE = re.compile(r"\b(?:订单号|order_id|订单)[:：]?\s*[A-Z0-9-]{8,}\b", re.IGNORECASE)
TRACKING_RE = re.compile(r"\b(?:快递单号|tracking_id|运单号)[:：]?\s*[A-Z0-9-]{8,}\b", re.IGNORECASE)
BANK_RE = re.compile(r"(?<!\d)(?:\d[ -]?){15,19}(?!\d)")
ADDRESS_RE = re.compile(r"[\u4e00-\u9fa5]{2,}(?:省|市|区|县|路|街|小区|号楼|单元|室)")


PLACEHOLDERS = [
    (PHONE_RE, "[PHONE]"),
    (ID_CARD_RE, "[ID_CARD]"),
    (ORDER_RE, "[ORDER_ID]"),
    (TRACKING_RE, "[TRACKING_ID]"),
    (BANK_RE, "[BANK_CARD]"),
    (ADDRESS_RE, "[ADDRESS]"),
]


TEXT_FIELDS = (
    "question",
    "answer",
    "context",
    "policy",
    "gold_answer",
    "input",
    "output",
    "prompt",
    "chosen",
    "rejected",
)


@dataclass(frozen=True)
class CleanStats:
    input_count: int
    output_count: int
    duplicate_count: int


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("（", "(").replace("）", ")")
    return text


def redact_pii(text: str) -> str:
    redacted = text
    for pattern, replacement in PLACEHOLDERS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_pii(normalize_text(value))
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items()}
    return value


def record_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str):
            parts.append(value)
    if not parts:
        parts = [str(value) for value in record.values() if isinstance(value, str)]
    return normalize_text(" ".join(parts)).lower()


def is_near_duplicate(text: str, previous_texts: list[str], threshold: float) -> bool:
    return any(SequenceMatcher(None, text, prev).ratio() >= threshold for prev in previous_texts)


def clean_records(records: list[dict[str, Any]], dedup_threshold: float = 0.92) -> tuple[list[dict[str, Any]], CleanStats]:
    cleaned: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    duplicates = 0

    for record in records:
        sanitized = sanitize_value(record)
        text = record_text(sanitized)
        if text and is_near_duplicate(text, fingerprints, dedup_threshold):
            duplicates += 1
            continue
        fingerprints.append(text)
        cleaned.append(sanitized)

    return cleaned, CleanStats(
        input_count=len(records),
        output_count=len(cleaned),
        duplicate_count=duplicates,
    )
