from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


JsonObject = dict[str, Any]


def read_jsonl(path: str | Path) -> list[JsonObject]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{file_path}")

    records: list[JsonObject] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败：{file_path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL 每一行必须是对象：{file_path}:{line_no}")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[JsonObject]) -> int:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with file_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_json(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str | Path, content: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8-sig", newline="\n")
