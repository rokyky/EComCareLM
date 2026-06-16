"""从 JDDC-Baseline-Seq2Seq 导入真实京东客服对话数据。

JDDC baseline 数据包含约 1 万 session 的脱敏客服对话。
输出 SFT + Eval 格式，兼容 ecomcarelm 全链路。

用法:
    uv run python scripts/import_jddc_baseline.py [--sft-count 3000] [--eval-count 500]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CHAT_URL = "https://raw.githubusercontent.com/SimonJYang/JDDC-Baseline-Seq2Seq/master/data/chat.txt"

# 场景推断规则
SCENARIO_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"退货|退款|退钱|退换|售后|质量问题|破损|瑕疵|不想要了|退货运费"), "return_refund"),
    (re.compile(r"物流|快递|配送|发货|送到|到货|物流信息|物流核查|催件|未收到|丢件|签收"), "logistics"),
    (re.compile(r"投诉|举报|态度差|差评|升级|经理|人工|维权|12315"), "complaint"),
    (re.compile(r"密码|验证码|银行卡|身份证|隐私|账号被盗|信息泄露|手机号|改密"), "privacy"),
    (re.compile(r"编|伪造|假|凭证|证明|虚假|欺诈|骗"), "safety_refusal"),
    (re.compile(r"参数|规格|尺寸|功能|能不能用|怎么用|说明|说明书|功率"), "inquiry"),
    (re.compile(r"发票|增票|专票|普票|税号|开票"), "inquiry"),
    (re.compile(r"价格|降价|涨价|价保|保价|差价|优惠|折扣|券|满减"), "inquiry"),
    (re.compile(r"维修|换货|换新|返修|保修|售后"), "return_refund"),
    (re.compile(r"地址|修改地址|改地址|收货地址|自提"), "logistics"),
]

MIN_QUESTION_LENGTH = 4
MIN_ANSWER_LENGTH = 6
MAX_ANSWER_LENGTH = 300
SKIP_PATTERNS = re.compile(r"^[#E\-sa-z\d\[\]x\s，。？、！；：""''【】《》]+$")

# 脱敏标记正常化
NORMALIZE_PATTERNS = [
    (re.compile(r"\[ORDERID_\d+\]"), "[订单编号]"),
    (re.compile(r"\[金额x\]"), "[金额]"),
    (re.compile(r"\[日期x\]"), "[日期]"),
    (re.compile(r"\[时间x\]"), "[时间]"),
    (re.compile(r"\[数字x\]"), "[数字]"),
    (re.compile(r"\[姓名x\]"), "[姓名]"),
    (re.compile(r"\[电话x\]"), "[电话]"),
    (re.compile(r"\[地址x\]"), "[地址]"),
    (re.compile(r"\[站点x\]"), "[站点]"),
    (re.compile(r"USERID_\d+"), "用户"),
    (re.compile(r"#E-s\[\d+\]"), ""),
    (re.compile(r"\s+"), " "),
]


def normalize(text: str) -> str:
    for pat, repl in NORMALIZE_PATTERNS:
        text = pat.sub(repl, text)
    return text.strip()


def infer_scenario(question: str, answer: str) -> str:
    combined = f"{question} {answer}"
    for pattern, scenario in SCENARIO_RULES:
        if pattern.search(combined):
            return scenario
    return "general"


def fetch_chat_data(url: str) -> list[dict[str, str]]:
    local_path = Path("data/generated/jddc_chat.txt")
    if local_path.exists():
        print(f"使用本地缓存: {local_path}")
        raw = local_path.read_text("utf-8")
    else:
        print(f"下载 {url} ...")
        try:
            import subprocess, sys, tempfile
            tmp = Path(tempfile.mktemp(suffix=".txt"))
            code = f"import urllib.request; open(r'{tmp.as_posix()}','wb').write(urllib.request.urlopen('{url}', timeout=120).read())"
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                raise RuntimeError(f"下载失败: {result.stderr[:200]}")
            raw = tmp.read_text("utf-8")
            tmp.unlink(missing_ok=True)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(raw, encoding="utf-8")
            print(f"  已缓存到 {local_path}")
        except Exception as e:
            raise RuntimeError(
                f"无法从 GitHub 下载数据（{e}）。\n"
                f"请手动下载以下文件放到 {local_path.resolve()}：\n"
                f"  {url}\n"
                f"然后重新运行本脚本。"
            )

    lines = raw.strip().split("\n")
    print(f"  共 {len(lines)} 行")
    reader = csv.DictReader(lines, delimiter="\t")
    rows = [row for row in reader]
    print(f"  解析 {len(rows)} 条记录")
    return rows


def group_sessions(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sid = row.get("session_id", "").strip()
        if sid:
            sessions[sid].append(row)
    return sessions


def extract_qa_pairs(sessions: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    """从会话中提取 (question, answer) 对。取每个客服回复前最近的用户问题作为 question。"""
    pairs: list[dict[str, str]] = []
    for sid, messages in sessions.items():
        last_customer_msg = ""
        last_customer_msgs: list[str] = []
        for msg in messages:
            content = msg.get("content", "").strip()
            is_waiter = msg.get("waiter_send", "0") == "1"
            if not content or SKIP_PATTERNS.match(content):
                continue
            if not is_waiter:
                last_customer_msg = normalize(content)
                if last_customer_msg:
                    last_customer_msgs.append(last_customer_msg)
            elif is_waiter and last_customer_msg:
                answer = normalize(content)
                if len(answer) < MIN_ANSWER_LENGTH or len(answer) > MAX_ANSWER_LENGTH:
                    continue
                context_msgs = last_customer_msgs[-3:-1] if len(last_customer_msgs) > 1 else []
                context = "；".join(context_msgs) if context_msgs else ""
                pairs.append({"question": last_customer_msg, "context": context, "answer": answer})
                last_customer_msg = ""
    return pairs


def build_sft(pairs: list[dict[str, str]], count: int) -> list[dict[str, Any]]:
    sampled = random.sample(pairs, min(count, len(pairs)))
    records = []
    for idx, pair in enumerate(sampled):
        scenario = infer_scenario(pair["question"], pair["answer"])
        records.append({
            "case_id": f"sft_{idx+1:08d}",
            "scenario": scenario,
            "question": pair["question"],
            "context": pair["context"],
            "policy": "",
            "answer": pair["answer"],
            "must_include": [],
            "must_not_include": [],
        })
    return records


def build_eval(pairs: list[dict[str, str]], count: int) -> list[dict[str, Any]]:
    high_risk = {"return_refund", "logistics", "complaint", "privacy", "safety_refusal"}
    by_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
    for pair in pairs:
        scenario = infer_scenario(pair["question"], pair["answer"])
        by_scenario[scenario].append(pair)

    selected = []
    for scenario in high_risk:
        pool = by_scenario.get(scenario, [])
        random.shuffle(pool)
        selected.extend(pool[: min(20, len(pool))])

    remaining = count - len(selected)
    if remaining > 0:
        others = [p for s, pool in by_scenario.items() if s not in high_risk for p in pool]
        random.shuffle(others)
        selected.extend(others[:remaining])

    seen = set()
    unique = []
    for p in selected:
        key = p["question"][:40]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    sampled = unique[:count]
    records = []
    for idx, pair in enumerate(sampled):
        scenario = infer_scenario(pair["question"], pair["answer"])
        records.append({
            "case_id": f"eval_{idx+1:08d}",
            "scenario": scenario,
            "question": pair["question"],
            "context": pair["context"],
            "policy": "",
            "gold_answer": pair["answer"],
            "must_include": [],
            "must_not_include": [],
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="导入 JDDC Baseline 数据")
    parser.add_argument("--sft-count", type=int, default=3000)
    parser.add_argument("--eval-count", type=int, default=400)
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = fetch_chat_data(CHAT_URL)
    sessions = group_sessions(rows)
    print(f"会话数: {len(sessions)}")

    all_pairs = extract_qa_pairs(sessions)
    print(f"QA 对: {len(all_pairs)}")

    scenario_counts = Counter(infer_scenario(p["question"], p["answer"]) for p in all_pairs)
    print(f"场景分布: {dict(scenario_counts)}")

    sft = build_sft(all_pairs, args.sft_count)
    sft_path = out_dir / "sft_train.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SFT: {len(sft)} 条 -> {sft_path}")

    eval_records = build_eval(all_pairs, args.eval_count)
    eval_path = out_dir / "eval_set.jsonl"
    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Eval: {len(eval_records)} 条 -> {eval_path}")

    sft_scenarios = Counter(r["scenario"] for r in sft)
    eval_scenarios = Counter(r["scenario"] for r in eval_records)
    print(f"\nSFT 场景分布: {dict(sft_scenarios)}")
    print(f"Eval 场景分布: {dict(eval_scenarios)}")

    if eval_records:
        print(f"\nEval 示例:")
        print(json.dumps(eval_records[0], ensure_ascii=False, indent=2)[:500])


if __name__ == "__main__":
    main()
