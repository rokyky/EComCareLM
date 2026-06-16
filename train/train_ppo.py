from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="EComCareLM PPO 训练路线检查器")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-model-path", required=True)
    parser.add_argument("--reward-model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="outputs/ecomcarelm-ppo")
    parser.add_argument("--total-episodes", type=int, default=10000)
    args = parser.parse_args()

    try:
        import trl.experimental.ppo  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("缺少 PPO 训练依赖，请先运行：uv sync --extra train") from exc

    print("PPO 需要 policy、reference/value 结构和 reward model。")
    print("本项目建议先用 SFT + DPO/GRPO；只有在已经训练出稳定 reward model 后再跑 PPO。")
    print("当前参数已通过前置检查：")
    print(f"- model_name_or_path={args.model_name_or_path}")
    print(f"- sft_model_path={args.sft_model_path}")
    print(f"- reward_model_path={args.reward_model_path}")
    print(f"- train_file={args.train_file}")
    print(f"- output_dir={args.output_dir}")
    print(f"- total_episodes={args.total_episodes}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
