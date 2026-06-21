from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALGORITHMS = "rule_based,no_flex,happo,hatrpo"


@dataclass(frozen=True)
class RewardMatrixCase:
    case_id: str
    config_path: str
    description: str


REWARD_MATRIX_CASES: tuple[RewardMatrixCase, ...] = (
    RewardMatrixCase(
        "A_legacy_v1_reward",
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1_legacy_v1_reward.yaml",
        "Legacy reward v1 control arm.",
    ),
    RewardMatrixCase(
        "B_v2_minimal",
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml",
        "Main reward_v2_minimal design.",
    ),
    RewardMatrixCase(
        "C_v2_minimal_no_shield_eval",
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_no_shield_eval.yaml",
        "Safety layer still executes; shield/projection penalty coefficients are disabled for reward ablation.",
    ),
    RewardMatrixCase(
        "D_v2_minimal_no_portfolio_window_penalty",
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_no_portfolio_window_penalty.yaml",
        "Portfolio window future reliability/shield/projection penalties disabled.",
    ),
    RewardMatrixCase(
        "E_v2_minimal_with_preferred_bonus_0p05",
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_preferred_bonus_0p05.yaml",
        "Dispatch preferred-region bonus restored at a small 0.05 weight.",
    ),
    RewardMatrixCase(
        "F1_v2_minimal_contract_delivery_weight_5",
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_5.yaml",
        "Contract delivery shortfall weight = 5.",
    ),
    RewardMatrixCase(
        "F2_v2_minimal_contract_delivery_weight_10",
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_10.yaml",
        "Contract delivery shortfall weight = 10.",
    ),
    RewardMatrixCase(
        "F3_v2_minimal_contract_delivery_weight_20",
        "configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_20.yaml",
        "Contract delivery shortfall weight = 20.",
    ),
)


def build_matrix_commands(
    *,
    output_root: str | Path,
    preset: str,
    algorithms: str,
    progress_interval_seconds: float,
    resume_completed: bool = False,
) -> list[list[str]]:
    root = Path(output_root)
    commands: list[list[str]] = []
    for case in REWARD_MATRIX_CASES:
        output_dir = root / case.case_id
        command = [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "17_paper_training_experiment.py"),
            "--preset",
            preset,
            "--config-path",
            case.config_path,
            "--algorithms",
            algorithms,
            "--output-dir",
            str(output_dir),
            "--progress-interval-seconds",
            str(float(progress_interval_seconds)),
        ]
        if resume_completed:
            command.append("--resume-completed")
        commands.append(command)
    return commands


def write_matrix_plan(path: str | Path, commands: list[list[str]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "matrix_cases": [asdict(case) for case in REWARD_MATRIX_CASES],
        "commands": commands,
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or execute the reward_v2_minimal paper-training matrix.")
    parser.add_argument("--preset", default="paper_long_sensitivity_v1")
    parser.add_argument("--output-root", default="outputs/reward_v2_matrix")
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--progress-interval-seconds", type=float, default=60.0)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--plan-path", default=None)
    parser.add_argument("--execute", action="store_true", help="Run every matrix case sequentially. Without this, only print and write the plan.")
    args = parser.parse_args()

    commands = build_matrix_commands(
        output_root=args.output_root,
        preset=args.preset,
        algorithms=args.algorithms,
        progress_interval_seconds=float(args.progress_interval_seconds),
        resume_completed=bool(args.resume_completed),
    )
    plan_path = Path(args.plan_path) if args.plan_path else Path(args.output_root) / "reward_v2_matrix_plan.json"
    write_matrix_plan(plan_path, commands)
    print(f"Matrix plan: {plan_path}")
    for case, command in zip(REWARD_MATRIX_CASES, commands):
        print(f"[{case.case_id}] {' '.join(command)}")
        if args.execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
