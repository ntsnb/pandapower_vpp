from __future__ import annotations

from scripts.run_reward_v2_matrix import REWARD_MATRIX_CASES, build_matrix_commands


def test_reward_v2_matrix_runner_covers_requested_cases() -> None:
    case_ids = [case.case_id for case in REWARD_MATRIX_CASES]

    assert case_ids == [
        "A_legacy_v1_reward",
        "B_v2_minimal",
        "C_v2_minimal_no_shield_eval",
        "D_v2_minimal_no_portfolio_window_penalty",
        "E_v2_minimal_with_preferred_bonus_0p05",
        "F1_v2_minimal_contract_delivery_weight_5",
        "F2_v2_minimal_contract_delivery_weight_10",
        "F3_v2_minimal_contract_delivery_weight_20",
    ]


def test_reward_v2_matrix_runner_builds_paper_training_commands() -> None:
    commands = build_matrix_commands(
        output_root="outputs/test_matrix",
        preset="paper_long_sensitivity_v1",
        algorithms="rule_based,happo",
        progress_interval_seconds=30.0,
    )

    assert len(commands) == len(REWARD_MATRIX_CASES)
    assert all("examples/17_paper_training_experiment.py" in " ".join(command) for command in commands)
    assert all("--preset" in command and "paper_long_sensitivity_v1" in command for command in commands)
    assert all("--algorithms" in command and "rule_based,happo" in command for command in commands)
    assert any("legacy_v1_reward.yaml" in command for command in commands[0])
    assert any("reward_v2_minimal.yaml" in command for command in commands[1])
