#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "docs" / "agents" / "HANDOFF.md"


@dataclass(frozen=True)
class HarnessPhase:
    name: str
    goal: str
    tests: tuple[str, ...] = ()


PHASES: tuple[HarnessPhase, ...] = (
    HarnessPhase("phase_00_repo_map", "Map DSO, FR/DOE, actor, reward, trainer, config, tests."),
    HarnessPhase("phase_01_agents_and_memory", "Update AGENTS, memory, subagent specs, hooks."),
    HarnessPhase(
        "phase_02_schemas",
        "Validate ActionUnit, NetworkObject and sensitivity schema contracts.",
        ("tests/test_action_units.py", "tests/test_network_objects.py", "tests/test_sensitivity_shapes.py"),
    ),
    HarnessPhase(
        "phase_03_sensitivity",
        "Validate finite-difference sensitivity and cache behavior.",
        ("tests/test_sensitivity_finite_difference.py",),
    ),
    HarnessPhase(
        "phase_04_observation",
        "Validate structured DSO observation shapes, masks and privacy.",
        ("tests/test_structured_observation_shapes.py", "tests/test_privacy_no_private_cost_leak.py"),
    ),
    HarnessPhase(
        "phase_05_model",
        "Validate bipartite attention actor forward pass.",
        ("tests/test_bipartite_attention_actor.py",),
    ),
    HarnessPhase(
        "phase_06_safe_decoder",
        "Validate safe envelope decoder invariants.",
        ("tests/test_safe_decoder.py",),
    ),
    HarnessPhase(
        "phase_07_simulator_routing",
        "Validate rule_v0 and sensitivity_attention_v1 routing.",
        ("tests/test_legacy_baseline_unchanged.py", "tests/test_envelope_policy_switch.py"),
    ),
    HarnessPhase(
        "phase_08_reward_and_training",
        "Validate reward logging and no-NaN short training sanity.",
        ("tests/test_training_step_no_nan.py",),
    ),
    HarnessPhase("phase_09_experiment_configs", "Validate baseline/new/ablation configs."),
    HarnessPhase(
        "phase_10_tests",
        "Run structured and baseline guard tests.",
        (
            "tests/test_action_units.py",
            "tests/test_network_objects.py",
            "tests/test_sensitivity_shapes.py",
            "tests/test_sensitivity_finite_difference.py",
            "tests/test_structured_observation_shapes.py",
            "tests/test_bipartite_attention_actor.py",
            "tests/test_safe_decoder.py",
            "tests/test_legacy_baseline_unchanged.py",
            "tests/test_envelope_policy_switch.py",
            "tests/test_privacy_no_private_cost_leak.py",
        ),
    ),
    HarnessPhase("phase_11_docs", "Update architecture, experiment, handoff and known-failure docs."),
    HarnessPhase("phase_12_final_report", "Write final changed-files/tests/risks report."),
)


def _changed_files_from_git() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return (f"<git status failed: {result.returncode}>",)
    files: list[str] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        files.append(text[3:] if len(text) > 3 else text)
    return tuple(files)


def _append_handoff(
    phase: HarnessPhase,
    *,
    handoff_path: Path = HANDOFF,
    changed_files: Sequence[str] = (),
    test_command: Sequence[str] = (),
    test_return_code: int | None = None,
) -> None:
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    with handoff_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {phase.name}\n\n")
        handle.write(f"- Goal: {phase.goal}\n")
        handle.write("- Files touched:\n")
        if changed_files:
            for item in changed_files:
                handle.write(f"  - `{item}`\n")
        else:
            handle.write("  - `(none reported)`\n")
        if test_command:
            handle.write(f"- Tests run: `{' '.join(test_command)}`\n")
            handle.write(
                "- Test result: "
                + ("passed" if test_return_code == 0 else f"failed ({test_return_code})")
                + "\n"
            )
        else:
            handle.write("- Tests run: `(no direct tests for this phase)`\n")
            handle.write("- Test result: skipped\n")


def list_phases() -> None:
    for phase in PHASES:
        tests = " ".join(phase.tests) if phase.tests else "(no direct test)"
        print(f"{phase.name}: {phase.goal} | tests: {tests}")


def run_phase(
    name: str,
    *,
    run_tests: bool = True,
    handoff_path: Path = HANDOFF,
    changed_files: Sequence[str] | None = None,
) -> int:
    by_name = {phase.name: phase for phase in PHASES}
    if name not in by_name:
        raise SystemExit(f"Unknown phase {name!r}. Use --list.")
    phase = by_name[name]
    print(f"[agent_harness] {phase.name}")
    print(f"Goal: {phase.goal}")
    files = tuple(changed_files) if changed_files is not None else _changed_files_from_git()
    command: tuple[str, ...] = ()
    return_code: int | None = None
    if phase.tests and run_tests:
        command = (sys.executable, "-m", "pytest", "-q", *phase.tests)
        print(f"Running tests: {' '.join(command)}")
        completed = subprocess.run(command, cwd=ROOT)
        return_code = int(completed.returncode)
    elif phase.tests:
        print("Tests skipped by --skip-tests:")
        for test in phase.tests:
            print(f"- {test}")
    _append_handoff(
        phase,
        handoff_path=handoff_path,
        changed_files=files,
        test_command=command,
        test_return_code=return_code,
    )
    return int(return_code or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSO sensitivity_attention_v1 agent harness.")
    parser.add_argument("--list", action="store_true", help="List harness phases.")
    parser.add_argument("--phase", help="Run one phase and append handoff note.")
    parser.add_argument("--skip-tests", action="store_true", help="Append phase note without running suggested tests.")
    args = parser.parse_args()
    if args.list:
        list_phases()
        return
    if args.phase:
        raise SystemExit(run_phase(args.phase, run_tests=not args.skip_tests))
    parser.print_help()


if __name__ == "__main__":
    main()
