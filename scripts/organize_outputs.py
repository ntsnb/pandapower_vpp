from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil


ACTIVE_OUTPUTS = {
    "paper_training_long_reward_v2_minimal_20260604_gpu_decoder_bounds_happo_hatrpo_logfix",
}

CONVENTIONAL_KEEP_DIRS = {
    "dashboard_data",
    "figures",
    "vpp_first_person",
}

CONVENTIONAL_KEEP_FILES = {
    "interactive_report.html",
    "rl_architecture.html",
}


@dataclass(frozen=True)
class ArchivePlan:
    old_path: Path
    new_path: Path
    category: str
    reason: str


def classify(name: str, path: Path) -> tuple[str, str] | None:
    if name in ACTIVE_OUTPUTS:
        return None
    if path.is_dir() and name in CONVENTIONAL_KEEP_DIRS:
        return None
    if path.is_file() and name in CONVENTIONAL_KEEP_FILES:
        return None
    if name.startswith("test_") or name.startswith("pytest_tmp_") or name.startswith("tmp_"):
        return ("tests", "test or temporary output")
    if name.startswith("paper_training_long") or name.startswith("paper_long_"):
        return ("paper_long", "paper-long experiment output")
    if name.startswith("reward_v2_"):
        return ("paper_long", "reward-v2 experiment output")
    if name.startswith("audit_") or name.endswith("_audit") or "audit" in name:
        return ("audits", "audit or preflight output")
    if path.is_file() and (name.endswith(".md") or name.endswith(".html") or name.endswith(".pdf")):
        return ("reports", "root generated report")
    return None


def build_plan(outputs_dir: Path) -> list[ArchivePlan]:
    archive_root = outputs_dir / "_archive"
    plans: list[ArchivePlan] = []
    if not outputs_dir.exists():
        return plans
    for child in sorted(outputs_dir.iterdir(), key=lambda item: item.name):
        if child.name in {"_archive", "_manifests", ".cache"}:
            continue
        classified = classify(child.name, child)
        if classified is None:
            continue
        category, reason = classified
        plans.append(
            ArchivePlan(
                old_path=child,
                new_path=archive_root / category / child.name,
                category=category,
                reason=reason,
            )
        )
    return plans


def write_manifest(path: Path, plans: list[ArchivePlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp_utc", "old_path", "new_path", "category", "reason"],
        )
        writer.writeheader()
        timestamp = datetime.now(timezone.utc).isoformat()
        for plan in plans:
            writer.writerow(
                {
                    "timestamp_utc": timestamp,
                    "old_path": str(plan.old_path),
                    "new_path": str(plan.new_path),
                    "category": plan.category,
                    "reason": plan.reason,
                }
            )


def execute(plans: list[ArchivePlan]) -> None:
    for plan in plans:
        if plan.new_path.exists():
            raise FileExistsError(f"Archive target already exists: {plan.new_path}")
        plan.new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(plan.old_path), str(plan.new_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive inactive outputs without touching active runs.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--manifest", default="outputs/_manifests/output_archive_manifest.csv")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    plans = build_plan(outputs_dir)
    write_manifest(Path(args.manifest), plans)
    for plan in plans:
        print(f"{plan.category}: {plan.old_path} -> {plan.new_path} ({plan.reason})")
    if args.apply:
        execute(plans)


if __name__ == "__main__":
    main()
