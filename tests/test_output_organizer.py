from pathlib import Path

from scripts.organize_outputs import ACTIVE_OUTPUTS, build_plan, execute, write_manifest


def test_output_organizer_archives_only_inactive_and_non_conventional_paths(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    active = outputs / next(iter(ACTIVE_OUTPUTS))
    active.mkdir()
    keep_dirs = ["dashboard_data", "figures", "vpp_first_person", "unclassified_run"]
    archive_dirs = [
        "test_training_smoke",
        "pytest_tmp_config",
        "tmp_arch_review",
        "paper_training_long_old",
        "paper_long_guardrail_smoke",
        "reward_v2_train_probe",
        "audit_reverseflow",
    ]
    for name in keep_dirs + archive_dirs:
        (outputs / name).mkdir()
    (outputs / "interactive_report.html").write_text("keep", encoding="utf-8")
    (outputs / "rl_architecture.html").write_text("keep", encoding="utf-8")
    (outputs / "current_training_report.md").write_text("archive", encoding="utf-8")

    plans = build_plan(outputs)
    planned_names = {plan.old_path.name for plan in plans}

    assert next(iter(ACTIVE_OUTPUTS)) not in planned_names
    assert not set(keep_dirs).intersection(planned_names)
    assert "interactive_report.html" not in planned_names
    assert "rl_architecture.html" not in planned_names
    assert set(archive_dirs).issubset(planned_names)
    assert "current_training_report.md" in planned_names


def test_output_organizer_manifest_and_execute_move_paths(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "test_training_smoke").mkdir()
    manifest = outputs / "_manifests" / "output_archive_manifest.csv"

    plans = build_plan(outputs)
    write_manifest(manifest, plans)
    execute(plans)

    assert manifest.exists()
    assert not (outputs / "test_training_smoke").exists()
    assert (outputs / "_archive" / "tests" / "test_training_smoke").exists()
    text = manifest.read_text(encoding="utf-8")
    assert "test_training_smoke" in text
    assert "_archive/tests/test_training_smoke" in text
