# Repository Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize configs, documents, root generated files, and inactive outputs while preserving old paths and fixing config path resolution.

**Architecture:** Keep stable compatibility wrappers at `configs/*.yaml`, move real config bodies into canonical versioned directories, and add a registry-aware YAML resolver. Archive inactive runtime outputs under `outputs/_archive/` with a manifest, but leave active and conventional dashboard outputs in place.

**Tech Stack:** Python 3.12, PyYAML, pytest, pandapower project code under `src/vpp_dso_sim`, shell moves for large ignored output folders.

---

### Task 1: Config Resolver Tests

**Files:**
- Create: `tests/test_config_path_registry.py`
- Modify: `src/vpp_dso_sim/utils/config.py`

- [ ] **Step 1: Write failing tests for legacy, canonical, alias, and nested extends paths**

```python
from pathlib import Path

import pytest

from vpp_dso_sim.utils.config import load_yaml, resolve_config_path


def test_legacy_and_canonical_demo_config_load_same_network_type():
    legacy = load_yaml("configs/ieee33_multi_vpp.yaml")
    canonical = load_yaml("configs/scenarios/demo/ieee33_multi_vpp.yaml")

    assert legacy["network"]["type"] == "ieee33"
    assert canonical["network"]["type"] == "ieee33"
    assert legacy["network"] == canonical["network"]


def test_registry_alias_loads_config():
    cfg = load_yaml("happo_sensitivity_attention_v1")

    assert cfg["name"] == "happo_sensitivity_attention_v1"
    assert cfg["dso"]["envelope_policy"] == "sensitivity_attention_v1"


def test_nested_extends_resolves_from_canonical_config():
    cfg = load_yaml(
        "configs/experiments/paper_long/sensitivity_attention_v1/"
        "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml"
    )

    assert cfg["network"]["type"] == "european_lv_benchmark_v2"
    assert cfg["reward"]["version"] == "v2_minimal"
    assert cfg["dso"]["envelope_policy"] == "sensitivity_attention_v1"


def test_resolve_config_path_reports_alias_target():
    resolved = resolve_config_path("reward_v2_minimal")

    assert resolved == Path("configs/rewards/v2_minimal/reward_v2_minimal.yaml").resolve()


def test_unknown_config_path_error_mentions_request():
    with pytest.raises(FileNotFoundError, match="does_not_exist_config"):
        load_yaml("does_not_exist_config")
```

- [ ] **Step 2: Run test to verify it fails before implementation**

Run:

```bash
python -m pytest tests/test_config_path_registry.py -q --basetemp=outputs/pytest_tmp_repo_org_config_fail -o cache_dir=outputs/pytest_cache_repo_org_config_fail
```

Expected: fail because canonical directories, registry, and `resolve_config_path` do not exist yet.

- [ ] **Step 3: Implement config path resolver**

Add to `src/vpp_dso_sim/utils/config.py`:

```python
def config_registry_path() -> Path:
    return project_root() / "configs" / "registry.yaml"


def _load_config_registry() -> dict[str, str]:
    path = config_registry_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    aliases = data.get("aliases", data)
    if not isinstance(aliases, dict):
        raise ValueError(f"Config registry aliases must be a mapping: {path}")
    return {str(key): str(value) for key, value in aliases.items()}


def resolve_config_path(path: str | Path, *, child_path: Path | None = None) -> Path:
    requested = Path(path)
    attempts: list[Path] = []
    if requested.is_absolute():
        if requested.exists():
            return requested.resolve()
        raise FileNotFoundError(f"YAML file not found: {path}")

    candidates: list[Path] = []
    if child_path is not None:
        candidates.append(child_path.parent / requested)
    candidates.append(project_root() / requested)

    registry = _load_config_registry()
    alias = registry.get(str(path))
    if alias is not None:
        alias_path = Path(alias)
        candidates.append(alias_path if alias_path.is_absolute() else project_root() / alias_path)

    for candidate in candidates:
        attempts.append(candidate)
        if candidate.exists():
            return candidate.resolve()

    resource = files("vpp_dso_sim.resources.configs").joinpath(requested.name)
    if resource.is_file():
        return Path(str(resource))

    attempted = ", ".join(str(item) for item in attempts)
    raise FileNotFoundError(f"YAML file not found: {path}; attempted: {attempted}")
```

Then update `_resolve_extends_path()` and `load_yaml()` to call `resolve_config_path()` instead of duplicating path logic.

- [ ] **Step 4: Run config path tests**

Run:

```bash
python -m pytest tests/test_config_path_registry.py -q --basetemp=outputs/pytest_tmp_repo_org_config -o cache_dir=outputs/pytest_cache_repo_org_config
```

Expected: pass.

### Task 2: Config Directory Reorganization

**Files:**
- Create directories under `configs/scenarios/`, `configs/algorithms/`, `configs/rewards/`, and `configs/experiments/`
- Move config bodies from `configs/*.yaml`
- Create: `configs/registry.yaml`
- Create: `configs/README.md`
- Modify: root `configs/*.yaml` compatibility wrappers

- [ ] **Step 1: Create canonical directories**

Run:

```bash
mkdir -p configs/scenarios/demo configs/scenarios/benchmark configs/algorithms/dso_sensitivity_attention/v1 configs/rewards/v2_minimal configs/experiments/paper_long/sensitivity_attention_v1
```

Expected: directories exist.

- [ ] **Step 2: Move config bodies into canonical paths without overwriting**

Run:

```bash
mv -n configs/default_profiles.yaml configs/scenarios/demo/default_profiles.yaml
mv -n configs/ieee33_multi_vpp.yaml configs/scenarios/demo/ieee33_multi_vpp.yaml
mv -n configs/lv_taiqu_demo.yaml configs/scenarios/demo/lv_taiqu_demo.yaml
mv -n configs/european_lv_mixed_vpp.yaml configs/scenarios/demo/european_lv_mixed_vpp.yaml
mv -n configs/european_lv_benchmark_v2.yaml configs/scenarios/benchmark/european_lv_benchmark_v2.yaml
mv -n configs/european_lv_benchmark_v2_safety_tight.yaml configs/scenarios/benchmark/european_lv_benchmark_v2_safety_tight.yaml
mv -n configs/baseline_rule_v0.yaml configs/algorithms/dso_sensitivity_attention/v1/baseline_rule_v0.yaml
mv -n configs/happo_legacy_mlp.yaml configs/algorithms/dso_sensitivity_attention/v1/happo_legacy_mlp.yaml
mv -n configs/happo_sensitivity_attention_v1.yaml configs/algorithms/dso_sensitivity_attention/v1/happo_sensitivity_attention_v1.yaml
mv -n configs/ablation_no_sensitivity_edges.yaml configs/algorithms/dso_sensitivity_attention/v1/ablation_no_sensitivity_edges.yaml
mv -n configs/ablation_no_action_self_attention.yaml configs/algorithms/dso_sensitivity_attention/v1/ablation_no_action_self_attention.yaml
mv -n configs/ablation_no_width_penalty.yaml configs/algorithms/dso_sensitivity_attention/v1/ablation_no_width_penalty.yaml
mv -n configs/reward_v2_minimal.yaml configs/rewards/v2_minimal/reward_v2_minimal.yaml
mv -n configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1.yaml
mv -n configs/european_lv_benchmark_v2_sensitivity_attention_v1_legacy_v1_reward.yaml configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_legacy_v1_reward.yaml
mv -n configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml
mv -n configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_5.yaml configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_5.yaml
mv -n configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_10.yaml configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_10.yaml
mv -n configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_20.yaml configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_contract_delivery_weight_20.yaml
mv -n configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_no_portfolio_window_penalty.yaml configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_no_portfolio_window_penalty.yaml
mv -n configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_no_shield_eval.yaml configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_no_shield_eval.yaml
mv -n configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_preferred_bonus_0p05.yaml configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal_preferred_bonus_0p05.yaml
```

Expected: canonical files exist; no active output files are touched.

- [ ] **Step 3: Add root compatibility wrappers**

For every moved file, create a root wrapper such as:

```yaml
# Compatibility wrapper. Canonical file:
# configs/scenarios/demo/ieee33_multi_vpp.yaml
extends: configs/scenarios/demo/ieee33_multi_vpp.yaml
```

Expected: old paths remain loadable.

- [ ] **Step 4: Add registry and README**

`configs/registry.yaml` must contain aliases like:

```yaml
aliases:
  ieee33_multi_vpp: configs/scenarios/demo/ieee33_multi_vpp.yaml
  lv_taiqu_demo: configs/scenarios/demo/lv_taiqu_demo.yaml
  european_lv_mixed_vpp: configs/scenarios/demo/european_lv_mixed_vpp.yaml
  european_lv_benchmark_v2: configs/scenarios/benchmark/european_lv_benchmark_v2.yaml
  happo_sensitivity_attention_v1: configs/algorithms/dso_sensitivity_attention/v1/happo_sensitivity_attention_v1.yaml
  reward_v2_minimal: configs/rewards/v2_minimal/reward_v2_minimal.yaml
  paper_long_sensitivity_v1_reward_v2_minimal: configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml
```

`configs/README.md` must explain canonical paths, wrappers, and alias names.

- [ ] **Step 5: Verify config reorganization**

Run:

```bash
python -m pytest tests/test_config_path_registry.py tests/test_structured_happo_training.py tests/test_paper_training_experiment.py -q --basetemp=outputs/pytest_tmp_repo_org_configs -o cache_dir=outputs/pytest_cache_repo_org_configs
```

Expected: pass.

### Task 3: Root Generated File Cleanup

**Files:**
- Move root `paper_method_experiment_cn.*`
- Move `task_0.md` and `task2.md`
- Modify: `.gitignore`

- [ ] **Step 1: Create document directories**

Run:

```bash
mkdir -p docs/reports/paper_method_experiment_cn docs/tasks
```

Expected: directories exist.

- [ ] **Step 2: Move root generated and task files**

Run:

```bash
mv -n paper_method_experiment_cn.* docs/reports/paper_method_experiment_cn/
mv -n task_0.md docs/tasks/reward_v2_minimal_task.md
mv -n task2.md docs/tasks/current_training_diagnosis_task.md
```

Expected: project root no longer contains these generated/ad hoc files.

- [ ] **Step 3: Add LaTeX generated file ignores**

Append to `.gitignore`:

```gitignore
# LaTeX build byproducts
*.aux
*.fdb_latexmk
*.fls
*.xdv
*.synctex.gz
*.toc
*.bbl
*.blg
```

- [ ] **Step 4: Check status**

Run:

```bash
git status --short
```

Expected: moved report/task files appear under `docs/`, root generated files no longer appear at repository root.

### Task 4: Inactive Output Archive Manifest

**Files:**
- Create: `scripts/organize_outputs.py`
- Create: `outputs/_manifests/output_archive_manifest.csv`
- Create: `outputs/_manifests/README.md`

- [ ] **Step 1: Implement dry-run output organizer**

Create `scripts/organize_outputs.py` with:

```python
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

CONVENTIONAL_KEEP = {
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
    if name in CONVENTIONAL_KEEP:
        return None
    if path.is_file() and name in CONVENTIONAL_KEEP_FILES:
        return None
    if name.startswith("test_") or name.startswith("pytest_tmp_") or name.startswith("tmp_"):
        return ("tests", "test or temporary output")
    if name.startswith("paper_training_long") or name.startswith("paper_long_") or name.startswith("reward_v2_"):
        return ("paper_long", "paper-long or reward experiment output")
    if name.startswith("audit_") or name.endswith("_audit") or "audit" in name:
        return ("audits", "audit or preflight output")
    if path.is_file() and (name.endswith(".md") or name.endswith(".html") or name.endswith(".pdf")):
        return ("reports", "root generated report")
    return None


def build_plan(outputs_dir: Path) -> list[ArchivePlan]:
    archive_root = outputs_dir / "_archive"
    plans: list[ArchivePlan] = []
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
```

- [ ] **Step 2: Run dry-run manifest**

Run:

```bash
python scripts/organize_outputs.py --outputs-dir outputs --manifest outputs/_manifests/output_archive_manifest.csv
```

Expected: manifest is written; no files are moved.

- [ ] **Step 3: Apply archive only after reviewing manifest**

Run:

```bash
python scripts/organize_outputs.py --outputs-dir outputs --manifest outputs/_manifests/output_archive_manifest.csv --apply
```

Expected: inactive classified outputs move under `outputs/_archive/`; active output directory remains at root.

### Task 5: Documentation and Context Update

**Files:**
- Modify: `README.md`
- Modify: `CODEX_PROJECT_CONTEXT.md`
- Modify: `docs/agents/repo_map.md`

- [ ] **Step 1: Update README layout section**

Add a section explaining:

```markdown
## Repository Layout and Path Compatibility

Canonical configs now live under `configs/scenarios/`, `configs/algorithms/`,
`configs/rewards/`, and `configs/experiments/`. Root-level `configs/*.yaml`
files are compatibility wrappers so existing commands and old experiment
manifests remain readable.

Use `configs/registry.yaml` aliases for new scripts when possible.
```

- [ ] **Step 2: Update CODEX context**

Record active output freeze and canonical config layout.

- [ ] **Step 3: Update repo map**

Record where configs, output archives, generated reports, and ad hoc task docs live.

### Task 6: Verification

**Files:**
- No new files unless tests produce ignored outputs.

- [ ] **Step 1: Run focused config tests**

Run:

```bash
python -m pytest tests/test_config_path_registry.py -q --basetemp=outputs/pytest_tmp_repo_org_verify_config -o cache_dir=outputs/pytest_cache_repo_org_verify_config
```

Expected: pass.

- [ ] **Step 2: Run scenario load smoke**

Run:

```bash
./.venv-server/bin/python -c "from vpp_dso_sim.simulation.scenario import load_scenario; names=['configs/ieee33_multi_vpp.yaml','configs/scenarios/demo/ieee33_multi_vpp.yaml','happo_sensitivity_attention_v1']; print([len(load_scenario(n).vpps) for n in names])"
```

Expected: prints three VPP-count integers without exception.

- [ ] **Step 3: Run regression subset**

Run:

```bash
python -m pytest tests/test_env_smoke.py tests/test_timeseries_smoke.py tests/test_structured_happo_training.py tests/test_paper_training_experiment.py -q --basetemp=outputs/pytest_tmp_repo_org_regression -o cache_dir=outputs/pytest_cache_repo_org_regression
```

Expected: pass or report pre-existing failures with exact failing tests.

- [ ] **Step 4: Check active output stayed in place**

Run:

```bash
test -d outputs/paper_training_long_reward_v2_minimal_20260604_gpu_decoder_bounds_happo_hatrpo_logfix
```

Expected: exit code 0.

- [ ] **Step 5: Summarize final layout**

Run:

```bash
find configs -maxdepth 4 -type f | sort
find docs/reports docs/tasks -maxdepth 3 -type f | sort
find outputs/_archive -maxdepth 2 -type d | sort
```

Expected: config canonical directories, report/task docs, and output archive categories are visible.
