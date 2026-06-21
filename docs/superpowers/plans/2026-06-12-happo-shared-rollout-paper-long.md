# HAPPO Shared Rollout Paper-Long Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend paper-long HAPPO from independent shard training to an optional shared-weight synchronous parallel rollout mode while preserving the existing single-worker and shard workflows.

**Architecture:** Keep the existing `train_happo()` role-update semantics intact, extract rollout collection and HAPPO update into reusable units, then add a new serial/vectorized shared rollout path that gathers multiple worker fragments under one behavior-policy version before one centralized update. Keep multiprocessing out of the first implementation; correctness comes before wall-clock speed.

**Tech Stack:** Python 3.12, PyTorch, Gymnasium-style multi-agent environment, pandapower, pandas, pytest, existing paper-long entry `examples/17_paper_training_experiment.py`.

---

## 0. Execution Gate Audit

Current code evidence:

- `examples/17_paper_training_experiment.py` already supports `--train-episodes`, `--seeds`, and `--hparam-cases`, so short-run shard probe does not need a new launcher.
- `src/vpp_dso_sim/experiments/paper_training.py` maps paper-long HAPPO into `train_happo()` and already overrides `episodes`, `horizon_steps`, `gamma`, `hidden_dim`, learning rates, PPO epochs, entropy, and dispatch actor encoder type.
- `src/vpp_dso_sim/learning/advanced_marl.py::train_happo()` currently performs one environment rollout per episode, then updates critic, DSO actor, dispatch actors, and portfolio actors sequentially.
- `src/vpp_dso_sim/learning/deep_rl.py::_gae_returns_advantages()` currently assumes a finite rollout ending with zero bootstrap value. It is valid for full-episode training, but not valid for artificial fragment cuts in shared rollout.
- Current active tmux sessions `pl2_sa_*` are independent HAPPO shards. They train separate checkpoints, not one shared model.
- Current GPU memory is nearly saturated by 12 shard processes. A GPU smoke or new paper-long shared rollout should not be started until these sessions are stopped or finish the short probe.

Execution decision:

- Short-run shard probe summary: safe to implement now, low risk.
- Shared rollout implementation: safe to implement behind default-off config and CPU smoke tests; GPU smoke/paper-long launch should wait until current 12 shards are stopped or finish enough probe data.
- Existing training behavior must remain the default: `shared_rollout_workers=1`, `shared_rollout_enabled=False`.

## 1. File Responsibilities

- Modify `src/vpp_dso_sim/learning/deep_rl.py`
  - Add a bootstrap-aware GAE helper that supports true terminal masks and fragment-cut bootstrap values.
  - Keep existing `_gae_returns_advantages()` behavior unchanged for old tests.

- Modify `src/vpp_dso_sim/learning/advanced_marl.py`
  - Add default-off HAPPO config fields.
  - Introduce small internal rollout/update helpers without changing public output artifacts.
  - Add serial/vectorized shared rollout for multiple workers.
  - Record policy version, worker id, fragment metadata, ratio diagnostics, and bootstrap diagnostics.

- Modify `src/vpp_dso_sim/experiments/paper_training.py`
  - Thread shared-rollout config fields from paper-long experiment config into `HAPPOConfig`.
  - Preserve existing shard behavior.

- Modify `examples/17_paper_training_experiment.py`
  - Add CLI options for shared rollout workers, fragment length, and backend.
  - Keep options default-off.

- Create `scripts/summarize_happo_probe.py`
  - Summarize existing short shard probe outputs and logs.
  - Avoid changing training code just to support metrics that are not logged.

- Add tests:
  - `tests/test_happo_shared_rollout.py`
  - Extend `tests/test_deep_rl_training.py`
  - Extend `tests/test_paper_training_experiment.py`

## 2. Task 1: Bootstrap-Aware GAE Helper

**Files:**
- Modify: `src/vpp_dso_sim/learning/deep_rl.py`
- Test: `tests/test_deep_rl_training.py`

- [ ] **Step 1: Write failing tests for terminal and fragment bootstrap**

Add tests that call a new helper named `_gae_returns_advantages_bootstrap()`:

```python
@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_gae_bootstrap_fragment_cut_uses_next_value():
    import torch
    from vpp_dso_sim.learning.deep_rl import _gae_returns_advantages_bootstrap

    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    returns, advantages = _gae_returns_advantages_bootstrap(
        rewards=[0.0, 0.0],
        values=values,
        next_value=torch.tensor(10.0),
        terminals=[False, False],
        gamma=0.5,
        gae_lambda=1.0,
        torch=torch,
    )

    assert torch.allclose(returns, torch.tensor([2.5, 5.0]), atol=1e-6)
    assert torch.allclose(advantages, returns - values, atol=1e-6)


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_gae_bootstrap_true_terminal_ignores_next_value():
    import torch
    from vpp_dso_sim.learning.deep_rl import _gae_returns_advantages_bootstrap

    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    returns, advantages = _gae_returns_advantages_bootstrap(
        rewards=[0.0, 0.0],
        values=values,
        next_value=torch.tensor(10.0),
        terminals=[False, True],
        gamma=0.5,
        gae_lambda=1.0,
        torch=torch,
    )

    assert torch.allclose(returns, torch.tensor([0.0, 0.0]), atol=1e-6)
    assert torch.allclose(advantages, returns - values, atol=1e-6)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_deep_rl_training.py::test_gae_bootstrap_fragment_cut_uses_next_value tests/test_deep_rl_training.py::test_gae_bootstrap_true_terminal_ignores_next_value -q
```

Expected: FAIL because `_gae_returns_advantages_bootstrap` does not exist.

- [ ] **Step 3: Implement helper without changing old helper**

Add below `_gae_returns_advantages()`:

```python
def _gae_returns_advantages_bootstrap(
    *,
    rewards: list[float],
    values: Any,
    next_value: Any,
    terminals: list[bool],
    gamma: float,
    gae_lambda: float,
    torch: Any,
) -> tuple[Any, Any]:
    """Compute GAE for a rollout fragment with explicit terminal masks.

    ``terminals[index]`` means the transition after ``index`` is a true
    environment terminal and must not bootstrap. A fragment cut uses
    ``terminal=False`` on the final transition and bootstraps from
    ``next_value``.
    """

    if len(rewards) != len(terminals):
        raise ValueError("rewards and terminals must have the same length")
    device = values.device if hasattr(values, "device") else None
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    terminal_tensor = torch.tensor(terminals, dtype=torch.bool, device=device)
    detached_values = values.detach()
    detached_next_value = torch.as_tensor(next_value, dtype=torch.float32, device=device).detach()
    advantages = torch.zeros_like(reward_tensor)
    running_advantage = torch.tensor(0.0, dtype=torch.float32, device=device)
    next_value_t = detached_next_value.reshape(())
    for index in range(len(rewards) - 1, -1, -1):
        non_terminal = (~terminal_tensor[index]).to(dtype=torch.float32)
        delta = reward_tensor[index] + float(gamma) * next_value_t * non_terminal - detached_values[index]
        running_advantage = delta + float(gamma) * float(gae_lambda) * non_terminal * running_advantage
        advantages[index] = running_advantage
        next_value_t = detached_values[index]
    returns = advantages + detached_values
    return returns, advantages
```

- [ ] **Step 4: Run old and new GAE tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_deep_rl_training.py::test_gae_and_ppo_clip_helpers_are_finite_and_clipped tests/test_deep_rl_training.py::test_gae_bootstrap_fragment_cut_uses_next_value tests/test_deep_rl_training.py::test_gae_bootstrap_true_terminal_ignores_next_value -q
```

Expected: PASS.

## 3. Task 2: Short-Run HAPPO Probe Summary

**Files:**
- Create: `scripts/summarize_happo_probe.py`
- Test: `tests/test_happo_probe_summary.py`

- [ ] **Step 1: Write tests for summary extraction**

Create a temporary directory with two fake shard runs:

```python
def test_summarize_happo_probe_reads_episode_and_update_metrics(tmp_path):
    import pandas as pd
    from scripts.summarize_happo_probe import summarize_probe_root

    run = tmp_path / "seed_9401_base" / "runs" / "happo_base_train_mixed_seed_9401" / "train"
    run.mkdir(parents=True)
    pd.DataFrame(
        [
            {"episode": 0, "episode_reward": -10.0, "total_cost": 100.0, "violation_count": 0, "projection_gap_mw": 0.0},
            {"episode": 1, "episode_reward": -8.0, "total_cost": 95.0, "violation_count": 0, "projection_gap_mw": 0.0},
        ]
    ).to_csv(run / "happo_episode_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"episode": 1, "role": "dso_global_guidance", "policy_loss": -1.0, "entropy_mean": 2.0, "approx_kl": 0.01},
            {"episode": 1, "role": "vpp_1_dispatch", "policy_loss": -0.2, "entropy_mean": 1.5, "approx_kl": 0.02},
        ]
    ).to_csv(run / "happo_update_metrics.csv", index=False)

    summary = summarize_probe_root(tmp_path)

    assert len(summary) == 1
    assert summary.iloc[0]["seed"] == 9401
    assert summary.iloc[0]["hparam_case"] == "base"
    assert summary.iloc[0]["final_reward"] == -8.0
    assert summary.iloc[0]["mean_reward"] == -9.0
    assert summary.iloc[0]["completed"] is True
    assert summary.iloc[0]["nan_or_inf"] is False
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_happo_probe_summary.py -q
```

Expected: FAIL because script does not exist.

- [ ] **Step 3: Implement summary script**

Implement `summarize_probe_root(root: Path) -> pandas.DataFrame` that:

- Finds `*/runs/*/train/happo_episode_metrics.csv`.
- Parses `seed` and `hparam_case` from parent directory names like `seed_9401_base` and run ids like `happo_base_train_mixed_seed_9401`.
- Reads optional `happo_update_metrics.csv`.
- Computes final reward, mean reward, reward std, best reward, final cost, total violations, projection gap sum, mean critic loss if present, mean actor policy loss if present, mean entropy, mean approx KL.
- Marks `completed=True` when at least one episode row exists.
- Marks `nan_or_inf=True` if any numeric metric contains NaN or Inf.
- Writes `happo_probe_summary.csv` and `happo_probe_summary.json` when used from CLI.

- [ ] **Step 4: Run test and a real probe summary**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_happo_probe_summary.py -q
./.venv-server/bin/python scripts/summarize_happo_probe.py outputs/paper_training_long_dispatch_set_attention_v1_happo_20260612_sharded_v2 --output-dir outputs/paper_training_long_dispatch_set_attention_v1_happo_20260612_sharded_v2/probe_summary
```

Expected: test PASS. Real summary may be partial while current shard runs have not flushed `happo_episode_metrics.csv`; in that case the script should report zero completed metric files and not crash.

## 4. Task 3: HAPPO Config and Paper-Long CLI Wiring

**Files:**
- Modify: `src/vpp_dso_sim/learning/advanced_marl.py`
- Modify: `src/vpp_dso_sim/experiments/paper_training.py`
- Modify: `examples/17_paper_training_experiment.py`
- Test: `tests/test_paper_training_experiment.py`

- [ ] **Step 1: Write config-default tests**

Add a test that asserts paper-long defaults preserve old behavior:

```python
def test_happo_shared_rollout_defaults_are_off():
    from vpp_dso_sim.learning.advanced_marl import HAPPOConfig

    cfg = HAPPOConfig()
    assert cfg.shared_rollout_enabled is False
    assert cfg.shared_rollout_workers == 1
    assert cfg.shared_rollout_backend == "serial"
    assert cfg.rollout_fragment_steps is None
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_paper_training_experiment.py::test_happo_shared_rollout_defaults_are_off -q
```

Expected: FAIL because fields do not exist.

- [ ] **Step 3: Add HAPPOConfig fields**

Add fields to `HAPPOConfig`:

```python
shared_rollout_enabled: bool = False
shared_rollout_workers: int = 1
shared_rollout_backend: str = "serial"
rollout_fragment_steps: int | None = None
rollout_policy_version_check: bool = True
```

- [ ] **Step 4: Add PaperTrainingExperimentConfig fields**

Add fields:

```python
happo_shared_rollout_enabled: bool = False
happo_shared_rollout_workers: int = 1
happo_shared_rollout_backend: str = "serial"
happo_rollout_fragment_steps: int | None = None
```

Thread them into the `replace(happo_config, ...)` call for HAPPO.

- [ ] **Step 5: Add CLI options**

Add args to `examples/17_paper_training_experiment.py`:

```python
parser.add_argument("--happo-shared-rollout", action="store_true")
parser.add_argument("--happo-shared-rollout-workers", type=int, default=None)
parser.add_argument("--happo-rollout-fragment-steps", type=int, default=None)
parser.add_argument("--happo-shared-rollout-backend", choices=["serial"], default=None)
```

Map them to the new config fields.

- [ ] **Step 6: Run config tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_paper_training_experiment.py::test_happo_shared_rollout_defaults_are_off -q
```

Expected: PASS.

## 5. Task 4: Extract Rollout and Update Helpers Without Behavior Change

**Files:**
- Modify: `src/vpp_dso_sim/learning/advanced_marl.py`
- Test: `tests/test_hasac_happo.py`

- [ ] **Step 1: Add a worker=1 regression test**

Add a CPU smoke test:

```python
@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_single_worker_shared_rollout_disabled_preserves_artifacts(tmp_path):
    result = train_happo(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=tmp_path / "happo_single_worker_regression",
        config=HAPPOConfig(
            episodes=1,
            horizon_steps=2,
            hidden_dim=16,
            ppo_epochs=1,
            seed=52,
            device="cpu",
            shared_rollout_enabled=False,
            shared_rollout_workers=1,
        ),
    )

    summary = result["summary"]
    assert summary["algorithm"] == "happo_sequential_ctde"
    assert summary["shared_rollout_enabled"] is False
    assert summary["shared_rollout_workers"] == 1
    assert not result["episode_metrics"].empty
    assert not result["update_metrics"].empty
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_hasac_happo.py::test_happo_single_worker_shared_rollout_disabled_preserves_artifacts -q
```

Expected: FAIL until config fields and summary fields exist.

- [ ] **Step 3: Add summary fields only**

Add to HAPPO summary:

```python
"shared_rollout_enabled": bool(cfg.shared_rollout_enabled),
"shared_rollout_workers": int(cfg.shared_rollout_workers),
"shared_rollout_backend": str(cfg.shared_rollout_backend),
"rollout_fragment_steps": cfg.rollout_fragment_steps,
```

- [ ] **Step 4: Refactor carefully**

Extract only internal functions first:

- `_new_happo_rollout_buffer()`
- `_collect_happo_rollout_single_env(...)`
- `_update_happo_from_rollout(...)`

The first refactor must preserve the old tensor shapes:

- `critic_state`: `[time, state_dim]`
- `action_summary`: `[time, action_dim]`
- `dso_obs`: `[time, dso_obs_dim]`
- `dispatch_obs`: `[time, vpp_count, dispatch_obs_dim]`
- `portfolio_obs`: `[time, vpp_count, portfolio_obs_dim]`
- `rewards`: `[time, head_count]`

- [ ] **Step 5: Run existing HAPPO tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_hasac_happo.py::test_happo_training_writes_sequential_update_artifacts tests/test_structured_happo_training.py::test_happo_uses_structured_dso_actor_when_config_requests_it -q
```

Expected: PASS.

## 6. Task 5: Serial Shared-Weight Multi-Worker Rollout

**Files:**
- Modify: `src/vpp_dso_sim/learning/advanced_marl.py`
- Test: `tests/test_happo_shared_rollout.py`

- [ ] **Step 1: Write tests for policy version and worker dimensions**

Create tests:

```python
@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_shared_rollout_multi_worker_completes_one_update(tmp_path):
    result = train_happo(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=tmp_path / "shared_rollout",
        config=HAPPOConfig(
            episodes=1,
            horizon_steps=4,
            hidden_dim=16,
            ppo_epochs=1,
            seed=61,
            device="cpu",
            shared_rollout_enabled=True,
            shared_rollout_workers=2,
            shared_rollout_backend="serial",
            rollout_fragment_steps=2,
        ),
    )

    summary = result["summary"]
    assert summary["shared_rollout_enabled"] is True
    assert summary["shared_rollout_workers"] == 2
    assert summary["shared_rollout_policy_version_mismatch_count"] == 0
    assert summary["shared_rollout_batches"] >= 1
    assert not result["update_metrics"].empty
    assert "policy_version" in result["update_metrics"].columns
    assert "worker_count" in result["update_metrics"].columns
    assert "ratio_mean" in result["update_metrics"].columns
    assert result["update_metrics"]["ratio_mean"].notna().all()
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_happo_shared_rollout.py::test_happo_shared_rollout_multi_worker_completes_one_update -q
```

Expected: FAIL until shared rollout path exists.

- [ ] **Step 3: Implement shared rollout semantics**

Implement default serial backend:

- For each training update/episode index, define `policy_version = f"{cfg.algorithm}:episode={episode}:update={episode}"`.
- For each worker `worker_index in range(cfg.shared_rollout_workers)`, create a separate `MultiAgentVPPDSOEnv` with seed `cfg.seed + episode * 1000 + worker_index`.
- Collect `rollout_fragment_steps` transitions, or `horizon_steps` if `rollout_fragment_steps is None`.
- Workers must not optimizer-step.
- Save old log probabilities during sampling.
- Save `worker_index`, local step, global step, terminal mask, and policy version per row.
- Compute per-worker next critic value at fragment end when not terminal.
- Compute GAE per worker/head with `_gae_returns_advantages_bootstrap()`.
- Flatten only after GAE, giving `[worker*time, ...]` tensors to the existing HAPPO sequential update.

- [ ] **Step 4: Preserve HAPPO sequential role update**

Do not change the role order:

1. centralized value critic
2. DSO actor
3. dispatch actors
4. portfolio actors

When flattening shared rollout:

- DSO correction shape becomes `[worker*time]`.
- Dispatch tensors become `[worker*time, vpp_count, ...]`.
- Portfolio masks remain `[worker*time, vpp_count]`.
- Centralized critic uses each sample's own `critic_state` and `action_summary`.

- [ ] **Step 5: Add diagnostics**

Add to `update_rows`:

```python
"policy_version": policy_version,
"worker_count": int(cfg.shared_rollout_workers),
"rollout_fragment_steps": int(fragment_steps),
"bootstrap_value_mean": bootstrap_value_mean,
"terminal_transition_count": terminal_count,
"fragment_cut_count": fragment_cut_count,
```

Add summary counters:

```python
"shared_rollout_batches": shared_rollout_batches,
"shared_rollout_policy_version_mismatch_count": policy_version_mismatch_count,
"shared_rollout_backend": cfg.shared_rollout_backend,
```

- [ ] **Step 6: Run shared rollout tests**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_happo_shared_rollout.py -q
```

Expected: PASS.

## 7. Task 6: Paper-Long Shared Rollout Smoke

**Files:**
- Modify only if failures expose wiring defects:
  - `src/vpp_dso_sim/experiments/paper_training.py`
  - `examples/17_paper_training_experiment.py`
- Test: command smoke

- [ ] **Step 1: Run CPU smoke that does not require GPU**

Run a direct trainer smoke:

```bash
./.venv-server/bin/python -m pytest tests/test_happo_shared_rollout.py tests/test_hasac_happo.py::test_happo_training_writes_sequential_update_artifacts -q
```

Expected: PASS.

- [ ] **Step 2: Do not start GPU paper-long while current 12 shards saturate GPU**

Before GPU smoke:

```bash
nvidia-smi
tmux list-sessions
```

Expected gate: free GPU memory should be enough for at least one HAPPO shared rollout process. If 12 `pl2_sa_*` sessions are still active, either wait for the probe stop point or stop them intentionally.

- [ ] **Step 3: Launch a minimal paper-long-style shared rollout smoke**

Only after GPU is available:

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --config-path configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml \
  --output-dir outputs/paper_long_shared_rollout_smoke_20260612 \
  --algorithms happo \
  --seeds 9401 \
  --hparam-cases base \
  --train-episodes 1 \
  --horizon-steps 96 \
  --eval-horizon-steps 96 \
  --checkpoint-selection final \
  --happo-shared-rollout \
  --happo-shared-rollout-workers 4 \
  --happo-rollout-fragment-steps 96 \
  --progress-interval-seconds 60
```

Expected: one HAPPO run completes, writes `happo_update_metrics.csv`, and summary reports `shared_rollout_enabled=true`.

## 8. Task 7: Main Paper-Long Adjustment

**Files:**
- No source modification if Tasks 1-6 pass.
- Runtime output under `outputs/`.

- [ ] **Step 1: Stop or archive independent probe shards after 1-3 completed episodes**

Use probe summary to choose the main hparam case. Do not keep 12 long independent runs consuming GPU once shared rollout is ready.

- [ ] **Step 2: Start one main shared-weight HAPPO model**

Recommended first main run:

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --config-path configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml \
  --output-dir outputs/paper_training_long_shared_rollout_happo_20260612_main \
  --algorithms happo \
  --seeds 9401 \
  --hparam-cases base \
  --checkpoint-selection both \
  --happo-shared-rollout \
  --happo-shared-rollout-workers 4 \
  --happo-rollout-fragment-steps 96 \
  --progress-interval-seconds 60
```

- [ ] **Step 3: Monitor metrics**

Required checks:

- `happo_update_metrics.csv`: ratio mean near 1 at update start, finite KL, finite grad norm.
- `happo_episode_metrics.csv`: no NaN/Inf, reward not collapsing, safety violation count stays auditable.
- reward dynamic cards: per-agent type reward shares still generated.
- `nvidia-smi`: no OOM, GPU memory stable.

## 9. Self-Review

Spec coverage:

- Short-run shard probe: covered by Task 2 and Task 7 Step 1.
- Shared-weight synchronous rollout: covered by Tasks 1, 4, 5, and 6.
- Existing behavior default-off: covered by Tasks 3 and 4.
- Correct old log probability: preserved in Task 5 Step 3.
- Worker/time/agent dimensions: preserved in Task 5 Step 3 and Step 4.
- Fragment bootstrap vs true terminal: covered by Task 1.
- HAPPO sequential role update: preserved in Task 5 Step 4.
- Policy version check: covered by Task 5.
- Paper-long command adjustment: covered by Task 7.

Execution risks:

- The current GPU is nearly saturated by independent shards; do not launch the new GPU smoke until those are stopped or finish the probe checkpoint.
- Refactoring `train_happo()` is high risk because the function is large and currently writes many downstream reports. Implement helper extraction and shared rollout in small tested steps.
- Multiprocessing is intentionally excluded from v1. Serial/vectorized workers give algorithmic correctness first; multiprocessing can be a later optimization.
