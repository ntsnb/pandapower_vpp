# Dispatch Actor Set-Attention Upgrade Record

Date: 2026-06-12

## Scope

This change strengthens the VPP dispatch actor for paper-long MARL experiments while preserving the existing physical decoder, reward design, DOE/AC safety shield, and action payload contract.

## Implemented

- Added optional dispatch actor architecture `set_attention_v1`.
- Preserved default legacy architecture `deepset_v1` for existing tests, old configs, and compatibility runs.
- Replaced mean/max-only DER pooling with mask-aware explicit Q/K/V self-attention over DER tokens.
- Kept output contract unchanged:
  - aggregate normalized action: shape `[batch, 1]`
  - DER normalized actions: shape `[batch, max_der_per_vpp]`
  - Gaussian log std tensors unchanged.
- Made DER action heads token-level for `set_attention_v1`, so each DER action is produced from that DER's attended token latent instead of a single global latent.
- Avoided `torch.nn.MultiheadAttention` because HATRPO requires second-order/Fisher-vector-product gradients and CPU flash attention did not support the needed derivative path.
- Wired `dispatch_actor_encoder_type` through HAPPO, HASAC, MATD3, HATRPO, early CTDE utilities, checkpoint loading, and paper-training config.
- Added paper-long sensitivity preset default:
  - `dispatch_actor_encoder_type: set_attention_v1`
  - `require_cuda_for_trainable: true`

## Verification

Fresh targeted verification passed:

```bash
./.venv-server/bin/python -m pytest \
  tests/test_dispatch_actor_set_attention.py \
  tests/test_paper_training_experiment.py::test_paper_long_sensitivity_v1_preset_uses_structured_happo_config \
  tests/test_paper_training_experiment.py::test_trainable_cuda_guard_blocks_paper_long_cpu_fallback \
  tests/test_paper_training_experiment.py::test_trainable_cuda_guard_allows_smoke_cpu_fallback \
  tests/test_hasac_happo.py::test_happo_training_writes_sequential_update_artifacts \
  tests/test_hasac_happo.py::test_hasac_training_records_resolved_device_and_reward_artifacts \
  tests/test_matd3_training.py::test_matd3_training_writes_full_off_policy_artifacts \
  tests/test_hatrpo_training.py::test_hatrpo_training_runs_against_multi_agent_ctde_env \
  -q
```

Result: `9 passed`.

Additional 2-step smoke runs confirmed `set_attention_v1` for HAPPO, HATRPO, MATD3, and HASAC.

## Active Paper-Long Run

Started tmux session:

```bash
tmux new-session -d -s paper_long_dispatch_set_attention_v1_20260612
```

Output directory:

```text
outputs/paper_training_long_dispatch_set_attention_v1_happo_20260612_full
```

Command:

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --config-path configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml \
  --output-dir outputs/paper_training_long_dispatch_set_attention_v1_happo_20260612_full \
  --algorithms happo \
  --checkpoint-selection both \
  --progress-interval-seconds 60
```

Matrix:

- algorithm: HAPPO
- seeds: 9401, 9402, 9403, 9404, 9405
- hparam cases: base, lower_lr, higher_entropy, larger_network
- train horizon: 672 steps
- train episodes: 120
- eval scenarios: holdout_peak, holdout_cloudy, holdout_reverseflow
- checkpoint evaluation: best and final

Initial process evidence:

- tmux session exists.
- process command is running.
- `nvidia-smi` showed project Python process using GPU memory.
- `experiment_progress.csv` recorded campaign start and first train_start row.

## Live Status Snapshot

Checked at 2026-06-12 14:25 Asia/Shanghai:

- Active run: `happo_base_train_mixed_seed_9401`
- Active hparam case: `base`
- Active episode: 1 / 120
- Latest recorded step: 72 / 672
- Latest reward so far: -81.32013721411332
- Latest total cost so far: 3378.4145048618866
- Violations so far: 0
- Projection gap MW: 0.0
- Process PID: 841372
- Process state: running
- GPU evidence: `nvidia-smi` showed `./.venv-server/bin/python` using GPU memory.

Interpretation:

- The run is alive and progressing.
- GPU is used for the neural network, but GPU utilization is expected to be low because the dominant per-step cost is still pandapower/powerflow and environment simulation.
- No loss curves or checkpoints are available yet because the first 672-step episode has not completed.
- Do not tune from this partial 72-step prefix; wait at least until episode 1 finishes, then inspect reward components, policy/value losses, entropy, KL, action landing, and private-profit traces.
