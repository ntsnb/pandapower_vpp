# Known Failures

Updated: 2026-05-28 Asia/Shanghai

## Current status

No failing tests are currently recorded for the implemented `sensitivity_attention_v1` smoke/sanity slice.

Known limitation:

- Structured HAPPO minimal training is implemented and writes `happo_update_metrics.csv` with `policy_loss`, `entropy_mean`, `approx_kl` and `grad_norm`.
- Frozen evaluation for structured HAPPO checkpoints is implemented for the current trainer/evaluator slice.
- Runtime `SensitivityAttentionEnvelopePolicy` can load direct or structured-HAPPO attention actor checkpoints via config.
- Current HAPPO evidence is 1 episode / 2 step chain verification, not paper-long convergence evidence.
- HAPPO stability fields `target_kl`, `nan_guard`, observation normalization, advantage normalization, config hash and seed logging are implemented and covered by focused tests.
- Observation normalization is deterministic per-vector normalization, not a checkpointed running mean/std normalizer.

## Pre-existing dirty worktree

Before this upgrade pass, the worktree already contained modifications in:

- `src/vpp_dso_sim/experiments/paper_training.py`
- `src/vpp_dso_sim/optimization/ac_security_projection.py`
- `src/vpp_dso_sim/simulation/simulator.py`
- `tests/test_ac_security_projection.py`
- `tests/test_paper_training_experiment.py`

These are treated as pre-existing until proven otherwise.
