# DSO Sensitivity Attention Archived Metrics

Updated: 2026-05-28 Asia/Shanghai

This directory keeps small, report-grade CSV/JSON artifacts copied from the
current smoke and minimal training runs. The original run outputs stay under
`outputs/`, which is ignored by git and is suitable for large experiment data.

| File | Source | Meaning |
|---|---|---|
| `current_smoke_summary.json` | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/smoke_summary.json` | 2-step structured smoke summary |
| `current_smoke_step_metrics.csv` | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/smoke_step_metrics.csv` | step-level smoke reward/cost |
| `current_dso_operating_envelope.csv` | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/dso_operating_envelope.csv` | envelope diagnostics including cache refresh reasons, priority ActionUnits, partial refresh fields and sensitivity allocation weights |
| `current_happo_legacy_mlp_smoke_summary.json` | `outputs/dso_sensitivity_attention/happo_legacy_mlp_smoke_seed0/smoke_summary.json` | 2-step legacy MLP/flat-observation config smoke summary |
| `current_happo_legacy_mlp_smoke_step_metrics.csv` | `outputs/dso_sensitivity_attention/happo_legacy_mlp_smoke_seed0/smoke_step_metrics.csv` | legacy config step-level smoke reward/cost |
| `current_happo_legacy_mlp_dso_operating_envelope.csv` | `outputs/dso_sensitivity_attention/happo_legacy_mlp_smoke_seed0/dso_operating_envelope.csv` | legacy config envelope diagnostics |
| `current_baseline_rule_v0_smoke_summary.json` | `outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0/smoke_summary.json` | 2-step rule_v0 baseline smoke summary |
| `current_baseline_rule_v0_smoke_step_metrics.csv` | `outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0/smoke_step_metrics.csv` | rule_v0 baseline step-level smoke reward/cost |
| `current_baseline_rule_v0_dso_operating_envelope.csv` | `outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0/dso_operating_envelope.csv` | rule_v0 baseline envelope diagnostics |
| `current_dso_actor_outputs.csv` | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/dso_actor_outputs.csv` | raw DSO actor center/width/direction/lambda outputs |
| `current_decoded_operating_envelope.csv` | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/decoded_operating_envelope.csv` | safe-decoded preferred ranges and hard bounds |
| `current_bc_loss_metrics.csv` | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0/dso_sensitivity_attention_short_train_loss_metrics.csv` | BC warm-start loss, component losses and grad norm |
| `current_bc_loss_curve.svg` | generated from `current_bc_loss_metrics.csv` | log10 BC loss curve for the 256-step short train |
| `current_bc_decoded_operating_envelope.csv` | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0/decoded_operating_envelope.csv` | decoded envelope after the 256-step BC warm-start |
| `current_bc_short_train_summary.json` | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0/short_train_summary.json` | BC short-training summary |
| `current_happo_episode_metrics.csv` | `outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/happo_episode_metrics.csv` | HAPPO minimal episode reward/cost/projection/critic metrics |
| `current_happo_training_summary.json` | `outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/happo_training_summary.json` | HAPPO minimal structured training summary and stability metadata |
| `current_happo_update_metrics.csv` | `outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/happo_update_metrics.csv` | HAPPO role policy loss, entropy, approx KL and grad norm |
| `current_happo_frozen_eval_summary.json` | `outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current/happo_frozen_eval_summary.json` | structured HAPPO frozen deterministic eval summary |
| `current_happo_frozen_eval_step_metrics.csv` | `outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current/happo_frozen_eval_step_metrics.csv` | structured HAPPO frozen eval step reward/cost/safety metrics |

Boundary: these are small sanity artifacts, not paper-long convergence evidence.
