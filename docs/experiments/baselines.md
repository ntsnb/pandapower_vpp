# DSO sensitivity_attention_v1 Baselines

Updated: 2026-05-28 Asia/Shanghai

## Baseline files

| Config | Purpose |
|---|---|
| `configs/baseline_rule_v0.yaml` | Rule-based DSO envelope baseline |
| `configs/happo_legacy_mlp.yaml` | Legacy HAPPO/MLP DSO actor baseline route |
| `configs/happo_sensitivity_attention_v1.yaml` | Main structured DSO actor smoke/sanity config |
| `configs/ablation_no_sensitivity_edges.yaml` | Ablation: remove sensitivity-edge information |
| `configs/ablation_no_action_self_attention.yaml` | Ablation: disable ActionUnit self-attention |
| `configs/ablation_no_width_penalty.yaml` | Ablation: disable DSO envelope width penalty |

## Current verified baseline

`rule_v0` was verified with:

```bash
./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/baseline_rule_v0.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0
```

Result:

- 2 steps completed.
- No NaN/Inf detected.
- `dso_operating_envelope.csv` written.

## Interpretation

Baseline smoke verifies runtime compatibility. It does not provide paper-level statistical evidence.
