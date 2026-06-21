# DSO sensitivity_attention_v1 Training Protocol

Updated: 2026-05-28 Asia/Shanghai

## 1. Required sequence before paper-long

1. Run schema/model/decoder tests.
2. Run `rule_v0` smoke.
3. Run `sensitivity_attention_v1` smoke.
4. Run short BC warm-start sanity and inspect loss metrics.
5. Only after the above, connect structured actor to full HAPPO online training.
6. Run short HAPPO structured training before paper-long.

## 2. Current verified short sanity

Command:

```bash
./.venv-server/bin/python scripts/run_short_train.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 256 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0
```

Artifacts:

- `dso_sensitivity_attention_short_train_loss_metrics.csv`
- `dso_sensitivity_attention_actor.pt`
- `decoded_operating_envelope.csv`
- `short_train_summary.json`

Observed:

- initial `bc_loss`: 0.5577945709228516
- final `bc_loss`: 0.00032648438354954123
- NaN/Inf: false

## 3. Interpretation boundary

This is a behavior-cloning warm-start sanity check. It proves gradients, masks, decoder and metrics are functional. It does not prove online RL convergence.
