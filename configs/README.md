# Config Layout

This directory keeps old `configs/<name>.yaml` paths working while the actual
config bodies live in versioned directories.

## Canonical Directories

| Directory | Purpose |
|---|---|
| `scenarios/demo/` | Small demo and smoke-test scenarios. |
| `scenarios/benchmark/` | Benchmark feeder scenarios and holdout variants. |
| `algorithms/dso_sensitivity_attention/v1/` | DSO sensitivity-attention v1 baseline, HAPPO, and ablation configs. |
| `rewards/v2_minimal/` | Reusable reward configuration fragments. |
| `experiments/paper_long/sensitivity_attention_v1/` | Paper-long sensitivity-attention experiment configs and reward variants. |

## Compatibility Wrappers

Root-level YAML files in this directory are wrappers. They contain only an
`extends:` pointer to the canonical file so historical commands and experiment
manifests remain readable.

New scripts should prefer either canonical paths or aliases from
`configs/registry.yaml`.

Examples:

```bash
python examples/17_paper_training_experiment.py \
  --config-path configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml
```

and:

```python
from vpp_dso_sim.utils.config import load_yaml

cfg = load_yaml("paper_long_sensitivity_v1_reward_v2_minimal")
```
