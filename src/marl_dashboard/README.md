# MARL Dashboard

Local realtime experiment dashboard for multi-agent VPP/DSO training.

## Start a Demo

```bash
marl-dashboard demo --data-dir runs --host 127.0.0.1 --port 8765
```

For CI/smoke data generation without a foreground server:

```bash
marl-dashboard demo --data-dir /tmp/marl-dashboard-demo --run-id smoke_demo --no-serve
```

The service binds to `127.0.0.1` by default and prints:

```text
Dashboard running at http://127.0.0.1:8765
```

## Serve Existing Runs

```bash
marl-dashboard serve --data-dir runs --host 127.0.0.1 --port 8765
```

The frontend exposes:

- Overview: current run status, selected date/time/VPP, key values and recent events.
- Dataset: six physical-data panels plus a combined dataset chart and table.
- Reward/Cost: reward and cost trajectories, composition views, tables and formulas.
- Loss: learner loss terms by policy/gradient step when the trainer logs them.
- Compare: same-time comparison across VPPs, policies or epochs.
- Variable Dictionary: searchable names, units, symbols, meanings and formulas.
- Topology: pandapower distribution topology plus detailed bilingual VPP/DER configuration.
- Run Config: run metadata, config, formulas and event log.

## Training Integration

For a runnable minimal example that shows where to place logger calls around
`env.step`, episode end, and learner update hooks:

```bash
PYTHONPATH=src python examples/integrate_logger_example.py \
  --data-dir /tmp/marl-dashboard-integration-example \
  --run-id integration_example_run \
  --dry-run
```

```python
from marl_dashboard.logging import ExperimentLogger, start_dashboard

dashboard = start_dashboard(data_dir="runs", host="127.0.0.1", port=8765)
logger = ExperimentLogger(run_id="my_run", data_dir="runs", config={"algorithm": "happo"})

logger.log_dataset(
    epoch_id=0,
    episode_id=0,
    env_id="env_0",
    vpp_id="vpp_001",
    date=None,
    time_index=0,
    timestamp=None,
    values={"electricity_price": 85.0, "storage_soc": 61.0},
    units={"electricity_price": "$/MWh", "storage_soc": "%"},
)
logger.log_reward_terms(epoch_id=0, episode_id=0, vpp_id="vpp_001", terms={"total_reward": 1.0})
logger.log_loss_terms(epoch_id=0, gradient_step=1, policy_id="dispatch_shared", terms={"actor_loss": 0.1})
logger.close()
dashboard.stop()
```

Recommended hook points are:

- after `env.reset`: log initial dataset/observation context;
- after `env.step`: log dataset values, action-derived values, reward/cost terms and events;
- after learner updates: log `actor_loss`, `critic_loss`, `entropy_loss`, `value_loss`,
  `q_loss`, `total_loss`, learning rate and optional gradient norm;
- at episode end: log `episode_return`, `episode_length` and terminal reason;
- at checkpoint/eval/train end: log events and flush/close the logger.

The logger is a side-effect sink. It should not alter rewards, costs, losses, random
state, optimizer state, environment state, or checkpoint contents.

## Paper Training Integration

The paper training entry point can start the local dashboard and export summary
metrics after the experiment finishes:

```bash
PYTHONPATH=src python examples/17_paper_training_experiment.py \
  --preset smoke \
  --algorithms rule_based \
  --seeds 9709 \
  --horizon-steps 1 \
  --eval-horizon-steps 1 \
  --no-tensorboard \
  --no-html \
  --dashboard \
  --dashboard-auto-port
```

This integration reads the existing paper-training output DataFrames/CSVs and
writes dashboard Parquet tables. It does not modify `env.reset`, `env.step`,
learner updates, rewards, costs, losses, or checkpoints.

## Data Schema

All metric rows include stable dimensions such as `run_id`, `epoch_id`, `episode_id`,
`batch_id`, `gradient_step`, `global_env_step`, `env_id`, `vpp_id`, `agent_id`,
`policy_id`, `date`, `time_index`, `timestamp`, `metric_group`, `metric_name`,
`value`, `unit`, `formula_latex`, and `description`.

Tables are written under:

```text
runs/{run_id}/
  metadata.json
  config.json
  variable_dictionary.json
  formulas.json
  tables/
    dataset_timeseries/
    reward_terms/
    cost_terms/
    loss_terms/
    scalar_metrics/
    events/
```

`dataset_timeseries` should contain physical or de-normalized values where possible.
If only normalized values are available, use a metric name or description that makes
the normalization explicit. Units are part of the schema and should be present for
all charted quantities.

## Concept Glossary

`epoch_id` is a learner/update round. It is not assumed to mean a full dataset pass.
`episode_id` is one environment reset-to-terminal/truncated trajectory. In VPP
day-ahead dispatch this is often one scheduling period such as 24 hourly steps or
96 quarter-hour steps, but the logger does not enforce that. `batch_id` identifies
the sampled batch used for one or more optimizer updates. `gradient_step` is the
actual optimizer update count. `global_env_step` is cumulative environment
interaction. `time_index` is the energy profile step within a `date`; it is not a
gradient step. `vpp_id`, `agent_id`, and `policy_id` are kept separate because
multiple VPP agents may share one policy.

If a live run shows only `epoch_id = 0` but many `episode_id` values, the UI is not
duplicating filters. It means the current logger stream has one learner/update
dimension and multiple reset-to-end trajectories. In that case the epoch selector
is informational, while the episode selector is the useful control for choosing
one trajectory and preventing several trajectories from being mixed on the same
date/time axis.

## Tests and Build

Python dashboard tests:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_marl_dashboard_logger.py \
  tests/test_marl_dashboard_api.py \
  tests/test_marl_dashboard_paper_training_integration.py
```

Frontend tests and production build:

```bash
cd src/marl_dashboard/frontend
npm test
npm run build
```

The Vite dev server and FastAPI server both default to `127.0.0.1`.
