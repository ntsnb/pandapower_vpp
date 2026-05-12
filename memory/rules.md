# Rules

## Physical Consistency

- DER setpoints must be written to their true pandapower element and bus.
- Multi-node VPPs must not be collapsed into a fake PCC injection for network
  security checks.
- Internal project active-power sign convention remains:
  `P > 0` injects into the grid and `P < 0` absorbs from the grid.
- pandapower storage sign convention remains:
  `storage.p_mw > 0` charges and `storage.p_mw < 0` discharges.

## Privacy

- DSO may use topology, network states, constraints, VPP bids, delivered
  response, settlement, and reliability history.
- DSO must not read private VPP cost functions or device-private preferences in
  default non-oracle modes.
- A VPP actor must not receive the full topology, other VPP private state, or
  oracle centralized solution unless the experiment is explicitly marked as an
  oracle / full-information baseline.

## Algorithm Pipeline

- Raw actions must pass through action decoding, device constraints, FR/DOE or
  dispatch-award projection, sign conversion, true bus injection, and pandapower
  power flow.
- FR/DOE is a safety envelope. LFP/LFB is an economic flexibility signal.
- Do not call a distributed result globally optimal without an oracle baseline.

## UI Companion Rule

Every new algorithmic state should have a matching inspectable artifact:

- CSV / JSON export for reproducibility.
- Offline HTML or Dash view for human inspection.
- Explanation record when the state affects VPP dispatch, reward, settlement, or
  constraint violations.
- Benchmark pages must be benchmark-aware: after changing algorithms, reward
  semantics, training/evaluation protocols, or benchmark metrics, refresh HTML
  from `outputs/<benchmark_dir>` artifacts (`seed_metrics.csv`,
  `step_summary.csv`, `frozen_eval_summary.csv`) rather than rerunning only a
  generic scenario.

## Testing

New modules should add at least one of:

- schema serialization test,
- sign convention test,
- physical injection mapping test,
- FR/DOE bounds test,
- local flexibility clearing test,
- privacy observation filter test,
- VPP inner solver feasibility test,
- dashboard / visualization smoke test.

## Subagent Quality

- When the tool allows explicit model selection, use `model="gpt-5.5"` and
  `reasoning_effort="xhigh"` for project subagents.
- If a specialist role has a fixed model that cannot be overridden, prefer a
  `default` subagent and describe the specialist responsibility in the prompt.
- Keep subagent tasks bounded, read-only unless implementation ownership is
  explicit, and require changed files / validation / residual risk in coding
  subtasks.

## UI And Algorithm Co-Design

- FR/DOE additions need visible envelopes, operating points, raw targets, and
  projected targets.
- LFP/LFB additions need zone / time heatmaps and service-type explanations,
  separate from ordinary price profiles.
- Clearing additions need bid, kappa, reliability, awarded quantity, settlement,
  delivered response, and deviation views.
- MARL/HRL additions need actor observation, critic state, raw action, projected
  action, reward component, and privacy-field inspection views.

## Project Agent Routing

- Read `agents/project_agent_registry.yaml` before selecting subagents for
  complex project work.
- Use project-level roles to avoid global-agent overlap:
  - `ppvpp-architect` for architecture and stage boundaries.
  - `ppvpp-domain-guardian` for physical consistency and privacy boundaries.
  - `ppvpp-market-mechanism-engineer` for FR/DOE, LFP/LFB, bids, clearing, and settlement.
  - `ppvpp-agent-engineer` for DRL/MARL/CTDE/HRL env and training changes.
  - `ppvpp-simulator-engineer` for pandapower simulation changes.
  - `ppvpp-experiment-critic` for paper-grade experiment audit.
  - `ppvpp-memory-keeper` for long-term project memory.
- Tool limitation: a full-history forked subagent inherits the parent model and
  role and cannot simultaneously override `agent_type`, `model`, or
  `reasoning_effort`. When project quality requires a specific specialist and
  model, do not fork full history; pass the needed context explicitly in the
  prompt.

## Collaboration Quality Gates

- A network or physical-modeling change is not complete until the domain
  guardian and experiment critic have reviewed the physical and experimental
  implications, or their review is explicitly deferred with a reason.
- An algorithm, reward, actor/critic, training, or MARL change is not complete
  until the MARL architecture engineer and experiment critic have reviewed it,
  and UI synchronization has been checked.
- A visible model or scenario change is not complete until matching dashboard
  frames and static HTML reports are regenerated or the final response clearly
  states why they were not.
- A stage is not closed until `memory/progress.md` records the result and
  `memory/decisions.md` or `memory/pitfalls.md` records durable decisions or
  risks.

## Workspace Test Command Preference

On this Windows workspace, run tests from:

```powershell
C:\Users\admin\Desktop\panda power\pandapower-vpp-dso-sim
```

Prefer pytest commands that keep temporary files under `outputs/`:

```powershell
python -m pytest -q --basetemp=outputs\pytest_tmp_<tag> -o cache_dir=outputs\pytest_cache_<tag>
```
