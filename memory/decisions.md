# Decisions

## 2026-05-02: Separate Continuous Off-Policy Heads From Stochastic Research Scaffolds

Background:

- The user asked for a sharper MARL core without broad repo churn:
  MATD3 should stop collapsing all dispatch value into one mean reward critic,
  while HAPPO and HASAC need runnable research skeletons rather than paper-only
  placeholders.
- The existing privacy-separated CTDE encoders and env contracts were already
  stable, so replacing observation schemas or simulator interfaces would have
  been unnecessary risk.

Decision:

- Keep the existing actor observation and simulator action contracts unchanged.
- Upgrade MATD3 to a centralized twin critic with one DSO Q head and one
  independent dispatch Q head per VPP.
- Implement HAPPO as a bounded sequential role-update scaffold:
  DSO actor, shared dispatch actor, and shared portfolio actor update in order
  with cumulative importance correction and a centralized multi-head value
  critic.
- Implement HASAC as a bounded continuous-control scaffold:
  shared DSO/VPP continuous actors, centralized multi-head twin soft Q,
  replay, automatic entropy tuning, and soft target backup.
- Keep the slow discrete portfolio action outside MATD3 and HASAC.

Rejected options:

- Rewriting the env contract or observation schema for algorithm-specific
  tensors.
- Claiming full paper-grade HAPPO/HASAC training completeness from smoke tests.
- Forcing slow discrete portfolio actions into continuous MATD3/HASAC updates.

Residual risk:

- HAPPO still uses shared VPP actor parameters, so its sequential correction is
  a practical role-level approximation rather than a strict per-agent theorem
  implementation.
- HASAC and MATD3 remain bounded to the fast continuous dispatch path; slow
  portfolio search still needs a separate trainer or mechanism.

## 2026-05-01: Add Project-Level Agent Pack And Memory Keeper

Background:

- The global subagent registry contains many overlapping roles. In this project,
  direct global-agent routing caused repeated ambiguity among UI, ML,
  experiment, review, and orchestration agents.
- The user asked for a reusable project-level agent layer and a dedicated
  experience-summary agent that records durable preferences and rules.

Decision:

- Add `agents/` as the project-level subagent pack.
- Keep the global Codex agents as backing roles, but route recurring project
  work through project-specific roles:
  `ppvpp-architect`, `ppvpp-domain-guardian`,
  `ppvpp-market-mechanism-engineer`, `ppvpp-agent-engineer`,
  `ppvpp-simulator-engineer`, `ppvpp-experiment-critic`, and
  `ppvpp-memory-keeper`.
- Add `memory/user_preferences.md` for stable user expectations.
- Treat `ppvpp-memory-keeper` / `memory_curator_agent` as a project memory
  agent, not an environment MARL agent.

Rejected options:

- Modifying the global `C:\Users\admin\.codex\agents` registry directly.
- Creating many near-duplicate implementation agents for every global role.
- Treating a training supervisor or memory keeper as a VPP/DSO environment
  agent.

## 2026-05-01: Add Algorithm-UI-Experiment-Memory Quality Gate

Background:

- The user repeatedly observed that algorithm updates can leave HTML pages,
  dashboard data, or model diagrams stale.
- The project also needs experiment review to avoid mistaking smoke tests for
  paper-grade evidence.

Decision:

- Any algorithm model, reward, agent, actor/critic, training, FR/DOE, LFP/LFB,
  market, network, or VPP scenario change must be checked against four gates:
  architecture/domain review, experiment review, UI/report synchronization, and
  memory update.
- Static HTML and dashboard CSV outputs are part of the deliverable when the
  changed behavior is user-visible.

Rejected options:

- Considering code-only algorithm changes complete when tests pass but reports
  still describe the old model.
- Treating one-episode smoke training as scientific evidence.

Rollback:

- If a fast patch cannot run all gates, the final response must explicitly list
  the deferred gates and why.

Architecture decisions are appended here. Do not silently rewrite history; mark
old entries as `SUPERSEDED` when needed.

## 2026-05-01: Upgrade Primary CTDE Trainer With Deep Sets VPP Encoder And Action-Conditioned Critic

Background:

- The primary trainer `privacy_separated_ctde_actor_critic` had already split
  DSO, VPP dispatch, VPP portfolio, and critic modules, but the user requested
  a more mature execution-side architecture than a flat padded-MLP baseline.
- The existing VPP dispatch observation layout already had a stable
  `16 + K * 15` structure that could support a light set encoder without
  changing the environment contract or adding heavy dependencies.

Decision:

- Keep the VPP dispatch observation schema unchanged, but interpret it as
  `16` global/context features plus `K` DER tokens of width `15`.
- Implement a shared-token Deep Sets style dispatch encoder:
  context MLP, shared DER-token MLP, masked pooling over padded DER tokens, and
  a fusion MLP before aggregate and DER action heads.
- Upgrade the centralized critic from state-only to a lightweight
  action-conditioned critic using `critic_global_state` plus a compact joint
  action summary vector.
- Export architecture metadata through
  `deep_rl_training_summary.csv/json`, step metrics, trajectory rows, and the
  checkpoint payload so existing report generators can reflect the change
  without custom HTML edits.

Rejected options:

- Introduce a heavier Set Transformer, GNN, or recurrent temporal encoder now.
  That would raise runtime and implementation risk beyond the requested
  minimal runnable upgrade.
- Expand the critic to full MAPPO/GAE or counterfactual multi-agent credit now.
  The compact action summary critic is the smaller coherent step.

Rollback:

- Keep the observation and artifact schema stable. If the new encoder causes
  instability, swap the encoder implementation behind the same
  `vpp_input_dim`, `critic_action_summary_dim`, and training-summary fields.

## 2026-04-29: Add Memory Layer And Roadmap Alignment

Background:

- The root `AGENTS.md` now defines a long-term research direction around
  bidirectional DSO / VPP guidance, FR/DOE, LFP/LFB, privacy boundaries,
  three-time-scale control, MARL/HRL, and UI co-evolution.
- The repository already has a working pandapower simulator and visualization
  stack, but no persistent project memory folder.

Decision:

- Add `memory/` files for concepts, decisions, rules, pitfalls, progress,
  open questions, and experiments.
- Expand `docs/roadmap.md` from a short version list into a staged research and
  engineering roadmap.
- Treat UI and explainability as companion work for every non-trivial algorithm
  phase.

Rejected options:

- Only keep the plan inside `AGENTS.md`. That would make it harder to track
  project progress and delivered artifacts.
- Start directly with MARL or oracle OPF. That would risk bypassing schema,
  physical-injection, and privacy contracts.

Rollback:

- If the memory layer becomes noisy, keep the files but tighten write rules
  instead of deleting historical records.

## 2026-04-29: Prefer GPT-5.5 xhigh For Future Subagents

Background:

- The user requested that project subagents use GPT-5.5 xhigh to maintain
  engineering quality.
- Some tool-defined specialist roles have fixed model settings that cannot be
  overridden.

Decision:

- When a subagent is needed and explicit model selection is available, spawn a
  `default` subagent with `model="gpt-5.5"` and
  `reasoning_effort="xhigh"`.
- Put the specialist responsibility in the prompt, such as UI reviewer,
  algorithm architect, test engineer, or documentation reviewer.
- Avoid fixed-model specialist roles when the user has asked for GPT-5.5 xhigh
  consistency, unless there is no practical alternative.

Rollback:

- If future tool routing allows overriding specialist-role models, specialist
  roles can be used again with the same GPT-5.5 xhigh quality policy.

## 2026-04-29: Algorithm And UI Reviews Confirm Next Milestone Order

Background:

- A GPT-5.5 xhigh algorithm / architecture subagent reviewed the current code
  against `AGENTS.md`.
- A GPT-5.5 xhigh UI / visualization subagent reviewed the visualization stack
  against the next algorithm phases.

Decision:

- Keep the next engineering milestone order as:
  schema and physical / privacy contracts, FR/DOE v0, privacy observation
  filters, LFP/LFB v0-v1, VPP inner solver v0, multi-VPP clearing and
  settlement.
- Pair each algorithm phase with a dashboard / HTML inspection artifact.
- Treat the current Gym environment as centralized / global smoke environment
  until privacy-filtered actor observations are implemented.

Key risks to monitor:

- VPP aggregate P/Q boxes are not enough for multi-node VPP network security.
- `DSO.calculate_reward_or_cost()` should not use private VPP true operating
  cost in non-oracle modes once market privacy modes are introduced.
- LFP/LFB must not be displayed as ordinary energy price or wholesale LMP.

## 2026-04-29: Implement v0 FR/DOE As Local Bounds With Audit Trace

Background:

- The current simulator needed a concrete step beyond planning so future
  market, RL, and UI work can rely on explicit contracts.
- Full OPF-based feasible regions are not yet appropriate because schema,
  physical injection, and privacy boundaries must stabilize first.

Decision:

- Implement v0 FR/DOE as static box bounds derived from current DER local
  constraints.
- Use `pcc` scope for true single-PCC VPPs and `bus_vector` scope for multi-node
  VPPs.
- Record a projection trace inside `Simulator.step()` so every command can be
  audited from raw target through device bounds, FR/DOE, pandapower write, and
  power-flow result.
- Export the new state through dashboard frames and show it in read-only HTML /
  Dash tables.

Rejected options:

- OPF-based envelopes now. That would add solver complexity before the data
  contracts are stable.
- Only aggregate VPP P/Q bounds. That would be acceptable for business summary
  but unsafe as a multi-node network security envelope.

Rollback:

- Keep schema and trace records stable. Future OPF or sensitivity FR methods can
  replace `source_method` while preserving `FRObject` and dashboard frame shapes.

## 2026-04-29: Add LFP/LFB v0 As Stress-Driven Network-Service Signal

Background:

- The next research step after FR/DOE is to separate economic flexibility
  guidance from ordinary energy price.
- A full distribution market clearing mechanism is premature before bid,
  dispatch award, settlement, and reliability schemas mature.

Decision:

- Add `optimization/local_flex_market.py` with stress-driven `LocalFlexNeed`,
  `LocalFlexPrice`, rule-based VPP bids, and simple effective-price clearing.
- Keep LFP/LFB explicitly labeled as a local network-service signal, not LMP.
- Do not expose private DER `cost_coefficients` through VPP bids.

Rejected options:

- Calling the signal LMP or deriving it from OPF duals immediately.
- Clearing purely by lowest bid without reliability / location-effectiveness
  normalization.

Rollback:

- The v0 functions are pure and can be replaced by a richer market module while
  preserving `LocalFlexNeed`, `VPPFlexBid`, and `DispatchAward`.

## 2026-04-29: Define Heterogeneous MARL Roles And Runnable Baseline Harness

Background:

- The user clarified that the project should not be a single generic RL
  environment. It needs a DSO global guidance agent, per-VPP fast dispatch
  agents, and per-VPP slow portfolio agents, plus a training supervisor.
- The project also needs classic heterogeneous MARL baselines and tuning records
  before deeper algorithm work.

Decision:

- Add explicit agent roles:
  `dso_global_guidance`, `vpp_i_dispatch`, `vpp_i_portfolio`, and
  `deep_training_supervisor`.
- Add first-pass bidirectional encoders:
  `NodeNeedEmbedding`, `VppCapabilityEmbedding`, and `VppGridNeedBelief`.
- Add runnable IPPO/MAPPO/MADDPG/QMIX-style baseline harness with lightweight
  NumPy policies so experiments can run quickly and deterministically now.
- Add a `TrainingSupervisor` that sweeps hyperparameters, writes
  `tuning_trials.csv` and `training_summary.csv`, and returns
  `needs_algorithm_review` when convergence or constraint criteria are not
  satisfied.
- Surface training status in Dash and offline HTML instead of leaving it as
  command-line-only output.

Rejected options:

- Starting directly with heavy deep networks inside the power-flow loop. That
  would slow iteration before the agent roles and experiment outputs are stable.
- Treating the existing centralized `VPPDSOEnv` observation as a valid VPP actor
  observation. It remains a smoke/global environment until privacy-filtered MARL
  envs are implemented.

Rollback:

- The lightweight policies can be replaced by PyTorch implementations while
  preserving the experiment CSV outputs and agent role map.

## 2026-04-30: Add Minimal CTDE Interface Contract Without Rewriting Training

Background:

- The current deep RL path is a real shared actor-critic trainer, but the next
  engineering step required a concrete CTDE contract instead of another
  roadmap-only placeholder.
- The user explicitly asked for work only in `learning/envs/tests`, with no
  visualization changes and no large training rewrite.

Decision:

- Add `learning/ctde_interface.py` as the minimal CTDE interface layer.
- Represent DSO actor, VPP dispatch actor, VPP portfolio actor, policy-module
  skeletons, and the centralized critic contract as explicit dataclasses.
- Support two layouts:
  `shared_actor_critic` for the current trainer and
  `independent_actor_scaffold` for the next split-actor step.
- Keep legacy action payloads valid:
  `normalized_setpoint_bias`, `response_bias`, `target_p_mw`,
  `selected_p_mw`, and `normalized_der_actions` are normalized into one schema.
- Surface the contract and validation results through
  `MultiAgentVPPDSOEnv.reset()`/`step()` info without changing simulator
  physics, visualization, or reward design.

Rejected options:

- Splitting `deep_rl.py` into separate DSO/VPP policies now. That would change
  the training algorithm instead of landing the requested next-step interface.
- Validating by hard-failing `env.step()` on every malformed action. That would
  risk breaking existing smoke paths; the current choice records errors while
  preserving the deterministic environment path.

Rollback:

- If a future trainer wants a different contract object, keep the field names
  and action normalization stable and swap the dataclass layer behind them.
