# Real Training Readiness Audit

This audit records what changed in the next algorithm architecture step and
what still blocks publication-grade long-cycle training.

## Implemented In This Step

1. The privacy-separated CTDE trainer now uses a MAPPO/HAPPO-lite update style:
   role-aware value heads, GAE-lambda advantages and PPO clipped surrogate
   losses.
2. The centralized critic remains training-only and action-conditioned, but now
   outputs three baselines: `V_dso`, `V_dispatch`, and `V_portfolio`.
3. The algorithm id remains `privacy_separated_ctde_actor_critic` for
   checkpoint/report compatibility. The precise update rule is recorded as
   `mappo_happo_lite_gae_single_epoch_clipped_surrogate`.
4. Loss CSVs now expose `dso_value_loss`, `dispatch_value_loss`,
   `portfolio_value_loss`, `gae_lambda`, `ppo_clip_ratio` and role advantage
   summaries.
5. The project now includes a dataset registry and a real open-data downloader.
   The first downloaded dataset is NREL SMART-DS Austin P1U OpenDSS data.
6. The project now includes a separate MATD3 implementation for the continuous
    dispatch subproblem. It has a replay buffer, centralized multi-head twin Q
    critics, one DSO Q head plus one dispatch Q head per VPP, target
    actor/critic networks, target policy smoothing and delayed actor updates.
    It now uses per-VPP dispatch actors by default and deliberately leaves the
    discrete slow portfolio action outside MATD3.
7. Dedicated smoke-capable HAPPO and HASAC research trainers now exist in
   `learning/advanced_marl.py`. HAPPO implements sequential role updates with
   cumulative importance correction. Its default topology is now hierarchical:
   one DSO global actor, one independent dispatch actor per VPP, and one
   independent slow-loop portfolio actor per VPP. HASAC implements squashed
   Gaussian actors, automatic entropy tuning, twin soft Q critics, soft Bellman
   backup and off-policy replay for the continuous dispatch subproblem. HASAC
   also uses per-VPP dispatch actors by default.
8. `learning/reward_contracts.py` now records the explicit reward contract:
   dispatch rewards are self-interested and do not share raw DSO global reward;
   portfolio rewards may use localized DSO-alignment/service signals on a
   slow-loop basis.

## What This Still Is Not

`privacy_separated_ctde_actor_critic` remains the backward-compatible
MAPPO/HAPPO-lite trainer for older reports. The recommended HAPPO path is now
the hierarchical trainer in `advanced_marl.py`: it uses per-VPP dispatch and
per-VPP slow-loop portfolio actors by default, with optional sharing retained
only as an ablation. HAPPO and HASAC still should not be used as
publication-grade convergence evidence without multi-seed, long-horizon
training and holdout evaluation.

`matd3_continuous_dispatch` is now a complete MATD3-style core for continuous
DSO/VPP dispatch: deterministic actors, centralized multi-head twin Q critics,
target networks, replay, target smoothing and delayed actor updates are
present. Its claim boundary is narrower than the whole research problem: it
does not train the discrete slow-cycle VPP portfolio agent.

## Remaining Blockers Before Long Training

| Area | Current risk | Required fix |
|---|---|---|
| Long horizon profiles | Long runs can still repeat finite profiles if adapters do not provide true multi-day data. | Add real profile adapters and split train/eval/holdout by date/feeder/scenario. |
| Scenario seed diversity | `reset(seed=...)` is not enough if profile generation uses static YAML seeds. | Make seeds control profile sampling, DER placement perturbations and price traces. |
| Multi-node VPP safety | Multi-node VPPs are still often reduced to aggregate P bounds. | Implement bus/zone vector DOE and projection for multi-node portfolios. |
| DSO operating envelope | Current envelopes are heuristic FR/DOE approximations. | Add sensitivity/OPF-certified envelopes and replay validation. |
| Local flexibility market | LFP/LFB and settlement objects are not yet the main step loop. | Connect LocalFlexNeed -> VPPFlexBid -> clearing -> DispatchAward -> delivery/deviation settlement. |
| Privacy proof | Current tests check observation boundaries but do not produce a formal audit artifact per run. | Export actor-visible, critic-visible and oracle-only fields every benchmark run. |
| First-person debugging | Some benchmark reports infer VPP narratives from aggregate data. | Drive first-person pages from true per-agent rollout/action/reward trajectories. |
| Algorithm comparison | Candidate campaign still uses adapters for many algorithms. | Only compare true implementations in paper tables; adapter rows stay engineering evidence. |

## Minimum Real-Training Protocol

Use this protocol before claiming learning performance:

- `episodes >= 100`
- `horizon_steps >= 672` for seven days at 15-minute resolution
- `seeds >= 5`
- at least three splits: train, validation/eval, holdout stress
- rule-based, OPF/safety-projection, oracle/full-information, and no-flexibility baselines
- report safety, cost, private VPP profit proxy, settlement components,
  projection gap, delivery reliability, fairness and privacy-audit metrics
- run frozen deterministic evaluation; do not report training rollout reward as
  policy performance

## Immediate Engineering Plan

1. Build a `SMARTDSOpenDSSAdapter` around the downloaded OpenDSS data.
2. Add a `ProfileSplitManager` that prevents silent modulo profile reuse.
3. Add `PrivacyAuditFrame` generation to benchmark outputs.
4. Upgrade multi-node VPP action/projection from scalar aggregate P to bus/zone
   vectors.
5. Connect local flexibility market settlement to the main simulator step.
6. Implement true MATD3 and MAPPO variants after the data/market/safety protocol
   is stable.
