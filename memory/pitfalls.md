# Pitfalls

Record risks, false assumptions, and direction drift here.

## Avoid Centralized-Only Control Drift

The project goal is not a pure DSO centralized dispatch of all DER. Centralized
or full-information solutions are useful as oracle baselines, but the main
framework should preserve VPP autonomy and privacy boundaries.

## Avoid Fake PCC Aggregation

A VPP may be a commercial aggregator across several buses. Do not replace all
member DER injections with one equivalent injection at the VPP PCC unless the
scenario explicitly models a single-PCC equivalent.

## Avoid Confusing FR/DOE With LFP/LFB

FR/DOE is a hard feasible envelope. LFP/LFB is a local flexibility economic
signal. They should be represented by separate schemas and displayed separately.

## Avoid UI Lag Behind Algorithms

If new algorithm states are invisible in the report or dashboard, debugging will
become guesswork. Add visual and tabular inspection paths when adding FR/DOE,
LFP/LFB, clearing, reward, settlement, or MARL observations.

## Avoid Actor Privacy Leakage

Do not let VPP actors consume full topology, other VPP costs, other VPP device
state, or oracle solutions in default decentralized execution.

## Avoid Stale Visualization After Algorithm Changes

If `learning/`, `envs/`, `optimization/`, reward logic, or agent architecture
changes but `outputs/interactive_report.html`, `outputs/rl_architecture.html`,
`outputs/vpp_first_person/*.html`, and `outputs/dashboard_data/*.csv` still show
old structures, the project becomes hard to debug. Treat stale UI as an
incomplete deliverable, not a cosmetic issue.

## Avoid Skipping Architecture And Experiment Review

Network, market, and DRL/MARL changes can silently break physical consistency,
privacy boundaries, or paper-claim validity. Use the project-level architecture
/ domain guardian and experiment critic roles before claiming a stage is done.

## Avoid Treating Smoke Tests As Research Evidence

One-episode or two-step training proves the code path executes. It does not
prove convergence, optimality, fairness, profitability, market realism, or
paper-grade performance.
