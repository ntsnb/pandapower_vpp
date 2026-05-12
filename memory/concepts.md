# Concepts

This file records stable project concepts. Add new entries as the research
framework grows.

## DSO

Distribution system operator. In this project, the DSO owns the full network
model, runs pandapower power flow, checks voltage / line / transformer
constraints, and issues feasible-region, price, or dispatch signals to VPPs.

## VPP

Virtual power plant aggregator. A VPP is a logical commercial / control entity,
not a pandapower native element. Its DER assets stay connected at their true
physical buses and pandapower element indices.

## DER Physical Bus

The real bus where a PV, storage system, flexible load, HVAC aggregator, EVCS,
or microturbine is injected into the pandapower network. VPP aggregation must
not move DER injections to a fake PCC bus unless the VPP is explicitly modeled
as a single-PCC equivalent.

## FR / DOE

Feasible region / dynamic operating envelope. This is a hard safety boundary,
not a price signal. It constrains what a VPP may inject or absorb at a PCC,
bus-vector, zone-vector, or DER-vector level.

## LFP / LFB

Local flexibility price / local flexibility bid. This is an economic signal for
network services such as voltage support, congestion relief, reverse-flow
absorption, peak shaving, or resilience. It is not automatically the same as LMP.

## LocalFlexNeed

A DSO-generated request describing where, when, and why the grid needs a
flexibility service.

## DispatchAward

A DSO clearing result that awards a flexibility task to one or more VPPs,
including quantity, time window, settlement proxy, and expected network effect.

## CTDE

Centralized training, decentralized execution. Critic-side training data may be
global, but each deployed VPP actor must only use its allowed local observation.

## NodeNeedEmbedding

A learned or statistical representation of the network service needs of a bus or
zone, based on voltage, loading, reverse-flow, forecast, and historical service
call signals.

## VppCapabilityEmbedding

A learned or statistical representation of a VPP's service capability, response
quality, reliability, location effectiveness, bid history, and deviation history.

## Project Memory Agent

A project-governance agent that updates `memory/*.md` with durable decisions,
user preferences, pitfalls, experiment levels, and progress after major changes.
It is not a DSO/VPP MARL agent and does not participate in environment
`reset()` / `step()` loops.

## Three-Gate Review

The minimum review loop for significant research changes:

1. Architecture/domain review for physical consistency, privacy, and CTDE
   boundaries.
2. Experiment review for benchmark quality, data realism, metrics, and claims.
3. UI/memory review for report synchronization and durable project knowledge.

## Smoke / Demo / Benchmark / Paper-Claim-Ready

Experiment evidence levels used by this project:

- `smoke`: proves code paths execute.
- `demo`: illustrates a scenario or user-facing workflow.
- `benchmark`: uses repeatable scenarios, multiple seeds, and comparable metrics.
- `paper-claim-ready`: adds realistic data, holdout scenarios, oracle/baseline
  comparisons, ablations, and statistically defensible results.
