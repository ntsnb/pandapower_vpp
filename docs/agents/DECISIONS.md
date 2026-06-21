# DSO Sensitivity Attention Decisions

Updated: 2026-05-28 Asia/Shanghai

## ADR-001: No zone fields in v1

`sensitivity_attention_v1` does not introduce or depend on `zone_id` in ActionUnit, NetworkObject, or DSO actor observation. Existing legacy portfolio metadata may still contain zones, but the v1 actor path must not consume them.

## ADR-002: No reliability field in v1

Reliability modeling is out of scope for this version. It must not be added to actor observation tokens.

## ADR-003: Keep private portfolio cost out of DSO actor observation

DSO execution actor may observe bids, physical connection information, FR/DOE bounds, public network state, and response/projection history. It must not observe true DER cost coefficients, user comfort preferences, or private SOC internals.

## ADR-004: Use ActionUnit x NetworkObject sensitivity edges

Location awareness is represented as an edge tensor between controllable ActionUnits and selected NetworkObjects.

## ADR-005: Use edge-biased bipartite attention

The trainable DSO actor should let each ActionUnit attend to critical NetworkObjects with attention bias from sensitivity edges and object severity.

## ADR-006: Use safe decoder instead of direct hard-bound output

The neural actor outputs center, width, direction and guidance strength only. Safe decoder maps those values into FR/DOE hard bounds.

## ADR-007: Preserve rule_v0 as baseline

Existing rule-based envelope behavior remains available and regression-tested.
