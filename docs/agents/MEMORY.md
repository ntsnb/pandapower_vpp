# Agent Memory

- FRObject / FR/DOE remains the safety hard boundary.
- `dso_operating_envelope` is guidance, not formal market award or settlement.
- Current `sensitivity_attention_v1` experiment uses no new zone partitioning.
- Current `sensitivity_attention_v1` actor schema uses no `zone_id`.
- Current `sensitivity_attention_v1` actor schema uses no `reliability` field.
- ActionUnit is VPP-PCC, VPP-bus, or DER fallback.
- DSO execution actor must not read private VPP true costs, comfort preference, or private SOC internals.
- Sensitivity matrix is initialized by finite-difference AC power flow, not all-zero learning.
- Rule-based envelope remains baseline as `rule_v0`.
- Legacy flat DSO observation remains baseline as `legacy_flat`.
- New method name is `sensitivity_attention_v1`.
- Smoke success does not prove convergence.
