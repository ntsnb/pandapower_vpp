from __future__ import annotations

from vpp_dso_sim.dso.envelope.rule_v0 import RuleV0EnvelopePolicy
from vpp_dso_sim.dso.envelope.sensitivity_attention_v1 import SensitivityAttentionEnvelopePolicy


def build_dso_envelope_policy(config: dict | None = None):
    payload = dict(config or {})
    dso_cfg = dict(payload.get("dso", {}))
    policy = str(dso_cfg.get("envelope_policy", "rule_v0"))
    if policy == "rule_v0":
        return RuleV0EnvelopePolicy()
    if policy == "sensitivity_attention_v1":
        return SensitivityAttentionEnvelopePolicy(payload)
    raise ValueError(f"Unsupported DSO envelope_policy: {policy}")
