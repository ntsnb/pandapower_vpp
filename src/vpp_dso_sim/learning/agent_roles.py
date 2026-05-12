from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AgentRole:
    agent_id: str
    role_type: str
    owner_id: str
    time_scale: str
    objective: str
    action_summary: str
    observation_summary: str
    privacy_scope: str
    trainable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EncoderRole:
    encoder_id: str
    owner: str
    update_scale: str
    output_name: str
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_agent_role_map(vpps) -> list[AgentRole]:
    """Return the heterogeneous agent taxonomy used by the MARL harness."""

    roles: list[AgentRole] = [
        AgentRole(
            agent_id="dso_global_guidance",
            role_type="dso_guidance_agent",
            owner_id="dso",
            time_scale="fast_to_middle",
            objective="Guide VPP targets, FR/DOE and local flexibility signals while preserving grid safety.",
            action_summary="VPP target deltas, local flex price/need, operating envelopes",
            observation_summary="Network state, VPP reports, capability embeddings, node need embeddings",
            privacy_scope="DSO global view without default access to private VPP cost functions",
            trainable=True,
        )
    ]
    for vpp in vpps:
        roles.append(
            AgentRole(
                agent_id=f"{vpp.id}_dispatch",
                role_type="vpp_dispatch_agent",
                owner_id=vpp.id,
                time_scale="fast",
                objective="Disaggregate DSO/VPP targets into DER setpoints inside local constraints and FR/DOE.",
                action_summary="PV curtailment/Q, ESS charge/discharge, flexible load, EVCS, HVAC, MT setpoints",
                observation_summary="Own DER state, own FR/DOE, own local flex signal, own response history",
                privacy_scope="Own VPP only; no other VPP private state or full topology",
                trainable=True,
            )
        )
        roles.append(
            AgentRole(
                agent_id=f"{vpp.id}_portfolio",
                role_type="vpp_portfolio_agent",
                owner_id=vpp.id,
                time_scale="slow",
                objective="Decide whether to change the VPP commercial DER portfolio without moving physical DER buses.",
                action_summary="Portfolio keep/add/remove/reweight proposals, evaluated on slow episodes",
                observation_summary="Own profit/reliability history, DSO service calls, own capability belief",
                privacy_scope="Own portfolio and history; no other VPP private strategy",
                trainable=True,
            )
        )
    roles.append(
        AgentRole(
            agent_id="deep_training_supervisor",
            role_type="training_supervisor_agent",
            owner_id="experiment",
            time_scale="experiment",
            objective="Run baseline algorithms, tune hyperparameters, detect convergence and return failures to the main thread.",
            action_summary="Select algorithm/config trials, early stop, request algorithm review on non-convergence",
            observation_summary="Episode metrics, reward trend, constraint violations, training stability",
            privacy_scope="Experiment metadata and allowed centralized-training data",
            trainable=False,
        )
    )
    return roles


def build_encoder_role_map() -> list[EncoderRole]:
    return [
        EncoderRole(
            encoder_id="node_need_encoder",
            owner="dso",
            update_scale="middle_to_slow",
            output_name="NodeNeedEmbedding",
            purpose="Encode voltage, congestion, reverse-flow and resilience needs by node or zone.",
        ),
        EncoderRole(
            encoder_id="vpp_capability_encoder",
            owner="dso",
            update_scale="slow",
            output_name="VppCapabilityEmbedding",
            purpose="Estimate each VPP service capability, reliability, delay and location effectiveness.",
        ),
        EncoderRole(
            encoder_id="vpp_grid_need_belief_encoder",
            owner="vpp",
            update_scale="slow",
            output_name="VppGridNeedBelief",
            purpose="Let each VPP learn where and when DSO flexibility needs are likely to be valuable.",
        ),
    ]

