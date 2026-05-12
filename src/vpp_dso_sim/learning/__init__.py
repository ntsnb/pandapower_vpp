"""Learning, MARL baseline and tuning utilities."""

import importlib

from vpp_dso_sim.learning.agent_roles import build_agent_role_map
from vpp_dso_sim.learning.advanced_marl import (
    ALGORITHM_REGISTRY,
    HAPPOConfig,
    HASACConfig,
    MultiHeadValueCriticSpec,
    TwinCriticSpec,
    build_matd3_twin_critic_spec,
    build_multi_head_value_critic,
    evaluate_happo_checkpoint,
    evaluate_hasac_checkpoint,
    get_algorithm_registry,
    rank_algorithm_candidates,
    train_happo,
    train_hasac,
)
from vpp_dso_sim.learning.advanced_trainers import (
    HASACReplayBuffer,
    advanced_algorithm_capability_rows,
    happo_sequential_surrogate_loss,
    hasac_actor_alpha_loss,
    hasac_soft_critic_loss,
)
from vpp_dso_sim.learning.ctde_interface import build_ctde_interface_contract, validate_multi_agent_actions
from vpp_dso_sim.learning.hatrpo import HATRPOConfig, evaluate_hatrpo_checkpoint, train_hatrpo
from vpp_dso_sim.learning.reward_contracts import default_reward_contracts


def __getattr__(name: str):
    if name in {"MATD3Config", "evaluate_matd3_checkpoint", "train_matd3"}:
        matd3 = importlib.import_module("vpp_dso_sim.learning.matd3")
        return getattr(matd3, name)
    if name == "run_marl_baselines":
        baselines = importlib.import_module("vpp_dso_sim.learning.marl_baselines")
        return getattr(baselines, name)
    if name == "TrainingSupervisor":
        tuning = importlib.import_module("vpp_dso_sim.learning.tuning")
        return getattr(tuning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ALGORITHM_REGISTRY",
    "HAPPOConfig",
    "HATRPOConfig",
    "HASACConfig",
    "HASACReplayBuffer",
    "MATD3Config",
    "MultiHeadValueCriticSpec",
    "TrainingSupervisor",
    "TwinCriticSpec",
    "advanced_algorithm_capability_rows",
    "build_matd3_twin_critic_spec",
    "build_multi_head_value_critic",
    "build_agent_role_map",
    "build_ctde_interface_contract",
    "evaluate_happo_checkpoint",
    "evaluate_hatrpo_checkpoint",
    "evaluate_hasac_checkpoint",
    "evaluate_matd3_checkpoint",
    "default_reward_contracts",
    "get_algorithm_registry",
    "happo_sequential_surrogate_loss",
    "hasac_actor_alpha_loss",
    "hasac_soft_critic_loss",
    "rank_algorithm_candidates",
    "run_marl_baselines",
    "train_happo",
    "train_hatrpo",
    "train_hasac",
    "train_matd3",
    "validate_multi_agent_actions",
]
