from __future__ import annotations

from typing import Any

import numpy as np


VPP_DISPATCH_CONTEXT_DIM = 16
VPP_DISPATCH_TOKEN_DIM = 26
JOINT_ACTION_GLOBAL_FEATURES = 16
JOINT_ACTION_FEATURES_PER_VPP = 8


def split_vpp_dispatch_vector(
    vector: np.ndarray | list[float],
    *,
    context_dim: int = VPP_DISPATCH_CONTEXT_DIM,
    token_dim: int = VPP_DISPATCH_TOKEN_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[-1] < context_dim:
        raise ValueError(
            f"Expected at least {context_dim} VPP context features, got {array.shape[-1]}."
        )
    token_width = array.shape[-1] - context_dim
    if token_width % token_dim != 0:
        raise ValueError(
            f"Expected padded DER token width to be divisible by {token_dim}, got remainder "
            f"{token_width % token_dim}."
        )
    context = array[:, :context_dim]
    token_count = token_width // token_dim
    tokens = array[:, context_dim:].reshape(array.shape[0], token_count, token_dim)
    mask = np.any(np.abs(tokens) > 1e-9, axis=-1)
    return context, tokens, mask


def _summary_stats(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(array.mean()),
        float(array.std()),
        float(array.min()),
        float(array.max()),
    ]


def joint_action_summary_dim(num_vpps: int) -> int:
    return JOINT_ACTION_GLOBAL_FEATURES + JOINT_ACTION_FEATURES_PER_VPP * max(1, int(num_vpps))


def encode_joint_action_summary(
    *,
    vpp_ids: list[str],
    dso_action: np.ndarray | list[float],
    vpp_aggregate_actions: dict[str, float],
    der_actions_by_vpp: dict[str, np.ndarray | list[float]],
    portfolio_action_indices: dict[str, int],
) -> np.ndarray:
    ordered_vpp_ids = list(vpp_ids)
    dso = np.asarray(dso_action, dtype=np.float32).reshape(-1)
    aggregate = np.asarray(
        [float(vpp_aggregate_actions.get(vpp_id, 0.0)) for vpp_id in ordered_vpp_ids],
        dtype=np.float32,
    )
    portfolio = np.asarray(
        [float(portfolio_action_indices.get(vpp_id, 0)) / 2.0 for vpp_id in ordered_vpp_ids],
        dtype=np.float32,
    )
    der_vectors = {
        vpp_id: np.asarray(der_actions_by_vpp.get(vpp_id, []), dtype=np.float32).reshape(-1)
        for vpp_id in ordered_vpp_ids
    }
    all_der = (
        np.concatenate([values for values in der_vectors.values() if values.size > 0], dtype=np.float32)
        if any(values.size > 0 for values in der_vectors.values())
        else np.zeros(0, dtype=np.float32)
    )
    keep_rate = float((portfolio <= 0.0).mean()) if portfolio.size else 0.0
    reweight_rate = float(((portfolio > 0.0) & (portfolio < 1.0)).mean()) if portfolio.size else 0.0
    change_rate = float((portfolio >= 1.0).mean()) if portfolio.size else 0.0
    max_der_count = max((values.size for values in der_vectors.values()), default=1)
    features: list[float] = [
        *_summary_stats(dso),
        *_summary_stats(aggregate),
        *_summary_stats(all_der),
        keep_rate,
        reweight_rate,
        change_rate,
        float(len(ordered_vpp_ids)),
    ]
    for index, vpp_id in enumerate(ordered_vpp_ids):
        der_values = der_vectors[vpp_id]
        features.extend(
            [
                float(dso[index]) if index < dso.size else 0.0,
                float(vpp_aggregate_actions.get(vpp_id, 0.0)),
                float(portfolio_action_indices.get(vpp_id, 0)) / 2.0,
                float(der_values.mean()) if der_values.size else 0.0,
                float(der_values.std()) if der_values.size else 0.0,
                float(der_values.min()) if der_values.size else 0.0,
                float(der_values.max()) if der_values.size else 0.0,
                float(der_values.size) / max(1.0, float(max_der_count)),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def build_privacy_separated_ctde_modules(
    *,
    torch: Any,
    nn: Any,
    dso_input_dim: int,
    vpp_input_dim: int,
    portfolio_input_dim: int,
    critic_input_dim: int,
    action_summary_dim: int,
    action_dim: int,
    der_action_dim: int,
    hidden_dim: int,
) -> tuple[Any, dict[str, Any]]:
    class MLPEncoder(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.net(x)

    class DeepSetDispatchEncoder(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            token_width = max(0, input_dim - VPP_DISPATCH_CONTEXT_DIM)
            token_count = token_width // VPP_DISPATCH_TOKEN_DIM
            self.context_dim = VPP_DISPATCH_CONTEXT_DIM
            self.token_dim = VPP_DISPATCH_TOKEN_DIM
            self.token_count = token_count
            self.context_norm = nn.LayerNorm(self.context_dim)
            self.context_mlp = nn.Sequential(
                nn.Linear(self.context_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.token_norm = nn.LayerNorm(self.token_dim)
            self.token_mlp = nn.Sequential(
                nn.Linear(self.token_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3 + 1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            context = x[:, : self.context_dim]
            token_flat = x[:, self.context_dim :]
            tokens = token_flat.reshape(x.shape[0], self.token_count, self.token_dim)
            mask = tokens.abs().sum(dim=-1) > 1e-9
            context_latent = self.context_mlp(self.context_norm(context))
            token_latent = self.token_mlp(self.token_norm(tokens))
            mask_f = mask.unsqueeze(-1).float()
            token_count = mask_f.sum(dim=1).clamp(min=1.0)
            token_mean = (token_latent * mask_f).sum(dim=1) / token_count
            masked_max = token_latent.masked_fill(~mask.unsqueeze(-1), -1e9)
            token_max = masked_max.max(dim=1).values
            token_max = torch.where(mask.any(dim=1, keepdim=True), token_max, torch.zeros_like(token_max))
            count_ratio = token_count / max(1.0, float(self.token_count))
            fused = torch.cat([context_latent, token_mean, token_max, count_ratio], dim=-1)
            return self.fusion(fused)

    class DSOActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = MLPEncoder(dso_input_dim)
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

        def forward(self, x):
            latent = self.encoder(x)
            mean = torch.tanh(self.mean(latent))
            return mean, self.log_std.expand_as(mean)

    class VPPDispatchActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = DeepSetDispatchEncoder(vpp_input_dim)
            self.aggregate_mean = nn.Linear(hidden_dim, 1)
            self.aggregate_log_std = nn.Parameter(torch.full((1,), -0.8))
            self.der_mean = nn.Linear(hidden_dim, der_action_dim)
            self.der_log_std = nn.Parameter(torch.full((der_action_dim,), -0.8))

        def forward(self, x):
            latent = self.encoder(x)
            aggregate_mean = torch.tanh(self.aggregate_mean(latent))
            der_mean = torch.tanh(self.der_mean(latent))
            return (
                aggregate_mean,
                self.aggregate_log_std.expand_as(aggregate_mean),
                der_mean,
                self.der_log_std.expand_as(der_mean),
            )

    class PortfolioActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = MLPEncoder(portfolio_input_dim)
            self.logits = nn.Linear(hidden_dim, 3)

        def forward(self, x):
            return self.logits(self.encoder(x))

    class CentralizedActionConditionedCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = MLPEncoder(critic_input_dim)
            self.action_encoder = MLPEncoder(action_summary_dim)
            self.value_trunk = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
            )
            self.dso_value = nn.Linear(hidden_dim, 1)
            self.dispatch_value = nn.Linear(hidden_dim, 1)
            self.portfolio_value = nn.Linear(hidden_dim, 1)

        def _latent(self, state, action_summary):
            state_latent = self.state_encoder(state)
            action_latent = self.action_encoder(action_summary)
            fused = torch.cat([state_latent, action_latent], dim=-1)
            return self.value_trunk(fused)

        def forward_heads(self, state, action_summary):
            latent = self._latent(state, action_summary)
            return {
                "dso": self.dso_value(latent).squeeze(-1),
                "dispatch": self.dispatch_value(latent).squeeze(-1),
                "portfolio": self.portfolio_value(latent).squeeze(-1),
            }

        def forward(self, state, action_summary):
            return self.forward_heads(state, action_summary)["dso"]

    modules = nn.ModuleDict(
        {
            "dso_actor": DSOActor(),
            "vpp_dispatch_actor": VPPDispatchActor(),
            "vpp_portfolio_actor": PortfolioActor(),
            "centralized_critic": CentralizedActionConditionedCritic(),
        }
    )
    metadata = {
        "architecture_version": "ctde_v2_deepsets_action_conditioned",
        "vpp_encoder_type": "deep_sets_shared_token_mlp",
        "critic_type": "centralized_action_conditioned_summary_critic",
        "critic_head_type": "role_multi_head_value_baselines",
        "critic_value_heads": "dso,dispatch,portfolio",
        "mappo_happo_lite_ready": True,
        "action_conditioned_critic": True,
        "vpp_dispatch_context_dim": VPP_DISPATCH_CONTEXT_DIM,
        "vpp_dispatch_token_dim": VPP_DISPATCH_TOKEN_DIM,
        "critic_action_summary_dim": action_summary_dim,
        "critic_action_summary_type": "global_stats_plus_per_vpp_action_tokens",
        "vpp_dispatch_token_fields": (
            "type_one_hot_6,bus,p_bounds_2,q_bounds_2,controllable,cost_3,current_pq_2,"
            "available_p,soc_triplet,average_soc,hvac_temp_triplet,comfort_penalty"
        ),
    }
    return modules, metadata
