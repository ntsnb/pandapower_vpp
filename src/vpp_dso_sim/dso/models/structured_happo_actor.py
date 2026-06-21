from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from vpp_dso_sim.dso.models.bipartite_attention_actor import BipartiteSensitivityDSOActor
from vpp_dso_sim.dso.observation.happo_structured import StructuredDSOFlatSpec


ENVELOPE_ACTION_CHANNELS = (
    "center_ratio",
    "width_ratio",
    "guidance_strength",
    "direction_absorb_logit",
    "direction_balanced_logit",
    "direction_inject_logit",
)


def structured_envelope_action_dim(spec: StructuredDSOFlatSpec) -> int:
    return int(spec.max_action_units * len(ENVELOPE_ACTION_CHANNELS))


def normalized_envelope_action_to_payload(
    action: np.ndarray | list[float] | tuple[float, ...],
    spec: StructuredDSOFlatSpec,
    *,
    action_clip: float = 1.0,
    min_width_ratio: float = 0.10,
    max_width_ratio: float = 1.00,
    direction_logit_scale: float = 5.0,
    source: str = "sensitivity_attention_v1_structured_actor",
) -> dict[str, object]:
    """Convert a fixed HAPPO Gaussian action into DSO envelope parameters."""

    channel_count = len(ENVELOPE_ACTION_CHANNELS)
    scale = max(1e-6, float(action_clip))
    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = structured_envelope_action_dim(spec)
    if raw.size < expected:
        raw = np.pad(raw, (0, expected - raw.size), mode="constant")
    raw = np.clip(raw[:expected] / scale, -1.0, 1.0).reshape(spec.max_action_units, channel_count)
    width_unit = 0.5 * (raw[:, 1] + 1.0)
    width_span = max(0.0, float(max_width_ratio) - float(min_width_ratio))
    return {
        "source": source,
        "action_unit_ids": list(spec.action_unit_ids),
        "center_ratio": [float(0.5 * (value + 1.0)) for value in raw[:, 0]],
        "width_ratio": [float(float(min_width_ratio) + width_span * value) for value in width_unit],
        "guidance_strength": [float(0.5 * (value + 1.0)) for value in raw[:, 2]],
        "direction_logits": [
            [float(component * float(direction_logit_scale)) for component in row]
            for row in raw[:, 3:6]
        ],
        "channel_order": list(ENVELOPE_ACTION_CHANNELS),
        "action_clip": float(action_clip),
    }


class StructuredDSOGaussianActor(nn.Module):
    """HAPPO-compatible Gaussian actor backed by structured DSO attention."""

    def __init__(
        self,
        *,
        spec: StructuredDSOFlatSpec,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        action_self_attention_layers: int = 1,
        dropout: float = 0.0,
        min_width_ratio: float = 0.10,
        max_width_ratio: float = 1.00,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.vpp_count = len(spec.vpp_ids)
        self.envelope_channels = ENVELOPE_ACTION_CHANNELS
        self.envelope_action_dim = structured_envelope_action_dim(spec)
        self.min_width_ratio = float(min_width_ratio)
        self.max_width_ratio = float(max_width_ratio)
        self.attention_actor = BipartiteSensitivityDSOActor(
            global_feature_dim=spec.global_dim,
            action_token_dim=spec.action_token_dim,
            object_token_dim=spec.object_token_dim,
            edge_feature_dim=spec.edge_feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            action_self_attention_layers=action_self_attention_layers,
            dropout=dropout,
            min_width_ratio=min_width_ratio,
            max_width_ratio=max_width_ratio,
        )
        self.log_std = nn.Parameter(torch.full((self.envelope_action_dim,), -0.7))
        self.register_buffer(
            "action_unit_vpp_indices",
            torch.tensor(spec.action_unit_vpp_indices, dtype=torch.long),
            persistent=False,
        )

    def _unflatten(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        spec = self.spec
        offset = 0
        global_features = x[:, offset : offset + spec.global_dim]
        offset += spec.global_dim
        action_size = spec.max_action_units * spec.action_token_dim
        action_tokens = x[:, offset : offset + action_size].reshape(
            x.shape[0],
            spec.max_action_units,
            spec.action_token_dim,
        )
        offset += action_size
        object_size = spec.max_network_objects * spec.object_token_dim
        object_tokens = x[:, offset : offset + object_size].reshape(
            x.shape[0],
            spec.max_network_objects,
            spec.object_token_dim,
        )
        offset += object_size
        edge_size = spec.max_network_objects * spec.max_action_units * spec.edge_feature_dim
        sensitivity_edges = x[:, offset : offset + edge_size].reshape(
            x.shape[0],
            spec.max_network_objects,
            spec.max_action_units,
            spec.edge_feature_dim,
        )
        offset += edge_size
        action_mask = x[:, offset : offset + spec.max_action_units] > 0.5
        offset += spec.max_action_units
        object_mask = x[:, offset : offset + spec.max_network_objects] > 0.5
        offset += spec.max_network_objects
        edge_mask = x[:, offset : offset + spec.max_network_objects * spec.max_action_units].reshape(
            x.shape[0],
            spec.max_network_objects,
            spec.max_action_units,
        ) > 0.5
        return {
            "global_features": global_features,
            "action_tokens": action_tokens,
            "object_tokens": object_tokens,
            "sensitivity_edges": sensitivity_edges,
            "action_mask": action_mask,
            "object_mask": object_mask,
            "edge_mask": edge_mask,
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self._unflatten(x)
        outputs = self.attention_actor(**tensors)
        per_unit = torch.cat(
            [
                2.0 * outputs["center_ratio"].unsqueeze(-1) - 1.0,
                2.0
                * (
                    (outputs["width_ratio"].unsqueeze(-1) - self.min_width_ratio)
                    / max(1e-6, self.max_width_ratio - self.min_width_ratio)
                )
                - 1.0,
                2.0 * outputs["guidance_strength"].unsqueeze(-1) - 1.0,
                2.0 * outputs["direction_probs"] - 1.0,
            ],
            dim=-1,
        )
        action_mask = tensors["action_mask"].unsqueeze(-1).float()
        mean = torch.clamp(per_unit * action_mask, -1.0, 1.0).reshape(x.shape[0], self.envelope_action_dim)
        return mean, self.log_std.expand_as(mean)
