from __future__ import annotations

import math

import torch
import torch.nn as nn


class _TokenEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BipartiteSensitivityDSOActor(nn.Module):
    """ActionUnit-to-NetworkObject edge-biased bipartite-attention actor."""

    def __init__(
        self,
        *,
        global_feature_dim: int,
        action_token_dim: int,
        object_token_dim: int,
        edge_feature_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        action_self_attention_layers: int = 1,
        dropout: float = 0.05,
        min_width_ratio: float = 0.10,
        max_width_ratio: float = 1.00,
    ) -> None:
        super().__init__()
        self.min_width_ratio = float(min_width_ratio)
        self.max_width_ratio = float(max_width_ratio)
        self.action_encoder = _TokenEncoder(action_token_dim, d_model, dropout)
        self.object_encoder = _TokenEncoder(object_token_dim, d_model, dropout)
        self.edge_encoder = _TokenEncoder(edge_feature_dim, d_model, dropout)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.edge_value = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Linear(d_model, 1)
        self.fusion_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model * 3, d_model),
                    nn.LayerNorm(d_model),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.self_attention = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=int(num_heads),
                    dim_feedforward=d_model * 2,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(max(0, int(action_self_attention_layers)))
            ]
        )
        self.center_head = nn.Linear(d_model, 1)
        self.width_head = nn.Linear(d_model, 1)
        self.direction_head = nn.Linear(d_model, 3)
        self.lambda_head = nn.Linear(d_model, 1)

    def forward(
        self,
        *,
        global_features: torch.Tensor,
        action_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        sensitivity_edges: torch.Tensor,
        action_mask: torch.Tensor,
        object_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        action_latent = self.action_encoder(action_tokens)
        object_latent = self.object_encoder(object_tokens)
        edge_latent = self.edge_encoder(sensitivity_edges)
        query = self.q_proj(action_latent)
        key = self.k_proj(object_latent)
        value = self.v_proj(object_latent)
        scores = torch.einsum("bad,bkd->bak", query, key) / math.sqrt(query.shape[-1])
        scores = scores + self.edge_bias(edge_latent).squeeze(-1).transpose(1, 2)
        valid_edges = object_mask.unsqueeze(1).bool() & edge_mask.transpose(1, 2).bool()
        scores = scores.masked_fill(~valid_edges, -1e9)
        attention = torch.softmax(scores, dim=-1)
        edge_context_values = value.unsqueeze(1) + self.edge_value(edge_latent).permute(0, 2, 1, 3)
        context = torch.sum(attention.unsqueeze(-1) * edge_context_values, dim=2)
        global_latent = self.global_encoder(global_features).unsqueeze(1).expand_as(action_latent)
        fused = action_latent
        for layer in self.fusion_layers:
            fused = layer(torch.cat([fused, context, global_latent], dim=-1))
        key_padding_mask = ~action_mask.bool()
        for layer in self.self_attention:
            fused = layer(fused, src_key_padding_mask=key_padding_mask)

        center_ratio = torch.sigmoid(self.center_head(fused).squeeze(-1))
        width_unit = torch.sigmoid(self.width_head(fused).squeeze(-1))
        width_ratio = self.min_width_ratio + (self.max_width_ratio - self.min_width_ratio) * width_unit
        direction_probs = torch.softmax(self.direction_head(fused), dim=-1)
        guidance_strength = torch.sigmoid(self.lambda_head(fused).squeeze(-1))
        return {
            "center_ratio": center_ratio,
            "width_ratio": width_ratio,
            "direction_probs": direction_probs,
            "guidance_strength": guidance_strength,
            "attention_weights": attention,
        }
