from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class PPElementRef:
    element_type: Literal["sgen", "load", "storage"]
    element_index: int


@dataclass(frozen=True)
class ActionUnitId:
    action_unit_id: str
    vpp_id: str
    unit_type: Literal["vpp_pcc", "vpp_bus", "der"]
    pcc_id: str | None
    bus_id: int
    pp_element_refs: tuple[PPElementRef, ...]


@dataclass
class ActionUnitState:
    id: ActionUnitId
    p_cur_mw: float
    q_cur_mvar: float
    p_min_mw: float
    p_max_mw: float
    q_min_mvar: float
    q_max_mvar: float
    bid_up: float | None
    bid_down: float | None
    projection_gap_hist_mw: float
    q_control_available: bool

    def hard_width_mw(self) -> float:
        return float(max(0.0, self.p_max_mw - self.p_min_mw))


@dataclass(frozen=True)
class NetworkObjectId:
    object_id: str
    object_type: Literal["bus", "line", "trafo"]
    primary_id: int
    endpoint_bus_ids: tuple[int, ...]


@dataclass
class NetworkObjectState:
    id: NetworkObjectId
    value: float
    lower_limit: float | None
    upper_limit: float
    value_kind: Literal["vm_pu", "line_loading_percent", "trafo_loading_percent"]


@dataclass
class SensitivityEdgeTensor:
    """ActionUnit x NetworkObject sensitivity tensor.

    Shape contract:
    - `values`: float32 array `[K, A, C]`, where K is NetworkObject count, A is ActionUnit count.
    - `edge_valid_mask`: bool array `[K, A]`; false means the edge is not physically estimated.
    - `q_channel_mask`: false means Q channels are structurally present but disabled and zeroed.
    """

    values: np.ndarray
    channel_names: tuple[str, ...]
    edge_valid_mask: np.ndarray
    q_channel_mask: bool
    action_unit_ids: tuple[str, ...]
    network_object_ids: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class StructuredDSOObservation:
    """Structured DSO actor observation.

    Shape contract:
    - `global_features`: float32 `[G]`
    - `action_tokens`: float32 `[A_max, D_a]`
    - `object_tokens`: float32 `[K_max, D_k]`
    - `sensitivity_edges`: float32 `[K_max, A_max, C]`
    - `action_mask`: bool `[A_max]`; true entries are real ActionUnits.
    - `object_mask`: bool `[K_max]`; true entries are real NetworkObjects.
    - `edge_mask`: bool `[K_max, A_max]`; true entries are valid physical sensitivity edges.
    """

    global_features: np.ndarray
    action_tokens: np.ndarray
    object_tokens: np.ndarray
    sensitivity_edges: np.ndarray
    action_mask: np.ndarray
    object_mask: np.ndarray
    edge_mask: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodedOperatingEnvelopeRecord:
    action_unit_id: str
    vpp_id: str
    pcc_id: str | None
    bus_id: int
    p_hard_min_mw: float
    p_hard_max_mw: float
    p_pref_lo_mw: float
    p_pref_target_mw: float
    p_pref_hi_mw: float
    direction_probs: tuple[float, float, float]
    guidance_strength_lambda: float
    award_status: str = "envelope_guidance"
    source: str = "sensitivity_attention_v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
