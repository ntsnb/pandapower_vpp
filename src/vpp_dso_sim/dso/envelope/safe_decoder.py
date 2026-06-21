from __future__ import annotations

from collections.abc import Sequence

from vpp_dso_sim.dso.envelope.schemas import DecodedOperatingEnvelopeRecord


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def decode_operating_envelope(
    *,
    action_unit_ids: Sequence[str],
    vpp_ids: Sequence[str],
    pcc_ids: Sequence[str | None],
    bus_ids: Sequence[int],
    p_hard_min_mw: Sequence[float],
    p_hard_max_mw: Sequence[float],
    center_ratio: Sequence[float],
    width_ratio: Sequence[float],
    direction_probs: Sequence[Sequence[float]],
    guidance_strength: Sequence[float],
) -> list[DecodedOperatingEnvelopeRecord]:
    """Decode actor center/width outputs inside FR/DOE hard bounds."""

    records: list[DecodedOperatingEnvelopeRecord] = []
    for index, action_unit_id in enumerate(action_unit_ids):
        p_min = float(p_hard_min_mw[index])
        p_max = float(p_hard_max_mw[index])
        if p_min > p_max:
            midpoint = 0.5 * (p_min + p_max)
            p_min = p_max = midpoint
        width_hard = max(0.0, p_max - p_min)
        center = _clip01(float(center_ratio[index]))
        width = _clip01(float(width_ratio[index]))
        target = p_min + center * width_hard
        delta = 0.5 * width * width_hard
        lo = max(p_min, target - delta)
        hi = min(p_max, target + delta)
        target = max(lo, min(hi, target))
        probs = tuple(float(value) for value in list(direction_probs[index])[:3])
        if len(probs) != 3:
            probs = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        records.append(
            DecodedOperatingEnvelopeRecord(
                action_unit_id=str(action_unit_id),
                vpp_id=str(vpp_ids[index]),
                pcc_id=pcc_ids[index],
                bus_id=int(bus_ids[index]),
                p_hard_min_mw=float(p_min),
                p_hard_max_mw=float(p_max),
                p_pref_lo_mw=float(lo),
                p_pref_target_mw=float(target),
                p_pref_hi_mw=float(hi),
                direction_probs=probs,  # type: ignore[arg-type]
                guidance_strength_lambda=_clip01(float(guidance_strength[index])),
            )
        )
    return records
