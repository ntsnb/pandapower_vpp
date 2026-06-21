from __future__ import annotations

from vpp_dso_sim.dso.envelope.safe_decoder import decode_operating_envelope


def test_safe_decoder_preserves_hard_bounds_and_guidance_status() -> None:
    records = decode_operating_envelope(
        action_unit_ids=["au_0", "au_1"],
        vpp_ids=["vpp_a", "vpp_b"],
        pcc_ids=["pcc_1", None],
        bus_ids=[1, 2],
        p_hard_min_mw=[-1.0, 0.2],
        p_hard_max_mw=[1.0, 0.2],
        center_ratio=[0.75, 0.5],
        width_ratio=[0.25, 0.9],
        direction_probs=[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
        guidance_strength=[0.8, 0.1],
    )

    assert len(records) == 2
    for record in records:
        assert record.p_hard_min_mw <= record.p_pref_lo_mw
        assert record.p_pref_lo_mw <= record.p_pref_target_mw
        assert record.p_pref_target_mw <= record.p_pref_hi_mw
        assert record.p_pref_hi_mw <= record.p_hard_max_mw
        assert record.award_status == "envelope_guidance"
        assert record.source == "sensitivity_attention_v1"
