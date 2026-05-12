from __future__ import annotations

import base64
from functools import lru_cache

DER_SHORT_LABELS = {
    "PVModel": "PV",
    "MicroTurbineModel": "MT",
    "StorageModel": "ESS",
    "FlexibleLoadModel": "Flex",
    "HVACModel": "HVAC",
    "EVCSModel": "EVCS",
}

DER_ICON_NAMES = {
    "PVModel": "solar panel",
    "MicroTurbineModel": "microturbine",
    "StorageModel": "battery storage",
    "FlexibleLoadModel": "flexible load",
    "HVACModel": "HVAC fan",
    "EVCSModel": "charging station",
}


def der_short_label(der_type: str) -> str:
    return DER_SHORT_LABELS.get(str(der_type), str(der_type).replace("Model", ""))


def der_icon_name(der_type: str) -> str:
    return DER_ICON_NAMES.get(str(der_type), der_short_label(der_type))


@lru_cache(maxsize=128)
def der_icon_data_uri(der_type: str, accent_color: str = "#2563eb") -> str:
    svg = _icon_svg(str(der_type), accent_color)
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


@lru_cache(maxsize=32)
def grid_source_icon_data_uri(accent_color: str = "#0f172a") -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" '
        'viewBox="0 0 72 72">'
        '<rect x="4" y="4" width="64" height="64" rx="12" fill="#ffffff" '
        f'stroke="{accent_color}" stroke-width="4"/>'
        f'<circle cx="36" cy="29" r="16" fill="#eff6ff" stroke="{accent_color}" stroke-width="3"/>'
        f'<path d="M22 29 C27 17 32 41 37 29 C42 17 47 41 52 29" fill="none" '
        f'stroke="{accent_color}" stroke-width="3" stroke-linecap="round"/>'
        f'<path d="M36 45 V60 M24 60 H48" stroke="{accent_color}" stroke-width="4" stroke-linecap="round"/>'
        '</svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


@lru_cache(maxsize=64)
def pcc_switch_icon_data_uri(accent_color: str = "#2563eb") -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" '
        'viewBox="0 0 72 72">'
        '<rect x="5" y="5" width="62" height="62" rx="10" fill="#ffffff" '
        f'stroke="{accent_color}" stroke-width="4"/>'
        f'<path d="M16 23 H56 M16 49 H56" stroke="{accent_color}" stroke-width="5" stroke-linecap="round"/>'
        '<circle cx="29" cy="36" r="5" fill="#ffffff" stroke="#0f172a" stroke-width="3"/>'
        '<circle cx="47" cy="36" r="5" fill="#ffffff" stroke="#0f172a" stroke-width="3"/>'
        '<path d="M31 35 L45 28" stroke="#0f172a" stroke-width="4" stroke-linecap="round"/>'
        '<path d="M18 36 H24 M52 36 H58" stroke="#0f172a" stroke-width="4" stroke-linecap="round"/>'
        '</svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _icon_svg(der_type: str, accent_color: str) -> str:
    accent = accent_color or "#2563eb"
    body = {
        "PVModel": _pv_svg,
        "EVCSModel": _evcs_svg,
        "StorageModel": _storage_svg,
        "HVACModel": _hvac_svg,
        "MicroTurbineModel": _microturbine_svg,
        "FlexibleLoadModel": _flexible_load_svg,
    }.get(der_type, _generic_der_svg)(accent)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" '
        'viewBox="0 0 72 72">'
        f'<rect x="3" y="3" width="66" height="66" rx="12" fill="#ffffff" '
        f'stroke="{accent}" stroke-width="4"/>'
        f"{body}"
        "</svg>"
    )


def _pv_svg(accent: str) -> str:
    return f"""
<circle cx="53" cy="17" r="8" fill="#facc15" stroke="#ca8a04" stroke-width="2"/>
<g stroke="#ca8a04" stroke-width="2" stroke-linecap="round">
  <line x1="53" y1="4" x2="53" y2="8"/>
  <line x1="53" y1="26" x2="53" y2="30"/>
  <line x1="40" y1="17" x2="44" y2="17"/>
  <line x1="62" y1="17" x2="66" y2="17"/>
</g>
<polygon points="13,31 47,27 55,51 19,56" fill="#1d4ed8" stroke="#0f172a" stroke-width="2"/>
<g stroke="#93c5fd" stroke-width="1.6">
  <line x1="21" y1="30" x2="27" y2="55"/>
  <line x1="34" y1="29" x2="39" y2="53"/>
  <line x1="47" y1="30" x2="51" y2="50"/>
  <line x1="16" y1="39" x2="50" y2="35"/>
  <line x1="18" y1="48" x2="53" y2="44"/>
</g>
<path d="M36 54 L36 63" stroke="{accent}" stroke-width="4" stroke-linecap="round"/>
"""


def _evcs_svg(accent: str) -> str:
    return f"""
<rect x="17" y="13" width="28" height="47" rx="6" fill="#e0f2fe" stroke="#0f172a" stroke-width="2"/>
<rect x="23" y="20" width="16" height="11" rx="2" fill="#38bdf8" stroke="#0369a1" stroke-width="1.6"/>
<path d="M32 35 L27 45 H33 L29 55 L41 40 H34 Z" fill="#facc15" stroke="#a16207" stroke-width="1.3"/>
<path d="M45 24 C58 25 58 40 51 45" fill="none" stroke="{accent}" stroke-width="4" stroke-linecap="round"/>
<rect x="48" y="42" width="8" height="12" rx="2" fill="#111827"/>
<line x1="50" y1="39" x2="50" y2="43" stroke="#111827" stroke-width="2"/>
<line x1="54" y1="39" x2="54" y2="43" stroke="#111827" stroke-width="2"/>
<rect x="13" y="59" width="36" height="4" rx="2" fill="{accent}"/>
"""


def _storage_svg(accent: str) -> str:
    return f"""
<rect x="12" y="25" width="43" height="25" rx="5" fill="#dcfce7" stroke="#0f172a" stroke-width="2.4"/>
<rect x="55" y="32" width="5" height="11" rx="2" fill="#0f172a"/>
<rect x="18" y="31" width="24" height="13" rx="3" fill="{accent}"/>
<line x1="25" y1="15" x2="25" y2="23" stroke="#16a34a" stroke-width="3" stroke-linecap="round"/>
<line x1="41" y1="15" x2="41" y2="23" stroke="#16a34a" stroke-width="3" stroke-linecap="round"/>
<path d="M24 58 H46" stroke="#16a34a" stroke-width="4" stroke-linecap="round"/>
"""


def _hvac_svg(accent: str) -> str:
    return f"""
<circle cx="36" cy="36" r="23" fill="#ecfeff" stroke="#0f172a" stroke-width="2.4"/>
<circle cx="36" cy="36" r="5" fill="{accent}" stroke="#0f172a" stroke-width="1.5"/>
<path d="M36 31 C42 18 55 22 51 34 C45 33 40 32 36 31 Z" fill="#67e8f9" stroke="#0891b2" stroke-width="1.5"/>
<path d="M40 39 C54 42 55 56 43 57 C42 51 41 44 40 39 Z" fill="#67e8f9" stroke="#0891b2" stroke-width="1.5"/>
<path d="M32 38 C22 49 10 42 16 31 C21 34 27 37 32 38 Z" fill="#67e8f9" stroke="#0891b2" stroke-width="1.5"/>
<path d="M17 17 L24 24 M55 17 L48 24 M17 55 L24 48 M55 55 L48 48" stroke="{accent}" stroke-width="2.2" stroke-linecap="round"/>
"""


def _microturbine_svg(accent: str) -> str:
    return f"""
<path d="M36 33 L29 61 H43 L36 33 Z" fill="#e5e7eb" stroke="#0f172a" stroke-width="2"/>
<circle cx="36" cy="29" r="6" fill="{accent}" stroke="#0f172a" stroke-width="2"/>
<path d="M36 23 C38 11 51 10 55 18 C48 20 42 22 36 29 Z" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.6"/>
<path d="M42 31 C55 36 54 49 46 54 C44 46 41 38 36 29 Z" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.6"/>
<path d="M31 31 C19 36 10 27 14 18 C20 22 27 25 36 29 Z" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.6"/>
<path d="M16 62 H56" stroke="{accent}" stroke-width="4" stroke-linecap="round"/>
"""


def _flexible_load_svg(accent: str) -> str:
    return f"""
<path d="M15 34 L36 17 L57 34 V59 H15 Z" fill="#fef3c7" stroke="#0f172a" stroke-width="2.3"/>
<rect x="25" y="42" width="11" height="17" rx="2" fill="#fdba74" stroke="#9a3412" stroke-width="1.5"/>
<rect x="42" y="39" width="8" height="8" rx="1" fill="#bae6fd" stroke="#0369a1" stroke-width="1.3"/>
<g stroke="{accent}" stroke-width="3" stroke-linecap="round">
  <line x1="20" y1="27" x2="28" y2="27"/>
  <line x1="44" y1="27" x2="52" y2="27"/>
  <line x1="24" y1="31" x2="48" y2="31"/>
</g>
<circle cx="36" cy="31" r="4" fill="{accent}"/>
"""


def _generic_der_svg(accent: str) -> str:
    return f"""
<circle cx="36" cy="35" r="19" fill="#f1f5f9" stroke="#0f172a" stroke-width="2.2"/>
<path d="M36 18 V52 M22 35 H50" stroke="{accent}" stroke-width="5" stroke-linecap="round"/>
"""
