from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class DataSourceRecord:
    """Structured metadata for datasets that can support real MARL training."""

    dataset_id: str
    name: str
    category: str
    url: str
    access: str
    temporal_resolution: str
    geographic_scope: str
    project_role: str
    integration_priority: str
    caveats: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def default_dataset_registry() -> list[DataSourceRecord]:
    """Return curated, source-linked datasets for DSO-VPP training.

    The registry is intentionally metadata-only. Most candidate datasets are
    too large, licensed, or account-gated for automated CI downloads, so data
    adapters should consume locally prepared extracts in future stages.
    """

    return [
        DataSourceRecord(
            dataset_id="nrel_smart_ds",
            name="NREL SMART-DS synthetic distribution systems",
            category="distribution_topology_and_scenarios",
            url="https://www.nrel.gov/grid/smart-ds",
            access="open metadata and downloadable scenario data",
            temporal_resolution="scenario dependent",
            geographic_scope="synthetic U.S. distribution systems",
            project_role="Large feeder topology, DER scenario scaling, OpenDSS-to-pandapower conversion targets.",
            integration_priority="P0",
            caveats="Synthetic networks still need conversion, validation, and pandapower sign-convention tests.",
        ),
        DataSourceRecord(
            dataset_id="simbench",
            name="SimBench benchmark grids",
            category="distribution_topology_and_time_series",
            url="https://simbench.de/en/",
            access="open benchmark package",
            temporal_resolution="time series benchmark profiles",
            geographic_scope="German representative transmission and distribution grids",
            project_role="Pandapower-friendly baseline for multi-voltage feeder studies and long time-series tests.",
            integration_priority="P0",
            caveats="Representative European data; economics and VPP settlement must be adapted to the study region.",
        ),
        DataSourceRecord(
            dataset_id="ieee_pes_test_feeders",
            name="IEEE PES Distribution Test Feeders",
            category="distribution_topology",
            url="https://cmte.ieee.org/pes-testfeeders/resources/",
            access="open feeder model files",
            temporal_resolution="static network models",
            geographic_scope="standard IEEE benchmark feeders",
            project_role="IEEE 123-node and LV feeder topology benchmarks for topology/generalization experiments.",
            integration_priority="P0",
            caveats="Static feeders need external profiles for load, PV, EV, and price processes.",
        ),
        DataSourceRecord(
            dataset_id="nrel_eulp",
            name="NREL End-Use Load Profiles",
            category="load_hvac_pv_ev_profiles",
            url="https://www.nrel.gov/buildings/end-use-load-profiles",
            access="open large-scale modeled profiles",
            temporal_resolution="subhourly/hourly profile products",
            geographic_scope="U.S. residential and commercial building stock",
            project_role="Residential/commercial load, HVAC end-use, electrification and DER profile generation.",
            integration_priority="P0",
            caveats="Modeled building profiles; feeder-node assignment and DER ownership synthesis are still required.",
        ),
        DataSourceRecord(
            dataset_id="acn_data",
            name="ACN-Data EV charging sessions",
            category="ev_charging_sessions",
            url="https://ev.caltech.edu/dataset",
            access="open API with attribution requirements",
            temporal_resolution="charging session and time-series records",
            geographic_scope="Caltech/JPL workplace charging sites",
            project_role="EVCS arrival/departure, charging demand and SOC/energy requirement calibration.",
            integration_priority="P0",
            caveats="Workplace charging is not the same as residential charging; scenario labels should preserve site type.",
        ),
        DataSourceRecord(
            dataset_id="pecan_street_dataport",
            name="Pecan Street Dataport",
            category="behind_meter_der_appliance_profiles",
            url="https://www.pecanstreet.org/dataport/",
            access="account/license gated",
            temporal_resolution="high-frequency residential meter and submeter data",
            geographic_scope="U.S. residential homes",
            project_role="PV, EV, HVAC and appliance-level calibration when licensed access is available.",
            integration_priority="P1",
            caveats="Not suitable for default open CI; use only with explicit access and documented license constraints.",
        ),
        DataSourceRecord(
            dataset_id="ausgrid_solar_home",
            name="Ausgrid Solar home electricity data",
            category="pv_and_household_load_profiles",
            url="https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data",
            access="open CSV-style data",
            temporal_resolution="half-hourly",
            geographic_scope="Australia residential solar homes",
            project_role="PV/load shape calibration and reverse-power-flow stress scenarios.",
            integration_priority="P1",
            caveats="Australian climate/tariff context; map units and seasons carefully before U.S./EU studies.",
        ),
        DataSourceRecord(
            dataset_id="low_carbon_london",
            name="Low Carbon London smart meter trial",
            category="smart_meter_load_and_tariff_response",
            url="https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households",
            access="open data portal",
            temporal_resolution="half-hourly",
            geographic_scope="London households",
            project_role="Demand-response and dynamic tariff behavior calibration.",
            integration_priority="P1",
            caveats="Anonymized trial data; household metadata and tariff-arm interpretation must be documented.",
        ),
        DataSourceRecord(
            dataset_id="openei_urdb",
            name="OpenEI Utility Rate Database",
            category="retail_tariffs",
            url="https://openei.org/wiki/Utility_Rate_Database",
            access="open API",
            temporal_resolution="tariff schedule dependent",
            geographic_scope="primarily U.S. utility tariffs",
            project_role="Retail tariff, TOU and demand-charge scenarios for VPP profit/reward accounting.",
            integration_priority="P1",
            caveats="Tariff structures can be complex; adapters must preserve demand charges and seasons.",
        ),
        DataSourceRecord(
            dataset_id="caiso_oasis",
            name="CAISO OASIS market data",
            category="wholesale_market_prices",
            url="https://www.caiso.com/market-operations/oasis",
            access="open market data portal/API",
            temporal_resolution="market product dependent",
            geographic_scope="California ISO market",
            project_role="Day-ahead/real-time price traces for service-price and settlement experiments.",
            integration_priority="P2",
            caveats="Wholesale prices are not retail tariffs; use only with explicit market-interface assumptions.",
        ),
        DataSourceRecord(
            dataset_id="entsoe_transparency",
            name="ENTSO-E Transparency Platform",
            category="wholesale_market_prices",
            url="https://transparency.entsoe.eu/",
            access="registered API/token access",
            temporal_resolution="market product dependent",
            geographic_scope="European power systems",
            project_role="European day-ahead price and system condition traces for sensitivity studies.",
            integration_priority="P2",
            caveats="Requires token/API terms; combine with European grid/profile data consistently.",
        ),
    ]


def dataset_registry_frame(records: list[DataSourceRecord] | None = None) -> pd.DataFrame:
    """Return the registry as a DataFrame for reports and CSV export."""

    selected = records if records is not None else default_dataset_registry()
    return pd.DataFrame([record.to_dict() for record in selected])
