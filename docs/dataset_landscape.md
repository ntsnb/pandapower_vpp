# Real-Training Dataset Landscape

This document records datasets that are suitable for moving the project from
demo profiles to real training experiments. The first open dataset has already
been downloaded into the local workspace.

## Downloaded Now

| Dataset | Local path | Files | Size | Status |
|---|---:|---:|---:|---|
| NREL SMART-DS Austin `AUS/P1U/base_timeseries/opendss` | `data/external/raw/smart_ds/v1.0/2018/AUS/P1U/base_timeseries/opendss` | 2369 | 127.059 MiB | downloaded, 0 failed |
| NREL SMART-DS Austin `AUS/P1U/profiles` | `data/external/raw/smart_ds/v1.0/2018/AUS/P1U/profiles` | 3021 | 1914.933 MiB | downloaded, 0 failed |

Detailed analysis has been exported to `outputs/smart_ds_analysis/`.

Key statistics from the downloaded subset:

- Primary feeder directories: 25
- Distribution-transformer / low-voltage portfolio directories: 93
- DSS files: 917
- Annual 15-minute profile CSV files: 3021
- Unique profile CSVs referenced by OpenDSS LoadShapes: 2912
- LoadShape profile references resolved locally: 62704 / 62704
- Approximate OpenDSS element definitions counted in feeder files:
  - loads: 127138
  - lines: 156683
  - transformers: 25350
- Main file types: `.dss`, `.csv`, `.txt`, `.png`

Download command:

```powershell
python scripts\download_open_datasets.py --dataset smart_ds_aus_p1u_base_opendss --output-root data\external\raw --workers 16
```

Generated audit files:

- `data/external/raw/smart_ds_aus_p1u_base_opendss_download_manifest.csv`
- `data/external/raw/smart_ds_aus_p1u_base_opendss_download_summary.json`
- `data/external/raw/smart_ds_aus_p1u_profiles_download_manifest.csv`
- `data/external/raw/smart_ds_aus_p1u_profiles_download_summary.json`

Analysis command:

```powershell
python examples\15_analyze_smart_ds_dataset.py --output-dir outputs\smart_ds_analysis
```

Analysis outputs:

- `outputs/smart_ds_analysis/smart_ds_summary.json`
- `outputs/smart_ds_analysis/smart_ds_feeders.csv`
- `outputs/smart_ds_analysis/smart_ds_suites.csv`
- `outputs/smart_ds_analysis/smart_ds_dataset_report.md`

`data/external/` is ignored by git because these files are local research data,
not source code.

## Source Fit Matrix

| Priority | Dataset | Best use in this project | Access reality |
|---|---|---|---|
| P0 | NREL SMART-DS | Real multi-feeder distribution topology, OpenDSS conversion, PV/storage scenarios, scale-up beyond IEEE toy feeders. | Open OEDI/S3 data; first subset downloaded. |
| P0 | SimBench | Pandapower-friendly benchmark grids and time series for European-style distribution studies. | Open package/data; good next adapter target. |
| P0 | IEEE PES Distribution Test Feeders | IEEE 123-node / standard feeder topology benchmarks. | Open feeder models, mostly static; needs external time series. |
| P0 | NREL End-Use Load Profiles | Residential/commercial load, HVAC, electrification and end-use profile calibration. | Open but large; needs extract pipeline. |
| P0 | ACN-Data | EVCS session arrival/departure/energy calibration. | Open API; site type must be preserved. |
| P1 | Pecan Street Dataport | PV, EV, HVAC and appliance-level behind-the-meter calibration. | Account/license gated, cannot be auto-downloaded without credentials. |
| P1 | Ausgrid Solar Home Electricity Data | Half-hourly PV/load profiles and reverse-flow scenarios. | Open; region/tariff context must be documented. |
| P1 | Low Carbon London | Smart meter and dynamic tariff response calibration. | Open data portal; trial-arm interpretation is required. |
| P1 | OpenEI Utility Rate Database | Retail tariff and TOU/demand-charge reward calibration. | Open API; tariff parsing is non-trivial. |
| P2 | CAISO OASIS | Day-ahead/real-time price traces for wholesale-facing experiments. | Open market portal/API; not a retail tariff. |
| P2 | ENTSO-E Transparency Platform | European market/system traces. | Requires registered token/API terms. |

The structured registry is implemented in:

- `src/vpp_dso_sim/data_sources/registry.py`
- `tests/test_dataset_registry.py`

## Next Adapters

1. `SMARTDSOpenDSSAdapter`: read `Master.dss`, feeder folders, loadshapes and
   bus coordinates; convert or bridge into pandapower with sign-convention tests.
2. `SimBenchScenarioAdapter`: use SimBench/pandapower-native data to create
   long train/eval/holdout profiles without modulo repetition.
3. `ACNDataEVCSAdapter`: transform EV charging sessions into EVCS arrivals,
   departure deadlines, target energy and charging power constraints.
4. `EULPProfileAdapter`: map building end-use profiles to residential,
   commercial, HVAC and flexible-load DER profiles.
5. `TariffAdapter`: parse OpenEI/CAISO/ENTSO-E prices into separate retail
   tariff, wholesale price and flexibility-service payment streams.

## Sources

- NREL SMART-DS: https://www.nrel.gov/grid/smart-ds
- OEDI SMART-DS S3 prefix used here: `https://oedi-data-lake.s3.amazonaws.com/?list-type=2&prefix=SMART-DS/v1.0/2018/AUS/P1U/scenarios/base_timeseries/opendss/`
- SimBench: https://simbench.de/en/
- IEEE PES Test Feeders: https://cmte.ieee.org/pes-testfeeders/resources/
- NREL End-Use Load Profiles: https://www.nrel.gov/buildings/end-use-load-profiles
- ACN-Data: https://ev.caltech.edu/dataset
- Pecan Street Dataport: https://www.pecanstreet.org/dataport/
- Ausgrid Solar Home Electricity Data: https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
- Low Carbon London smart meter data: https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households
- OpenEI Utility Rate Database: https://openei.org/wiki/Utility_Rate_Database
- CAISO OASIS: https://www.caiso.com/market-operations/oasis
- ENTSO-E Transparency Platform: https://transparency.entsoe.eu/
