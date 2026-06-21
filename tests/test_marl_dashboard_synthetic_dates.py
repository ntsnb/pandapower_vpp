from __future__ import annotations

from marl_dashboard.backend.storage.query_service import QueryService
from marl_dashboard.logging import ExperimentLogger


def test_query_service_synthesizes_profile_dates_for_long_horizon_rows_without_dates(tmp_path):
    logger = ExperimentLogger(run_id="run_profile_days", data_dir=tmp_path, config={}, async_writer=False)
    for absolute_step in (0, 95, 96, 191):
        logger.log_dataset(
            epoch_id=0,
            episode_id=1,
            env_id="env_0",
            vpp_id="vpp_001",
            agent_id="vpp_001_dispatch",
            policy_id="dispatch",
            date=None,
            time_index=absolute_step,
            global_env_step=absolute_step,
            values={"electricity_price": float(absolute_step)},
            units={"electricity_price": "currency/MWh"},
        )
    logger.close()

    service = QueryService(tmp_path)
    selectors = service.selectors("run_profile_days")

    assert selectors["dates"] == ["profile_day_001", "profile_day_002"]
    assert selectors["time_indices"][0] == 0
    assert selectors["time_indices"][-1] == 95

    day_two = service.query_metric_table(
        run_id="run_profile_days",
        table="dataset_timeseries",
        metrics="electricity_price",
        date="profile_day_002",
        start_time_index=0,
        end_time_index=0,
    )

    assert day_two["summary"]["row_count"] == 1
    assert day_two["table_rows"][0]["date"] == "profile_day_002"
    assert day_two["table_rows"][0]["time_index"] == 0
    assert day_two["table_rows"][0]["global_env_step"] == 96
    assert day_two["table_rows"][0]["value"] == 96.0


def test_query_service_does_not_create_extra_profile_day_from_terminal_boundary_step(tmp_path):
    logger = ExperimentLogger(run_id="run_terminal_boundary", data_dir=tmp_path, config={}, async_writer=False)
    for absolute_step in (0, 95, 96, 191):
        logger.log_dataset(
            epoch_id=0,
            episode_id=1,
            env_id="env_0",
            vpp_id="vpp_001",
            agent_id="vpp_001_dispatch",
            policy_id="dispatch",
            date=None,
            time_index=absolute_step,
            global_env_step=absolute_step,
            values={"electricity_price": float(absolute_step)},
            units={"electricity_price": "currency/MWh"},
        )
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="aggregate",
        agent_id=None,
        policy_id=None,
        date=None,
        time_index=192,
        global_env_step=192,
        values={"progress_rows": 200.0},
        units={"progress_rows": "count"},
    )
    logger.close()

    service = QueryService(tmp_path)
    selectors = service.selectors("run_terminal_boundary")

    assert selectors["dates"] == ["profile_day_001", "profile_day_002"]

    invalid_day = service.query_metric_table(
        run_id="run_terminal_boundary",
        table="dataset_timeseries",
        date="profile_day_003",
    )

    assert invalid_day["summary"]["row_count"] == 0
    assert invalid_day["table_rows"] == []
