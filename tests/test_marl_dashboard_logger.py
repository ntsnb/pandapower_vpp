from __future__ import annotations

import json
import time

from marl_dashboard.logging import ExperimentLogger
from marl_dashboard.backend.storage.parquet_writer import compact_partition
from marl_dashboard.backend.storage.query_service import QueryService
from marl_dashboard.demo.generate_demo_run import default_variable_dictionary


def test_experiment_logger_writes_partitioned_tables_and_metadata(tmp_path):
    logger = ExperimentLogger(
        run_id="run_test",
        data_dir=tmp_path,
        config={"algorithm": "happo", "environment": "multi_agent_vpp"},
        variable_dictionary=[
            {
                "name": "electricity_price",
                "display_name": "Electricity price",
                "symbol": "c_t",
                "unit": "$/MWh",
                "group": "dataset",
                "physical_meaning": "Market price at time t.",
                "formula_latex": "c_t",
                "source": "test",
            }
        ],
        formulas={"total_reward": "r_t = profit_t - penalty_t"},
        async_writer=False,
    )

    logger.log_dataset(
        epoch_id=1,
        episode_id=2,
        env_id="env_0",
        vpp_id="vpp_001",
        agent_id="vpp_001_dispatch",
        policy_id="dispatch_shared",
        date="2026-01-01",
        time_index=3,
        timestamp="2026-01-01T00:45:00",
        values={"electricity_price": 88.5, "storage_soc": 54.0},
        units={"electricity_price": "$/MWh", "storage_soc": "%"},
    )
    logger.log_reward_terms(
        epoch_id=1,
        episode_id=2,
        env_id="env_0",
        vpp_id="vpp_001",
        agent_id="vpp_001_dispatch",
        policy_id="dispatch_shared",
        date="2026-01-01",
        time_index=3,
        timestamp="2026-01-01T00:45:00",
        terms={"profit_reward": 2.5, "constraint_violation_penalty": -0.5, "total_reward": 2.0},
        units={"profit_reward": "scalar", "constraint_violation_penalty": "scalar", "total_reward": "scalar"},
    )
    logger.log_cost_terms(
        epoch_id=1,
        episode_id=2,
        env_id="env_0",
        vpp_id="vpp_001",
        date="2026-01-01",
        time_index=3,
        timestamp="2026-01-01T00:45:00",
        terms={"energy_purchase_cost": 1.2, "total_cost": 1.2},
        units={"energy_purchase_cost": "$", "total_cost": "$"},
    )
    logger.log_loss_terms(
        epoch_id=1,
        batch_id=4,
        gradient_step=5,
        vpp_id="vpp_001",
        agent_id="vpp_001_dispatch",
        policy_id="dispatch_shared",
        terms={"actor_loss": 0.12, "critic_loss": 0.34, "total_loss": 0.46},
        optimizer_name="adam",
        network_name="actor_critic",
    )
    logger.log_scalar("episode_return", 12.0, epoch_id=1, episode_id=2, global_env_step=96)
    logger.log_event("training_status", {"message": "epoch finished"}, epoch_id=1, episode_id=2)
    logger.close()

    run_dir = tmp_path / "run_test"
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "variable_dictionary.json").exists()
    assert (run_dir / "formulas.json").exists()
    assert list((run_dir / "tables" / "dataset_timeseries").glob("epoch_id=000001/vpp_id=vpp_001/*.parquet"))
    assert list((run_dir / "tables" / "reward_terms").glob("epoch_id=000001/vpp_id=vpp_001/*.parquet"))
    assert list((run_dir / "tables" / "cost_terms").glob("epoch_id=000001/vpp_id=vpp_001/*.parquet"))
    assert list((run_dir / "tables" / "loss_terms").glob("epoch_id=000001/vpp_id=vpp_001/*.parquet"))

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run_test"
    assert metadata["status"] == "finished"

    service = QueryService(tmp_path)
    dataset = service.query_metric_table(
        run_id="run_test",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        epoch_id=1,
        vpp_id="vpp_001",
    )
    assert dataset["summary"]["row_count"] == 1
    assert dataset["table_rows"][0]["metric_name"] == "electricity_price"
    assert dataset["table_rows"][0]["value"] == 88.5
    reward = service.query_metric_table(
        run_id="run_test",
        table="reward_terms",
        metrics=["total_reward"],
        epoch_id=1,
        vpp_id="vpp_001",
    )
    assert reward["summary"]["row_count"] == 1
    assert reward["table_rows"][0]["value"] == 2.0
    cost = service.query_metric_table(
        run_id="run_test",
        table="cost_terms",
        metrics=["total_cost"],
        epoch_id=1,
        vpp_id="vpp_001",
    )
    assert cost["summary"]["row_count"] == 1
    assert cost["table_rows"][0]["value"] == 1.2
    loss = service.query_metric_table(
        run_id="run_test",
        table="loss_terms",
        metrics=["total_loss"],
        epoch_id=1,
        vpp_id="vpp_001",
    )
    assert loss["summary"]["row_count"] == 1
    assert loss["table_rows"][0]["value"] == 0.46


def test_query_service_selectors_and_compare_use_logged_dimensions(tmp_path):
    logger = ExperimentLogger(run_id="run_compare", data_dir=tmp_path, config={}, async_writer=False)
    for vpp_id, value in [("vpp_001", 1.0), ("vpp_002", 3.0)]:
        logger.log_dataset(
            epoch_id=7,
            episode_id=0,
            env_id="env_0",
            vpp_id=vpp_id,
            date="2026-01-02",
            time_index=12,
            timestamp="2026-01-02T03:00:00",
            values={"net_load": value},
            units={"net_load": "MW"},
        )
    logger.close()

    service = QueryService(tmp_path)
    selectors = service.selectors("run_compare")
    assert selectors["epoch_ids"] == [7]
    assert selectors["dates"] == ["2026-01-02"]
    assert selectors["vpp_ids"] == ["vpp_001", "vpp_002"]
    assert selectors["time_indices"] == [12]

    comparison = service.compare(
        run_id="run_compare",
        table="dataset_timeseries",
        metric_names=["net_load"],
        group_by="vpp_id",
        fixed_epoch_id=7,
        fixed_date="2026-01-02",
        fixed_time_index=12,
    )
    rows_by_vpp = {row["vpp_id"]: row for row in comparison["table_rows"]}
    assert set(rows_by_vpp) == {"vpp_001", "vpp_002"}
    assert rows_by_vpp["vpp_001"]["group"] == "vpp_001"
    assert rows_by_vpp["vpp_002"]["group"] == "vpp_002"
    assert rows_by_vpp["vpp_001"]["metric_name"] == "net_load"
    assert rows_by_vpp["vpp_002"]["metric_name"] == "net_load"
    assert rows_by_vpp["vpp_001"]["value"] == 1.0
    assert rows_by_vpp["vpp_002"]["value"] == 3.0
    assert rows_by_vpp["vpp_001"]["unit"] == "MW"
    assert rows_by_vpp["vpp_002"]["unit"] == "MW"


def test_experiment_logger_appends_after_restart_without_overwriting_partitions(tmp_path):
    first = ExperimentLogger(run_id="run_restart", data_dir=tmp_path, config={}, async_writer=False)
    first.log_scalar("progress_rows", 1, epoch_id=0, vpp_id="aggregate", time_index=1)
    first.close()

    second = ExperimentLogger(run_id="run_restart", data_dir=tmp_path, config={}, async_writer=False)
    second.log_scalar("progress_rows", 2, epoch_id=0, vpp_id="aggregate", time_index=2)
    second.close()

    service = QueryService(tmp_path)
    scalars = service.query_metric_table(run_id="run_restart", table="scalar_metrics")
    values = [row["value"] for row in scalars["table_rows"] if row["metric_name"] == "progress_rows"]
    assert values == [1, 2]


def test_async_writer_keeps_partial_batches_in_memory_until_flush_or_close(tmp_path):
    logger = ExperimentLogger(run_id="run_async_batch", data_dir=tmp_path, config={}, async_writer=True, flush_rows=100)
    for step in range(4):
        logger.log_scalar("progress_rows", step, epoch_id=0, vpp_id="aggregate", time_index=step)

    time.sleep(0.35)

    table_dir = tmp_path / "run_async_batch" / "tables" / "scalar_metrics"
    assert not list(table_dir.rglob("*.parquet")) if table_dir.exists() else True

    logger.flush()

    files_after_flush = list(table_dir.rglob("*.parquet"))
    assert len(files_after_flush) == 1

    logger.close()


def test_compact_partition_reduces_part_files_without_losing_rows(tmp_path):
    logger = ExperimentLogger(run_id="run_compact", data_dir=tmp_path, config={}, async_writer=False)
    for step in range(5):
        logger.log_dataset(
            epoch_id=0,
            episode_id=1,
            env_id="env_0",
            vpp_id="vpp_001",
            date="2018-01-01",
            time_index=step,
            values={"electricity_price": 50.0 + step},
        )
    logger.close()
    partition_dir = tmp_path / "run_compact" / "tables" / "dataset_timeseries" / "epoch_id=000000" / "vpp_id=vpp_001"

    compact_file = compact_partition(partition_dir, min_part_files=2)

    assert compact_file is not None
    assert compact_file.name.startswith("compact-")
    assert list(partition_dir.glob("compact-*.parquet")) == [compact_file]
    assert not list(partition_dir.glob("part-*.parquet"))
    assert len(list((partition_dir / "_compacted_parts").glob("part-*.parquet"))) == 5

    dataset = QueryService(tmp_path).query_metric_table(
        run_id="run_compact",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_001",
    )
    assert dataset["summary"]["row_count"] == 5
    assert [row["time_index"] for row in dataset["table_rows"]] == [0, 1, 2, 3, 4]


def test_duckdb_store_reads_latest_compact_and_newer_part_files(tmp_path):
    logger = ExperimentLogger(run_id="run_compact_append", data_dir=tmp_path, config={}, async_writer=False)
    for step in range(4):
        logger.log_dataset(
            epoch_id=0,
            episode_id=1,
            env_id="env_0",
            vpp_id="vpp_001",
            date="2018-01-01",
            time_index=step,
            values={"electricity_price": 50.0 + step},
        )
    logger.close()
    partition_dir = tmp_path / "run_compact_append" / "tables" / "dataset_timeseries" / "epoch_id=000000" / "vpp_id=vpp_001"
    compact_file = compact_partition(partition_dir, min_part_files=2)
    assert compact_file is not None

    resumed = ExperimentLogger(run_id="run_compact_append", data_dir=tmp_path, config={}, async_writer=False)
    resumed.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_001",
        date="2018-01-01",
        time_index=4,
        values={"electricity_price": 54.0},
    )
    resumed.close()

    service = QueryService(tmp_path)
    table_files = service.duckdb_store.table_files(
        "run_compact_append",
        "dataset_timeseries",
        epoch_id=0,
        vpp_id="vpp_001",
    )
    assert len(table_files) == 2
    assert any("compact-" in path for path in table_files)
    assert any("part-" in path for path in table_files)

    dataset = service.query_metric_table(
        run_id="run_compact_append",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_001",
    )
    assert dataset["summary"]["row_count"] == 5
    assert [row["time_index"] for row in dataset["table_rows"]] == [0, 1, 2, 3, 4]


def test_compact_partition_can_keep_part_files_for_live_writers(tmp_path):
    logger = ExperimentLogger(run_id="run_live_compact", data_dir=tmp_path, config={}, async_writer=False)
    for step in range(4):
        logger.log_dataset(
            epoch_id=0,
            episode_id=1,
            env_id="env_0",
            vpp_id="vpp_001",
            date="2018-01-01",
            time_index=step,
            values={"electricity_price": 50.0 + step},
        )
    logger.close()
    partition_dir = tmp_path / "run_live_compact" / "tables" / "dataset_timeseries" / "epoch_id=000000" / "vpp_id=vpp_001"

    compact_file = compact_partition(partition_dir, min_part_files=2, archive_inputs=False)

    assert compact_file is not None
    assert len(list(partition_dir.glob("part-*.parquet"))) == 4
    table_files = QueryService(tmp_path).duckdb_store.table_files(
        "run_live_compact",
        "dataset_timeseries",
        epoch_id=0,
        vpp_id="vpp_001",
    )
    assert table_files == [str(compact_file)]


def test_demo_variable_dictionary_covers_generated_reward_cost_and_loss_terms():
    names = {item["name"] for item in default_variable_dictionary()}

    assert {
        "profit_reward",
        "grid_balance_reward",
        "storage_degradation_penalty",
        "constraint_violation_penalty",
        "total_reward",
        "energy_purchase_cost",
        "storage_degradation_cost",
        "constraint_violation_cost",
        "total_cost",
        "actor_loss",
        "critic_loss",
        "entropy_loss",
        "total_loss",
    }.issubset(names)
