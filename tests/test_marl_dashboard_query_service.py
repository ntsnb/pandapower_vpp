from __future__ import annotations

from marl_dashboard.backend.storage.metadata_store import MetadataStore
from marl_dashboard.backend.storage.query_service import QueryService
from marl_dashboard.backend.storage.variable_enrichment import variable_defaults
from marl_dashboard.logging import ExperimentLogger


def test_query_service_deduplicates_replayed_metric_rows_by_semantic_key(tmp_path):
    logger = ExperimentLogger(run_id="run_replayed_loss", data_dir=tmp_path, config={}, async_writer=False)
    for _ in range(2):
        logger.log_loss_terms(
            epoch_id=0,
            episode_id=1,
            env_id="paper_long",
            vpp_id="aggregate",
            agent_id="aggregate",
            policy_id="happo",
            gradient_step=1,
            global_env_step=672,
            terms={"critic_loss": 0.25},
            units={"critic_loss": "scalar"},
        )
    logger.close()

    service = QueryService(tmp_path)
    losses = service.query_metric_table(run_id="run_replayed_loss", table="loss_terms")

    assert losses["summary"]["row_count"] == 1
    assert len(losses["table_rows"]) == 1
    assert len(losses["chart_series"][0]["points"]) == 1


def test_query_service_normalizes_float_like_worker_env_ids_before_dataset_deduplication(tmp_path):
    logger = ExperimentLogger(run_id="run_worker_env_ids", data_dir=tmp_path, config={}, async_writer=False)
    common_context = {
        "epoch_id": 0,
        "episode_id": 1,
        "date": "2018-01-01",
        "time_index": 24,
        "global_env_step": 24,
        "vpp_id": "vpp_a",
        "agent_id": "vpp_a_dispatch",
        "policy_id": "happo",
    }
    logger.log_dataset(
        **common_context,
        env_id="worker_0.0",
        values={"electricity_price": 50.0},
        units={"electricity_price": "currency/MWh"},
    )
    logger.log_dataset(
        **common_context,
        env_id="worker_0",
        values={"electricity_price": 50.0},
        units={"electricity_price": "currency/MWh"},
    )
    logger.close()

    dataset = QueryService(tmp_path).query_metric_table(
        run_id="run_worker_env_ids",
        table="dataset_timeseries",
        vpp_id="vpp_a",
    )

    assert dataset["summary"]["row_count"] == 1
    assert dataset["table_rows"][0]["env_id"] == "worker_0"
    assert len(dataset["chart_series"][0]["points"]) == 1


def test_query_service_collapses_identical_dataset_points_from_parallel_workers(tmp_path):
    logger = ExperimentLogger(run_id="run_parallel_profile_points", data_dir=tmp_path, config={}, async_writer=False)
    common_context = {
        "epoch_id": 0,
        "episode_id": 6,
        "date": "2018-01-01",
        "time_index": 48,
        "global_env_step": 48,
        "vpp_id": "vpp_commercial_multi",
        "agent_id": "vpp_commercial_multi_dispatch",
        "policy_id": "happo",
    }
    for worker_id in range(7):
        logger.log_dataset(
            **common_context,
            env_id=f"worker_{worker_id}",
            values={"electricity_price": 58.76934371785372},
            units={"electricity_price": "currency/MWh"},
        )
    logger.close()

    dataset = QueryService(tmp_path).query_metric_table(
        run_id="run_parallel_profile_points",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_commercial_multi",
        episode_id=6,
        date="2018-01-01",
    )

    assert dataset["summary"]["row_count"] == 1
    assert len(dataset["table_rows"]) == 1
    assert len(dataset["chart_series"][0]["points"]) == 1


def test_query_service_canonicalizes_happo_policy_aliases_in_selectors_and_series(tmp_path):
    logger = ExperimentLogger(run_id="run_happo_aliases", data_dir=tmp_path, config={}, async_writer=False)
    common_context = {
        "epoch_id": 0,
        "episode_id": 1,
        "date": "2018-01-01",
        "time_index": 0,
        "global_env_step": 0,
        "env_id": "worker_0",
        "vpp_id": "vpp_commercial_multi",
        "agent_id": "vpp_commercial_multi_dispatch",
    }
    logger.log_dataset(
        **common_context,
        policy_id="happo",
        values={"electricity_price": 52.0},
        units={"electricity_price": "currency/MWh"},
    )
    logger.log_dataset(
        **common_context,
        policy_id="happo_sequential_ctde",
        values={"electricity_price": 52.0},
        units={"electricity_price": "currency/MWh"},
    )
    logger.close()

    service = QueryService(tmp_path)
    selectors = service.selectors("run_happo_aliases")
    dataset = service.query_metric_table(
        run_id="run_happo_aliases",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_commercial_multi",
        date="2018-01-01",
    )

    assert selectors["policy_ids"] == ["happo_sequential_ctde"]
    assert dataset["summary"]["row_count"] == 1
    assert len(dataset["chart_series"]) == 1
    assert dataset["chart_series"][0]["policy_id"] == "happo_sequential_ctde"
    assert dataset["chart_series"][0]["name"] == "electricity_price / vpp_commercial_multi / happo_sequential_ctde"
    assert dataset["table_rows"][0]["policy_id"] == "happo_sequential_ctde"


def test_query_service_expands_canonical_happo_policy_filter_to_raw_alias(tmp_path):
    logger = ExperimentLogger(run_id="run_raw_happo_only", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="worker_0",
        vpp_id="vpp_commercial_multi",
        agent_id="vpp_commercial_multi_dispatch",
        policy_id="happo",
        date="2018-01-01",
        time_index=0,
        global_env_step=0,
        values={"electricity_price": 52.0},
        units={"electricity_price": "currency/MWh"},
    )
    logger.close()

    dataset = QueryService(tmp_path).query_metric_table(
        run_id="run_raw_happo_only",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_commercial_multi",
        date="2018-01-01",
        policy_id="happo_sequential_ctde",
    )

    assert dataset["summary"]["row_count"] == 1
    assert dataset["table_rows"][0]["policy_id"] == "happo_sequential_ctde"
    assert dataset["chart_series"][0]["policy_id"] == "happo_sequential_ctde"


def test_selectors_exclude_aggregate_run_status_labels_from_policy_ids(tmp_path):
    logger = ExperimentLogger(run_id="run_status_policy_labels", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="paper_long",
        vpp_id="aggregate",
        agent_id="aggregate",
        policy_id="paper_long",
        time_index=0,
        values={"progress_rows": 10.0, "episode_rows": 1.0},
    )
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="worker_0",
        vpp_id="vpp_commercial_multi",
        agent_id="vpp_commercial_multi_dispatch",
        policy_id="happo",
        date="2018-01-01",
        time_index=0,
        values={"electricity_price": 52.0},
        units={"electricity_price": "currency/MWh"},
    )
    logger.close()

    selectors = QueryService(tmp_path).selectors("run_status_policy_labels")

    assert selectors["policy_ids"] == ["happo_sequential_ctde"]


def test_query_service_reuses_metric_query_cache_until_table_signature_changes(tmp_path, monkeypatch):
    logger = ExperimentLogger(run_id="run_cache", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        time_index=0,
        values={"electricity_price": 50.0},
        units={"electricity_price": "currency/MWh"},
    )
    logger.close()

    service = QueryService(tmp_path)
    filtered_count_calls = 0
    original_filtered_count = service.duckdb_store.filtered_count

    def counted_filtered_count(*args, **kwargs):
        nonlocal filtered_count_calls
        filtered_count_calls += 1
        return original_filtered_count(*args, **kwargs)

    monkeypatch.setattr(service.duckdb_store, "filtered_count", counted_filtered_count)

    first = service.query_metric_table(
        run_id="run_cache",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_a",
    )
    second = service.query_metric_table(
        run_id="run_cache",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_a",
    )

    assert filtered_count_calls == 1
    assert second["summary"] == first["summary"]

    logger = ExperimentLogger(run_id="run_cache", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        time_index=1,
        values={"electricity_price": 55.0},
        units={"electricity_price": "currency/MWh"},
    )
    logger.close()

    updated = service.query_metric_table(
        run_id="run_cache",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_a",
    )

    assert filtered_count_calls == 2
    assert updated["summary"]["row_count"] == 2


def test_query_service_refuses_large_unfiltered_dataset_queries_before_duckdb_scan(tmp_path, monkeypatch):
    service = QueryService(tmp_path)

    monkeypatch.setattr(service.duckdb_store, "table_signature", lambda run_id, table: (2000, 123, 456))

    def fail_if_duckdb_scan_is_started(*args, **kwargs):
        raise AssertionError("large unfiltered dataset queries must not start a DuckDB scan")

    monkeypatch.setattr(service.duckdb_store, "filtered_count", fail_if_duckdb_scan_is_started)

    result = service.query_metric_table(run_id="large_live_run", table="dataset_timeseries")

    assert result["summary"]["row_count"] == 0
    assert result["summary"]["requires_filter"] is True
    assert result["summary"]["reason"] == "large_unfiltered_dataset_query"


def test_selectors_do_not_scan_scalar_metrics_when_core_tables_provide_dimensions(tmp_path, monkeypatch):
    logger = ExperimentLogger(run_id="run_fast_selectors", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        date="2018-01-01",
        time_index=0,
        values={"electricity_price": 50.0},
    )
    logger.log_scalar(
        "training_wall_time_seconds",
        1.0,
        epoch_id=0,
        episode_id=1,
        vpp_id="aggregate",
        policy_id="happo",
    )
    logger.close()

    service = QueryService(tmp_path)
    original_table_signature = service.duckdb_store.table_signature
    original_distinct_values_for_columns = service.duckdb_store.distinct_values_for_columns

    def fail_on_scalar_signature(run_id, table):
        if table == "scalar_metrics":
            raise AssertionError("selectors should not stat scalar_metrics when core metric tables have dimensions")
        return original_table_signature(run_id, table)

    def fail_on_scalar_distinct(run_id, table, columns):
        if table == "scalar_metrics":
            raise AssertionError("selectors should not scan scalar_metrics when core metric tables have dimensions")
        return original_distinct_values_for_columns(run_id, table, columns)

    monkeypatch.setattr(service.duckdb_store, "table_signature", fail_on_scalar_signature)
    monkeypatch.setattr(service.duckdb_store, "distinct_values_for_columns", fail_on_scalar_distinct)

    selectors = service.selectors("run_fast_selectors")

    assert selectors["dates"] == ["2018-01-01"]
    assert selectors["vpp_ids"] == ["vpp_a"]
    assert selectors["episode_ids"] == [1]


def test_query_service_sorts_sampled_chart_points_by_display_time_axis(tmp_path):
    logger = ExperimentLogger(run_id="run_sampled_axis", data_dir=tmp_path, config={}, async_writer=False)
    for episode_id in (1, 2):
        for time_index in range(4):
            logger.log_dataset(
                epoch_id=0,
                episode_id=episode_id,
                env_id="env_0",
                vpp_id="vpp_a",
                date="2018-01-02",
                time_index=time_index,
                timestamp=f"2018-01-02T0{time_index}:00:00Z",
                values={"electricity_price": 50.0 + episode_id + time_index},
                units={"electricity_price": "currency/MWh"},
            )
    logger.close()

    result = QueryService(tmp_path).query_metric_table(
        run_id="run_sampled_axis",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_a",
        date="2018-01-02",
        max_points=3,
    )

    time_indices = [point["time_index"] for point in result["chart_series"][0]["points"]]
    assert time_indices == sorted(time_indices)


def test_query_service_samples_multi_metric_charts_without_dropping_selected_series(tmp_path):
    logger = ExperimentLogger(run_id="run_multi_metric_sampling", data_dir=tmp_path, config={}, async_writer=False)
    for time_index in range(5):
        logger.log_reward_terms(
            epoch_id=0,
            episode_id=1,
            env_id="env_0",
            vpp_id="vpp_a",
            date="2018-01-01",
            time_index=time_index,
            terms={
                "availability_payment": float(time_index),
                "dispatch_reward_env": float(time_index + 10),
                "evcs_user_revenue_total": float(time_index + 20),
            },
        )
    logger.close()

    result = QueryService(tmp_path).query_metric_table(
        run_id="run_multi_metric_sampling",
        table="reward_terms",
        metrics=["availability_payment", "dispatch_reward_env", "evcs_user_revenue_total"],
        vpp_id="vpp_a",
        date="2018-01-01",
        max_points=6,
    )

    assert {series["metric_name"] for series in result["chart_series"]} == {
        "availability_payment",
        "dispatch_reward_env",
        "evcs_user_revenue_total",
    }
    assert sum(len(series["points"]) for series in result["chart_series"]) <= 6


def test_query_service_replaces_english_code_formula_metadata_with_chinese_default(tmp_path):
    logger = ExperimentLogger(run_id="run_reward_formula", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_reward_terms(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        date="2018-01-01",
        time_index=0,
        terms={"energy_market_revenue": 12.5},
        formulas={"energy_market_revenue": "market_price * delivered_p_mw * dt_hours"},
    )
    logger.close()

    result = QueryService(tmp_path).query_metric_table(
        run_id="run_reward_formula",
        table="reward_terms",
        metrics=["energy_market_revenue"],
        vpp_id="vpp_a",
    )

    expected_formula = variable_defaults("energy_market_revenue")["formula_latex"]
    assert result["formulas"]["energy_market_revenue"] == expected_formula
    assert result["table_rows"][0]["formula_latex"] == expected_formula
    assert "market_price" not in result["formulas"]["energy_market_revenue"]


def test_query_service_replaces_legacy_symbol_formula_metadata_with_canonical_default(tmp_path):
    logger = ExperimentLogger(run_id="run_legacy_formula_symbols", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_cost_terms(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        date="2018-01-01",
        time_index=0,
        terms={"energy_purchase_cost": 12.5},
        formulas={"energy_purchase_cost": "C^{energy}_{i,t}=\\max(P^{net}_{i,t},0)c_t"},
    )
    logger.close()

    result = QueryService(tmp_path).query_metric_table(
        run_id="run_legacy_formula_symbols",
        table="cost_terms",
        metrics=["energy_purchase_cost"],
        vpp_id="vpp_a",
    )

    expected_formula = variable_defaults("energy_purchase_cost")["formula_latex"]
    assert result["formulas"]["energy_purchase_cost"] == expected_formula
    assert result["table_rows"][0]["formula_latex"] == expected_formula
    assert "c_t" not in result["formulas"]["energy_purchase_cost"]
    assert "C^{energy}" not in result["formulas"]["energy_purchase_cost"]


def test_query_service_formulas_include_canonical_defaults_when_run_metadata_is_empty(tmp_path):
    MetadataStore(tmp_path).initialize_run(
        run_id="run_empty_formulas",
        config={"algorithm": "happo", "environment": "paper_training"},
    )

    formulas = QueryService(tmp_path).formulas("run_empty_formulas")

    assert formulas["total_reward"] == variable_defaults("total_reward")["formula_latex"]
    assert formulas["energy_purchase_cost"] == variable_defaults("energy_purchase_cost")["formula_latex"]
    assert "c_t" not in formulas["energy_purchase_cost"]
    assert "C^{energy}" not in formulas["energy_purchase_cost"]


def test_selectors_use_day_local_time_indices_when_real_dates_exist(tmp_path):
    logger = ExperimentLogger(run_id="run_mixed_dates", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="old_adapter",
        vpp_id="vpp_a",
        time_index=671,
        values={"electricity_price": 90.0},
    )
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="calendar_adapter",
        vpp_id="vpp_a",
        date="2018-01-07",
        time_index=95,
        timestamp="2018-01-07T23:45:00Z",
        values={"electricity_price": 91.0},
    )
    logger.close()

    selectors = QueryService(tmp_path).selectors("run_mixed_dates")

    assert selectors["dates"] == ["2018-01-07"]
    assert selectors["time_indices"] == [95]


def test_selectors_report_partial_date_time_slot_status(tmp_path):
    logger = ExperimentLogger(run_id="run_partial_dates", data_dir=tmp_path, config={}, async_writer=False)
    for time_index in (0, 1):
        logger.log_dataset(
            epoch_id=0,
            episode_id=1,
            env_id="env_0",
            vpp_id="vpp_a",
            date="2018-01-01",
            time_index=time_index,
            values={"electricity_price": 50.0 + time_index},
        )
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        date="2018-01-02",
        time_index=0,
        values={"electricity_price": 55.0},
    )
    logger.close()

    selectors = QueryService(tmp_path).selectors("run_partial_dates")

    assert selectors["time_indices"] == [0, 1]
    assert selectors["date_statuses"] == [
        {
            "date": "2018-01-01",
            "observed_time_slots": 2,
            "expected_time_slots": 2,
            "complete": True,
            "status": "complete",
        },
        {
            "date": "2018-01-02",
            "observed_time_slots": 1,
            "expected_time_slots": 2,
            "complete": False,
            "status": "partial",
        },
    ]


def test_selectors_cast_float_like_episode_ids_to_ints(tmp_path):
    logger = ExperimentLogger(run_id="run_float_episode", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_dataset(
        epoch_id=0,
        episode_id=1.0,
        env_id="env_0",
        vpp_id="vpp_a",
        time_index=0,
        values={"electricity_price": 50.0},
    )
    logger.close()

    selectors = QueryService(tmp_path).selectors("run_float_episode")

    assert selectors["episode_ids"] == [1]
    assert isinstance(selectors["episode_ids"][0], int)


def test_query_service_normalizes_partitioned_integer_context_ids(tmp_path):
    logger = ExperimentLogger(run_id="run_partition_ids", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_event("training_status", {"message": "epoch finished"}, epoch_id=1, episode_id=2, time_index=3)
    logger.close()

    result = QueryService(tmp_path).query_metric_table(run_id="run_partition_ids", table="events")

    row = result["table_rows"][0]
    assert row["epoch_id"] == 1
    assert row["episode_id"] == 2
    assert row["time_index"] == 3
    assert isinstance(row["epoch_id"], int)


def test_loss_query_for_vpp_falls_back_to_aggregate_shared_losses(tmp_path):
    logger = ExperimentLogger(run_id="run_shared_loss", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_loss_terms(
        epoch_id=0,
        episode_id=1,
        gradient_step=2,
        vpp_id="aggregate",
        agent_id="aggregate",
        policy_id="happo",
        terms={"dispatch_policy_loss": -0.2, "critic_loss": 0.4},
    )
    logger.close()

    losses = QueryService(tmp_path).query_metric_table(
        run_id="run_shared_loss",
        table="loss_terms",
        vpp_id="vpp_a",
    )

    assert losses["summary"]["row_count"] == 2
    assert losses["summary"]["vpp_filter_fallback"] == "aggregate_shared_loss"
    assert losses["summary"]["requested_vpp_id"] == "vpp_a"
    assert losses["summary"]["effective_vpp_id"] == "aggregate"
    assert {row["metric_name"] for row in losses["table_rows"]} == {"critic_loss", "dispatch_policy_loss"}


def test_query_service_maps_real_dataset_dates_to_synthetic_reward_days(tmp_path):
    logger = ExperimentLogger(run_id="run_reward_calendar", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        date="2018-01-01",
        time_index=0,
        timestamp="2018-01-01T00:00:00Z",
        values={"electricity_price": 50.0},
    )
    logger.log_dataset(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        date="2018-01-02",
        time_index=0,
        timestamp="2018-01-02T00:00:00Z",
        values={"electricity_price": 55.0},
    )
    logger.log_reward_terms(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        time_index=0,
        terms={"dispatch_reward_train": 1.0},
    )
    logger.log_reward_terms(
        epoch_id=0,
        episode_id=1,
        env_id="env_0",
        vpp_id="vpp_a",
        time_index=96,
        terms={"dispatch_reward_train": 2.0},
    )
    logger.close()

    rewards = QueryService(tmp_path).query_metric_table(
        run_id="run_reward_calendar",
        table="reward_terms",
        vpp_id="vpp_a",
        date="2018-01-02",
    )

    assert rewards["summary"]["row_count"] == 1
    assert rewards["table_rows"][0]["value"] == 2.0
    assert rewards["table_rows"][0]["date"] == "2018-01-02"
    assert rewards["table_rows"][0]["time_index"] == 0


def test_compare_returns_one_latest_row_per_group_and_metric(tmp_path):
    logger = ExperimentLogger(run_id="run_compare_dedupe", data_dir=tmp_path, config={}, async_writer=False)
    for episode_id, value in [(1, 10.0), (2, 12.0)]:
        logger.log_dataset(
            epoch_id=0,
            episode_id=episode_id,
            env_id="env_0",
            vpp_id="vpp_a",
            date="2018-01-02",
            time_index=1,
            timestamp="2018-01-02T00:15:00Z",
            values={"net_load": value},
            units={"net_load": "MW"},
        )
    logger.close()

    comparison = QueryService(tmp_path).compare(
        run_id="run_compare_dedupe",
        table="dataset_timeseries",
        metric_names=["net_load"],
        group_by="vpp_id",
        fixed_date="2018-01-02",
        fixed_time_index=1,
    )

    assert comparison["summary"]["raw_row_count"] == 2
    assert comparison["summary"]["row_count"] == 2
    assert comparison["summary"]["comparison_row_count"] == 1
    assert comparison["summary"]["deduplicated_rows"] == 1
    assert comparison["summary"]["table_rows_limited"] is False
    assert len(comparison["table_rows"]) == 1
    assert comparison["table_rows"][0]["episode_id"] == 2
    assert comparison["table_rows"][0]["value"] == 12.0


def test_compare_filters_by_fixed_episode_id_before_deduplicating(tmp_path):
    logger = ExperimentLogger(run_id="run_compare_episode_filter", data_dir=tmp_path, config={}, async_writer=False)
    for episode_id, value in [(1, 10.0), (2, 12.0)]:
        logger.log_dataset(
            epoch_id=0,
            episode_id=episode_id,
            env_id="env_0",
            vpp_id="vpp_a",
            date="2018-01-02",
            time_index=1,
            timestamp="2018-01-02T00:15:00Z",
            values={"net_load": value},
            units={"net_load": "MW"},
        )
    logger.close()

    comparison = QueryService(tmp_path).compare(
        run_id="run_compare_episode_filter",
        table="dataset_timeseries",
        metric_names=["net_load"],
        group_by="vpp_id",
        fixed_episode_id=1,
        fixed_date="2018-01-02",
        fixed_time_index=1,
    )

    assert comparison["summary"]["row_count"] == 1
    assert comparison["summary"]["fixed_episode_id"] == 1
    assert len(comparison["table_rows"]) == 1
    assert comparison["table_rows"][0]["episode_id"] == 1
    assert comparison["table_rows"][0]["value"] == 10.0


def test_compare_returns_chart_series_and_formula_metadata(tmp_path):
    logger = ExperimentLogger(run_id="run_compare_chart", data_dir=tmp_path, config={}, async_writer=False)
    for vpp_id, value in [("vpp_a", 10.0), ("vpp_b", 12.0)]:
        logger.log_dataset(
            epoch_id=0,
            episode_id=1,
            env_id="env_0",
            vpp_id=vpp_id,
            date="2018-01-02",
            time_index=1,
            timestamp="2018-01-02T00:15:00Z",
            values={"net_load": value},
            units={"net_load": "MW"},
        )
    logger.close()

    comparison = QueryService(tmp_path).compare(
        run_id="run_compare_chart",
        table="dataset_timeseries",
        metric_names=["net_load"],
        group_by="vpp_id",
        fixed_date="2018-01-02",
        fixed_time_index=1,
    )

    assert comparison["formulas"]["net_load"]
    assert len(comparison["chart_series"]) == 2
    assert {series["name"] for series in comparison["chart_series"]} == {"net_load / vpp_a", "net_load / vpp_b"}
    assert {series["points"][0]["vpp_id"] for series in comparison["chart_series"]} == {"vpp_a", "vpp_b"}
