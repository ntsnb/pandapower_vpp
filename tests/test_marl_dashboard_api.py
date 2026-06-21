from __future__ import annotations

from fastapi.testclient import TestClient

from marl_dashboard.backend.app import create_app
from marl_dashboard.backend.api.websocket import coalesce_live_events, persistent_live_update_events, table_signature_snapshot
from marl_dashboard.backend.storage.duckdb_store import DuckDBStore
from marl_dashboard.backend.storage.metadata_store import MetadataStore
from marl_dashboard.backend.storage.query_service import QueryService
from marl_dashboard.demo.generate_demo_run import generate_demo_run
from marl_dashboard.logging import ExperimentLogger
from marl_dashboard.logging.event_bus import EventBus, default_event_bus


def test_runs_endpoint_does_not_scan_metric_tables(tmp_path, monkeypatch):
    MetadataStore(tmp_path).initialize_run(
        run_id="live_run",
        config={"algorithm": "happo", "environment": "paper_training"},
    )

    def fail_if_metric_table_is_read(*args, **kwargs):
        raise AssertionError("/api/runs must not scan metric parquet tables")

    monkeypatch.setattr(DuckDBStore, "read_table", fail_if_metric_table_is_read)
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == [
        {
            "run_id": "live_run",
            "status": "running",
            "started_at": response.json()[0]["started_at"],
            "ended_at": None,
            "algorithm": "happo",
            "environment": "paper_training",
            "vpp_count": None,
            "epoch_count": None,
            "metadata": {},
        }
    ]


def test_live_websocket_sends_initial_run_status_snapshot(tmp_path):
    MetadataStore(tmp_path).initialize_run(
        run_id="live_run",
        config={"algorithm": "happo", "environment": "paper_training"},
    )
    client = TestClient(create_app(data_dir=tmp_path))

    with client.websocket_connect("/ws/runs/live_run/live") as websocket:
        payload = websocket.receive_json()

    assert payload["run_id"] == "live_run"
    assert payload["event_type"] == "run_status"
    assert payload["status"] == "running"
    assert payload["algorithm"] == "happo"


def test_live_websocket_streams_new_event_bus_events_after_connection(tmp_path):
    MetadataStore(tmp_path).initialize_run(
        run_id="connected_live_run",
        config={"algorithm": "happo", "environment": "paper_training"},
    )
    client = TestClient(create_app(data_dir=tmp_path))

    with client.websocket_connect("/ws/runs/connected_live_run/live") as websocket:
        initial = websocket.receive_json()
        assert initial["event_type"] == "run_status"
        default_event_bus.publish(
            {
                "run_id": "connected_live_run",
                "event_type": "latest_scalar_metrics",
                "table": "scalar_metrics",
                "metric_name": "episode_return",
                "value": 123.0,
            }
        )
        payload = websocket.receive_json()

    assert payload["run_id"] == "connected_live_run"
    assert payload["event_type"] == "latest_scalar_metrics"
    assert payload["metric_name"] == "episode_return"
    assert payload["value"] == 123.0


def test_live_websocket_replays_logger_rows_as_semantic_latest_events(tmp_path):
    logger = ExperimentLogger(run_id="semantic_live_run", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_reward_terms(
        epoch_id=0,
        episode_id=1,
        vpp_id="vpp_a",
        agent_id="vpp_a_dispatch",
        policy_id="happo",
        date="2018-01-01",
        time_index=3,
        terms={"profit_reward": 1.5, "total_reward": 2.0},
    )
    logger.close()
    client = TestClient(create_app(data_dir=tmp_path))

    with client.websocket_connect("/ws/runs/semantic_live_run/live") as websocket:
        initial = websocket.receive_json()
        payload = websocket.receive_json()

    assert initial["event_type"] == "run_status"
    assert payload["run_id"] == "semantic_live_run"
    assert payload["event_type"] == "latest_reward_terms"
    assert payload["table"] == "reward_terms"
    assert payload["metric_group"] == "reward"
    assert payload["latest_context"]["vpp_id"] == "vpp_a"
    assert payload["latest_context"]["episode_id"] == 1
    assert payload["latest_context"]["time_index"] == 3
    assert {row["metric_name"] for row in payload["rows"]} == {"profit_reward", "total_reward"}


def test_live_websocket_coalesces_high_frequency_metric_events():
    events = [
        {"run_id": "run_a", "table": "loss_terms", "metric_name": "critic_loss", "vpp_id": "aggregate", "value": 0.1},
        {"run_id": "run_a", "table": "loss_terms", "metric_name": "critic_loss", "vpp_id": "aggregate", "value": 0.2},
        {"run_id": "run_a", "table": "loss_terms", "metric_name": "actor_loss", "vpp_id": "aggregate", "value": 0.3},
    ]

    coalesced = coalesce_live_events(events)

    assert len(coalesced) == 2
    assert {event["metric_name"] for event in coalesced} == {"critic_loss", "actor_loss"}
    assert next(event for event in coalesced if event["metric_name"] == "critic_loss")["value"] == 0.2


def test_persistent_live_update_events_detect_cross_process_parquet_changes(tmp_path):
    run_id = "cross_process_live"
    isolated_bus = EventBus()
    logger = ExperimentLogger(run_id=run_id, data_dir=tmp_path, config={}, async_writer=False, event_bus=isolated_bus)
    logger.log_event("training_status", {"message": "started"})
    logger.close()
    service = QueryService(tmp_path)
    previous = table_signature_snapshot(service, run_id)

    logger = ExperimentLogger(run_id=run_id, data_dir=tmp_path, config={}, async_writer=False, event_bus=isolated_bus)
    logger.log_reward_terms(
        epoch_id=0,
        episode_id=1,
        vpp_id="vpp_a",
        agent_id="vpp_a_dispatch",
        policy_id="happo",
        terms={"dispatch_reward_train": 1.0},
    )
    logger.close()

    events, current = persistent_live_update_events(service, run_id, previous)

    assert current != previous
    assert any(event["event_type"] == "persistent_table_update" and event["table"] == "reward_terms" for event in events)
    assert all(event["run_id"] == run_id for event in events)


def test_persistent_live_update_events_include_latest_semantic_rows(tmp_path):
    run_id = "cross_process_semantic_live"
    isolated_bus = EventBus()
    logger = ExperimentLogger(run_id=run_id, data_dir=tmp_path, config={}, async_writer=False, event_bus=isolated_bus)
    logger.log_event("training_status", {"message": "started"})
    logger.close()
    service = QueryService(tmp_path)
    previous = table_signature_snapshot(service, run_id)

    logger = ExperimentLogger(run_id=run_id, data_dir=tmp_path, config={}, async_writer=False, event_bus=isolated_bus)
    logger.log_reward_terms(
        epoch_id=0,
        episode_id=1,
        vpp_id="vpp_a",
        agent_id="vpp_a_dispatch",
        policy_id="happo",
        date="2018-01-01",
        time_index=4,
        terms={"dispatch_reward_train": 1.0, "total_reward": 2.0},
    )
    logger.close()

    events, _ = persistent_live_update_events(service, run_id, previous)

    semantic_events = [event for event in events if event["event_type"] == "latest_reward_terms"]
    assert semantic_events
    latest = semantic_events[0]
    assert latest["table"] == "reward_terms"
    assert latest["metric_group"] == "reward"
    assert latest["latest_context"]["vpp_id"] == "vpp_a"
    assert latest["latest_context"]["time_index"] == 4
    assert {row["metric_name"] for row in latest["rows"]} == {"dispatch_reward_train", "total_reward"}


def test_variables_endpoint_enriches_common_variables_with_bilingual_descriptions(tmp_path):
    MetadataStore(tmp_path).initialize_run(
        run_id="variable_run",
        config={"algorithm": "happo", "environment": "paper_training"},
        variable_dictionary=[
            {"name": "reward_so_far", "display_name": "Episode reward so far", "group": "reward"},
            {"name": "electricity_price", "display_name": "Electricity price", "group": "dataset"},
        ],
    )
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get("/api/runs/variable_run/variables")

    assert response.status_code == 200
    variables = {item["name"]: item for item in response.json()}
    assert "累计奖励 / Episode reward so far" == variables["reward_so_far"]["display_name"]
    assert "当前 episode" in variables["reward_so_far"]["physical_meaning"]
    assert "Cumulative reward" in variables["reward_so_far"]["physical_meaning"]
    assert "电价 / Electricity price" == variables["electricity_price"]["display_name"]
    assert "单位" in variables["electricity_price"]["notes"]


def test_selectors_endpoint_uses_lightweight_distinct_queries(tmp_path, monkeypatch):
    run_id = generate_demo_run(
        data_dir=tmp_path,
        run_id="selector_demo",
        vpp_count=2,
        epochs=2,
        days=1,
        steps_per_day=3,
        async_writer=False,
    )

    def fail_if_full_table_is_read(*args, **kwargs):
        raise AssertionError("/selectors must not fetch full metric tables")

    monkeypatch.setattr(DuckDBStore, "read_table", fail_if_full_table_is_read)
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(f"/api/runs/{run_id}/selectors")

    assert response.status_code == 200
    payload = response.json()
    assert payload["vpp_ids"] == ["vpp_001", "vpp_002"]
    assert payload["epoch_ids"] == [0, 1]
    assert payload["time_indices"] == [0, 1, 2]


def test_selectors_endpoint_batches_distinct_queries(tmp_path, monkeypatch):
    run_id = generate_demo_run(
        data_dir=tmp_path,
        run_id="batched_selector_demo",
        vpp_count=2,
        epochs=2,
        days=1,
        steps_per_day=3,
        async_writer=False,
    )

    def fail_if_per_column_distinct_is_used(*args, **kwargs):
        raise AssertionError("/selectors must batch distinct column scans per table")

    monkeypatch.setattr(DuckDBStore, "distinct_values", fail_if_per_column_distinct_is_used)
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(f"/api/runs/{run_id}/selectors")

    assert response.status_code == 200
    payload = response.json()
    assert payload["vpp_ids"] == ["vpp_001", "vpp_002"]
    assert payload["epoch_ids"] == [0, 1]
    assert payload["time_indices"] == [0, 1, 2]


def test_selectors_endpoint_reuses_cache_until_tables_change(tmp_path, monkeypatch):
    run_id = generate_demo_run(
        data_dir=tmp_path,
        run_id="cached_selector_demo",
        vpp_count=2,
        epochs=2,
        days=1,
        steps_per_day=3,
        async_writer=False,
    )
    original = DuckDBStore.distinct_values_for_columns
    calls = 0

    def count_distinct_queries(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBStore, "distinct_values_for_columns", count_distinct_queries)
    client = TestClient(create_app(data_dir=tmp_path))

    first = client.get(f"/api/runs/{run_id}/selectors")
    second = client.get(f"/api/runs/{run_id}/selectors")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert calls == 4


def test_metric_queries_limit_returned_table_rows(tmp_path):
    run_id = generate_demo_run(
        data_dir=tmp_path,
        run_id="limited_rows_demo",
        vpp_count=2,
        epochs=2,
        days=2,
        steps_per_day=8,
        async_writer=False,
    )
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        f"/api/runs/{run_id}/dataset",
        params={"vpp_id": "vpp_001", "metrics": "electricity_price,net_load", "max_points": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] > 5
    assert payload["summary"]["returned_table_rows"] == 5
    assert len(payload["table_rows"]) == 5
    assert payload["summary"]["returned_points"] <= 5


def test_metric_queries_enrich_reward_rows_with_bilingual_descriptions(tmp_path):
    run_id = generate_demo_run(
        data_dir=tmp_path,
        run_id="bilingual_metric_demo",
        vpp_count=1,
        epochs=1,
        days=1,
        steps_per_day=3,
        async_writer=False,
    )
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        f"/api/runs/{run_id}/rewards",
        params={"metrics": "profit_reward", "max_points": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["table_rows"]
    assert payload["table_rows"][0]["description"] == "收益奖励 / Profit reward"
    assert payload["chart_series"][0]["points"][0]["description"] == "收益奖励 / Profit reward"


def test_metric_queries_do_not_use_full_table_read_path(tmp_path, monkeypatch):
    run_id = generate_demo_run(
        data_dir=tmp_path,
        run_id="filtered_query_demo",
        vpp_count=2,
        epochs=2,
        days=2,
        steps_per_day=8,
        async_writer=False,
    )

    def fail_if_full_table_is_read(*args, **kwargs):
        raise AssertionError("metric endpoints must use filtered DuckDB queries, not full read_table()")

    monkeypatch.setattr(DuckDBStore, "read_table", fail_if_full_table_is_read)
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        f"/api/runs/{run_id}/dataset",
        params={"vpp_id": "vpp_001", "metrics": "electricity_price,net_load", "max_points": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] > 5
    assert {row["vpp_id"] for row in payload["table_rows"]} == {"vpp_001"}
    assert {row["metric_name"] for row in payload["table_rows"]}.issubset({"electricity_price", "net_load"})


def test_scalars_endpoint_queries_scalar_metrics(tmp_path):
    run_id = generate_demo_run(
        data_dir=tmp_path,
        run_id="scalar_endpoint_demo",
        vpp_count=2,
        epochs=2,
        days=1,
        steps_per_day=3,
        async_writer=False,
    )
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        f"/api/runs/{run_id}/scalars",
        params={"metrics": "epoch_return_mean", "max_points": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] == 2
    assert payload["summary"]["returned_table_rows"] == 2
    assert {row["metric_name"] for row in payload["table_rows"]} == {"epoch_return_mean"}


def test_events_endpoint_queries_recent_event_rows(tmp_path):
    logger = ExperimentLogger(run_id="event_endpoint", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_event("training_status", {"message": "epoch finished"}, epoch_id=1, episode_id=2)
    logger.close()
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get("/api/runs/event_endpoint/events", params={"max_points": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] == 1
    assert payload["table_rows"][0]["metric_group"] == "event"
    assert payload["table_rows"][0]["metric_name"] == "training_status"
    assert payload["table_rows"][0]["value"] == "epoch finished"
    assert payload["table_rows"][0]["epoch_id"] == 1
    assert payload["table_rows"][0]["episode_id"] == 2


def test_events_endpoint_limits_to_latest_logged_events(tmp_path):
    logger = ExperimentLogger(run_id="latest_events", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_event("training_status", {"message": "first event"}, epoch_id=1, episode_id=1)
    logger.log_event("training_status", {"message": "second event"}, epoch_id=2, episode_id=2)
    logger.close()
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get("/api/runs/latest_events/events", params={"max_points": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] == 2
    assert payload["summary"]["returned_table_rows"] == 1
    assert payload["table_rows"][0]["value"] == "second event"
    assert payload["table_rows"][0]["epoch_id"] == 2


def test_losses_endpoint_filters_by_gradient_step_range(tmp_path):
    logger = ExperimentLogger(run_id="loss_gradient_filter", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_loss_terms(
        epoch_id=0,
        episode_id=1,
        gradient_step=1,
        vpp_id="aggregate",
        agent_id="aggregate",
        policy_id="happo",
        terms={"critic_loss": 0.1},
    )
    logger.log_loss_terms(
        epoch_id=0,
        episode_id=1,
        gradient_step=2,
        vpp_id="aggregate",
        agent_id="aggregate",
        policy_id="happo",
        terms={"critic_loss": 0.2},
    )
    logger.close()
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        "/api/runs/loss_gradient_filter/losses",
        params={"start_gradient_step": 2, "end_gradient_step": 2, "metrics": "critic_loss"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] == 1
    assert payload["table_rows"][0]["gradient_step"] == 2
    assert payload["table_rows"][0]["value"] == 0.2


def test_metric_endpoints_filter_by_episode_id(tmp_path):
    logger = ExperimentLogger(run_id="episode_filter", data_dir=tmp_path, config={}, async_writer=False)
    for episode_id, value in [(1, 51.0), (2, 62.0)]:
        logger.log_dataset(
            epoch_id=0,
            episode_id=episode_id,
            env_id="env_0",
            vpp_id="vpp_a",
            date="2018-01-02",
            time_index=0,
            values={"electricity_price": value},
        )
    logger.close()
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        "/api/runs/episode_filter/dataset",
        params={"episode_id": 2, "metrics": "electricity_price", "vpp_id": "vpp_a"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] == 1
    assert payload["table_rows"][0]["episode_id"] == 2
    assert payload["table_rows"][0]["value"] == 62.0


def test_metric_endpoints_filter_by_agent_id(tmp_path):
    logger = ExperimentLogger(run_id="agent_filter", data_dir=tmp_path, config={}, async_writer=False)
    logger.log_reward_terms(
        epoch_id=0,
        episode_id=1,
        vpp_id="vpp_a",
        agent_id="agent_a",
        policy_id="happo",
        terms={"total_reward": 1.0},
    )
    logger.log_reward_terms(
        epoch_id=0,
        episode_id=1,
        vpp_id="vpp_b",
        agent_id="agent_b",
        policy_id="happo",
        terms={"total_reward": 2.0},
    )
    logger.close()
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        "/api/runs/agent_filter/rewards",
        params={"agent_id": "agent_b", "metrics": "total_reward"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] == 1
    assert payload["table_rows"][0]["agent_id"] == "agent_b"
    assert payload["table_rows"][0]["value"] == 2.0


def test_metric_endpoints_filter_dataset_rewards_and_costs_by_policy_alias(tmp_path):
    logger = ExperimentLogger(run_id="policy_endpoint_filter", data_dir=tmp_path, config={}, async_writer=False)
    common = {
        "epoch_id": 0,
        "episode_id": 1,
        "date": "2018-01-01",
        "time_index": 0,
        "vpp_id": "vpp_commercial_multi",
        "agent_id": "vpp_commercial_multi_dispatch",
    }
    for policy_id, offset in [("happo", 0.0), ("paper_long", 100.0)]:
        logger.log_dataset(
            **common,
            policy_id=policy_id,
            values={"electricity_price": 52.0 + offset},
            units={"electricity_price": "currency/MWh"},
        )
        logger.log_reward_terms(
            **common,
            policy_id=policy_id,
            terms={"total_reward": 1.0 + offset},
        )
        logger.log_cost_terms(
            **common,
            policy_id=policy_id,
            terms={"total_cost": 2.0 + offset},
        )
    logger.close()
    client = TestClient(create_app(data_dir=tmp_path))

    for endpoint, metric_name, expected_value in [
        ("dataset", "electricity_price", 52.0),
        ("rewards", "total_reward", 1.0),
        ("costs", "total_cost", 2.0),
    ]:
        response = client.get(
            f"/api/runs/policy_endpoint_filter/{endpoint}",
            params={
                "vpp_id": "vpp_commercial_multi",
                "metrics": metric_name,
                "policy_id": "happo_sequential_ctde",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"]["row_count"] == 1
        assert payload["table_rows"][0]["policy_id"] == "happo_sequential_ctde"
        assert payload["table_rows"][0]["value"] == expected_value


def test_compare_endpoint_limits_returned_rows(tmp_path):
    run_id = generate_demo_run(
        data_dir=tmp_path,
        run_id="limited_compare_demo",
        vpp_count=3,
        epochs=2,
        days=2,
        steps_per_day=8,
        async_writer=False,
    )
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        f"/api/runs/{run_id}/compare",
        params={
            "scope": "reward",
            "metric_names": "total_reward",
            "group_by": "vpp_id",
            "max_points": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] > 3
    assert payload["summary"]["returned_table_rows"] == 3
    assert len(payload["table_rows"]) == 3


def test_compare_endpoint_filters_to_selected_group_values(tmp_path):
    logger = ExperimentLogger(run_id="compare_group_filter", data_dir=tmp_path, config={}, async_writer=False)
    for vpp_id, value in [("vpp_a", 1.0), ("vpp_b", 2.0), ("vpp_c", 3.0)]:
        logger.log_dataset(
            epoch_id=0,
            episode_id=1,
            date="2018-01-01",
            time_index=0,
            vpp_id=vpp_id,
            agent_id=f"{vpp_id}_dispatch",
            policy_id="happo",
            values={"net_load": value},
            units={"net_load": "MW"},
        )
    logger.close()
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        "/api/runs/compare_group_filter/compare",
        params={
            "scope": "dataset",
            "fixed_epoch_id": 0,
            "fixed_date": "2018-01-01",
            "fixed_time_index": 0,
            "metric_names": "net_load",
            "group_by": "vpp_id",
            "group_values": "vpp_b,vpp_c",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert {row["group"] for row in payload["table_rows"]} == {"vpp_b", "vpp_c"}
    assert {row["vpp_id"] for row in payload["table_rows"]} == {"vpp_b", "vpp_c"}
    assert payload["summary"]["selected_group_values"] == ["vpp_b", "vpp_c"]


def test_compare_endpoint_filters_by_fixed_episode_id(tmp_path):
    logger = ExperimentLogger(run_id="compare_episode_filter", data_dir=tmp_path, config={}, async_writer=False)
    for episode_id, value in [(1, 10.0), (2, 12.0)]:
        logger.log_dataset(
            epoch_id=0,
            episode_id=episode_id,
            date="2018-01-01",
            time_index=0,
            vpp_id="vpp_a",
            agent_id="vpp_a_dispatch",
            policy_id="happo",
            values={"net_load": value},
            units={"net_load": "MW"},
        )
    logger.close()
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get(
        "/api/runs/compare_episode_filter/compare",
        params={
            "scope": "dataset",
            "fixed_epoch_id": 0,
            "fixed_episode_id": 1,
            "fixed_date": "2018-01-01",
            "fixed_time_index": 0,
            "metric_names": "net_load",
            "group_by": "vpp_id",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["fixed_episode_id"] == 1
    assert len(payload["table_rows"]) == 1
    assert payload["table_rows"][0]["episode_id"] == 1
    assert payload["table_rows"][0]["value"] == 10.0


def test_dashboard_api_serves_demo_run_endpoints(tmp_path):
    run_id = generate_demo_run(
        data_dir=tmp_path,
        run_id="demo_test",
        vpp_count=2,
        epochs=2,
        days=2,
        steps_per_day=4,
        async_writer=False,
    )
    client = TestClient(create_app(data_dir=tmp_path))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["data_dir"] == str(tmp_path)

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert [run["run_id"] for run in runs.json()] == [run_id]

    selectors = client.get(f"/api/runs/{run_id}/selectors")
    assert selectors.status_code == 200
    assert selectors.json()["vpp_ids"] == ["vpp_001", "vpp_002"]
    assert selectors.json()["epoch_ids"] == [0, 1]

    variables = client.get(f"/api/runs/{run_id}/variables")
    assert variables.status_code == 200
    names = {item["name"] for item in variables.json()}
    assert {"electricity_price", "storage_soc", "total_reward", "actor_loss"}.issubset(names)

    dataset = client.get(
        f"/api/runs/{run_id}/dataset",
        params={"epoch_id": 0, "vpp_id": "vpp_001", "metrics": "electricity_price,net_load"},
    )
    assert dataset.status_code == 200
    payload = dataset.json()
    assert payload["summary"]["row_count"] > 0
    assert set(payload["units"]).issuperset({"electricity_price", "net_load"})

    rewards = client.get(
        f"/api/runs/{run_id}/rewards",
        params={"epoch_id": 0, "vpp_id": "vpp_001", "metrics": "total_reward"},
    )
    assert rewards.status_code == 200
    assert rewards.json()["summary"]["row_count"] > 0

    compare = client.get(
        f"/api/runs/{run_id}/compare",
        params={
            "scope": "dataset",
            "fixed_epoch_id": 0,
            "fixed_date": "2026-01-01",
            "fixed_time_index": 0,
            "metric_names": "net_load",
            "group_by": "vpp_id",
        },
    )
    assert compare.status_code == 200
    compare_rows = compare.json()["table_rows"]
    assert len(compare_rows) == 2
    assert {row["group"] for row in compare_rows} == {"vpp_001", "vpp_002"}
    assert {row["vpp_id"] for row in compare_rows} == {"vpp_001", "vpp_002"}
