from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from marl_dashboard.backend.api.websocket import register_websocket_routes
from marl_dashboard.backend.schemas.common import QueryResponse
from marl_dashboard.backend.schemas.run import RunSummary
from marl_dashboard.backend.schemas.selector import SelectorOptions
from marl_dashboard.backend.schemas.variable import VariableDefinition
from marl_dashboard.backend.storage.query_service import QueryService, TABLE_BY_SCOPE
from marl_dashboard.backend.storage.static_topology import StaticTopologyService


def create_app(data_dir: str | Path = "runs") -> FastAPI:
    resolved_data_dir = Path(data_dir).expanduser().resolve()
    app = FastAPI(title="MARL VPP Dashboard", version="0.1.0")
    app.state.data_dir = resolved_data_dir
    app.state.query_service = QueryService(resolved_data_dir)
    app.state.static_topology_service = StaticTopologyService(resolved_data_dir)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1", "http://localhost"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=dict[str, Any])
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": "0.1.0", "data_dir": str(resolved_data_dir)}

    @app.get("/api/runs", response_model=list[RunSummary])
    def runs() -> list[dict[str, Any]]:
        return app.state.query_service.runs()

    @app.get("/api/runs/{run_id}/metadata", response_model=dict[str, Any])
    def metadata(run_id: str) -> dict[str, Any]:
        return app.state.query_service.metadata(run_id)

    @app.get("/api/runs/{run_id}/variables", response_model=list[VariableDefinition])
    def variables(run_id: str) -> list[dict[str, Any]]:
        return app.state.query_service.variables(run_id)

    @app.get("/api/runs/{run_id}/selectors", response_model=SelectorOptions)
    def selectors(run_id: str) -> dict[str, Any]:
        return app.state.query_service.selectors(run_id)

    @app.get("/api/runs/{run_id}/dataset", response_model=QueryResponse)
    def dataset(
        run_id: str,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        policy_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        metrics: str | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        return app.state.query_service.query_metric_table(
            run_id=run_id,
            table="dataset_timeseries",
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=date,
            vpp_id=vpp_id,
            agent_id=agent_id,
            policy_id=policy_id,
            start_time_index=start_time_index,
            end_time_index=end_time_index,
            max_points=max_points,
        )

    @app.get("/api/runs/{run_id}/rewards", response_model=QueryResponse)
    def rewards(
        run_id: str,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        policy_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        metrics: str | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        return app.state.query_service.query_metric_table(
            run_id=run_id,
            table="reward_terms",
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=date,
            vpp_id=vpp_id,
            agent_id=agent_id,
            policy_id=policy_id,
            start_time_index=start_time_index,
            end_time_index=end_time_index,
            max_points=max_points,
        )

    @app.get("/api/runs/{run_id}/costs", response_model=QueryResponse)
    def costs(
        run_id: str,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        policy_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        metrics: str | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        return app.state.query_service.query_metric_table(
            run_id=run_id,
            table="cost_terms",
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=date,
            vpp_id=vpp_id,
            agent_id=agent_id,
            policy_id=policy_id,
            start_time_index=start_time_index,
            end_time_index=end_time_index,
            max_points=max_points,
        )

    @app.get("/api/runs/{run_id}/losses", response_model=QueryResponse)
    def losses(
        run_id: str,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        policy_id: str | None = None,
        gradient_step: int | None = None,
        start_gradient_step: int | None = None,
        end_gradient_step: int | None = None,
        metrics: str | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        return app.state.query_service.query_metric_table(
            run_id=run_id,
            table="loss_terms",
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            vpp_id=vpp_id,
            agent_id=agent_id,
            policy_id=policy_id,
            gradient_step=gradient_step,
            start_gradient_step=start_gradient_step,
            end_gradient_step=end_gradient_step,
            max_points=max_points,
        )

    @app.get("/api/runs/{run_id}/scalars", response_model=QueryResponse)
    def scalars(
        run_id: str,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        policy_id: str | None = None,
        metrics: str | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        return app.state.query_service.query_metric_table(
            run_id=run_id,
            table="scalar_metrics",
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            vpp_id=vpp_id,
            agent_id=agent_id,
            policy_id=policy_id,
            max_points=max_points,
        )

    @app.get("/api/runs/{run_id}/events", response_model=QueryResponse)
    def events(
        run_id: str,
        epoch_id: int | None = None,
        episode_id: int | None = None,
        date: str | None = None,
        vpp_id: str | None = None,
        agent_id: str | None = None,
        start_time_index: int | None = None,
        end_time_index: int | None = None,
        policy_id: str | None = None,
        metrics: str | None = None,
        max_points: int = 200,
    ) -> dict[str, Any]:
        return app.state.query_service.query_metric_table(
            run_id=run_id,
            table="events",
            metrics=metrics,
            epoch_id=epoch_id,
            episode_id=episode_id,
            date=date,
            vpp_id=vpp_id,
            agent_id=agent_id,
            start_time_index=start_time_index,
            end_time_index=end_time_index,
            policy_id=policy_id,
            max_points=max_points,
            latest_first=True,
        )

    @app.get("/api/runs/{run_id}/compare", response_model=QueryResponse)
    def compare(
        run_id: str,
        scope: str = Query(pattern="^(dataset|reward|cost|loss)$"),
        fixed_epoch_id: int | None = None,
        fixed_episode_id: int | None = None,
        fixed_date: str | None = None,
        fixed_time_index: int | None = None,
        metric_names: str = "",
        group_by: str = Query(default="vpp_id", pattern="^(vpp_id|epoch_id|policy_id|agent_id)$"),
        group_values: str | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        return app.state.query_service.compare(
            run_id=run_id,
            table=TABLE_BY_SCOPE[scope],
            metric_names=metric_names,
            group_by=group_by,
            group_values=group_values,
            fixed_epoch_id=fixed_epoch_id,
            fixed_episode_id=fixed_episode_id,
            fixed_date=fixed_date,
            fixed_time_index=fixed_time_index,
            max_points=max_points,
        )

    @app.get("/api/runs/{run_id}/formulas", response_model=dict[str, Any])
    def formulas(run_id: str) -> dict[str, Any]:
        return app.state.query_service.formulas(run_id)

    @app.get("/api/runs/{run_id}/topology", response_model=dict[str, Any])
    def topology(run_id: str) -> dict[str, Any]:
        try:
            return app.state.static_topology_service.topology(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/vpp-config", response_model=dict[str, Any])
    def vpp_config(run_id: str) -> dict[str, Any]:
        try:
            return app.state.static_topology_service.vpp_config(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    register_websocket_routes(app)
    _register_frontend(app)
    return app


def _register_frontend(app: FastAPI) -> None:
    frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    assets_dir = frontend_dist / "assets"
    index_file = frontend_dist / "index.html"
    if not index_file.exists():
        return
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="dashboard-assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(index_file)

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        if path.startswith("api/") or path.startswith("ws/"):
            raise HTTPException(status_code=404)
        file_path = frontend_dist / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(index_file)
