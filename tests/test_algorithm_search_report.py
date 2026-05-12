from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vpp_dso_sim.visualization.algorithm_search_report import export_algorithm_search_report


def test_algorithm_search_report_includes_bilingual_title_top_candidate_and_all_rows():
    output_dir = Path("outputs") / "test_algorithm_search_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = [
        "HAPPO",
        "MATD3",
        "HASAC",
        "FACMAC",
        "MAPPO",
        "IPPO",
        "MADDPG",
        "QMIX",
        "VDN",
        "COMA",
        "MASAC",
        "MAAC",
        "HATRPO",
        "MAVEN",
        "ROMA",
        "LIIR",
        "DOP",
        "QPLEX",
        "IQL",
        "DDPG",
        "SAC",
        "PPO",
    ]
    rows = []
    for index, algorithm in enumerate(algorithms):
        rows.append(
            {
                "algorithm": algorithm,
                "score": 100.0 - index,
                "decision": "keep" if index < 5 else "defer",
                "keep_reason": f"{algorithm} keeps privacy-aware CTDE evaluation visible.",
                "reject_reason": f"{algorithm} still needs safety-projection and settlement checks.",
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "candidate_scores.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "search_objective": "advanced MARL candidate screening",
                "selection_policy": "rank by score, then review privacy and FR/DOE fit",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    path = export_algorithm_search_report(output_dir)

    assert path == output_dir / "algorithm_search_report.html"
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "算法搜索与高级 MARL 候选报告" in html
    assert "HAPPO" in html
    assert "最高优先候选" in html
    assert "MATD3" in html
    assert "HASAC" in html
    assert "FACMAC" in html
    assert html.count("data-testid='candidate-row'") >= 20


def test_algorithm_search_report_understands_real_algorithm_search_columns():
    output_dir = Path("outputs") / "test_algorithm_search_report_real_columns"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank in range(1, 8):
        rows.append(
            {
                "rank": rank,
                "algorithm_id": f"candidate_{rank}",
                "family": "MAPPO",
                "recommendation_status": "recommended" if rank <= 7 else "rejected_for_now",
                "proxy_score": 0.9 - rank * 0.01,
                "reward_fit": 0.9,
                "privacy_fit": 0.9,
                "continuous_action_fit": 0.9,
                "heterogeneity_fit": 0.8,
                "risk_penalty": 0.2,
                "expected_engineering_lift": 0.2,
                "keep_reason": "real search output keep reason",
                "rejection_reason": "",
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "candidate_scores.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps({"candidate_count": 7, "top_k": 7}, ensure_ascii=False),
        encoding="utf-8",
    )

    path = export_algorithm_search_report(output_dir, top_n=7)
    html = path.read_text(encoding="utf-8")

    assert "candidate_7" in html
    assert "Keep" in html
    assert "保留" in html
    assert "代理评分" in html
