"""Analyze blinded CLARK P1-P5 probe results and assign recovery mechanisms."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CONDITIONS = (
    "p1_natural",
    "p2_support_presence",
    "p3_support_first",
    "p4_evidence_only",
    "p5_fact_card",
)
SHORT = {
    "p1_natural": "P1 natural",
    "p2_support_presence": "P2 support present",
    "p3_support_first": "P3 support first",
    "p4_evidence_only": "P4 evidence only",
    "p5_fact_card": "P5 fact card",
}
COHORT_ORDER = (
    "new_degradation",
    "persistent_failure",
    "adaptive_control",
    "normal_control",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs" / "runs" / "clark_historical_failure_probe_luna",
    )
    parser.add_argument("--recovery-delta", type=float, default=0.25)
    parser.add_argument("--recovery-accuracy", type=float, default=0.50)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bootstrap_mean_ci(
    values: np.ndarray, rounds: int, seed: int
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(rounds, values.size), replace=True)
    means = sampled.mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def dataset_metadata(dataset_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(dataset_path):
        metadata = record.get("metadata") or {}
        intervention = metadata.get("probe_intervention") or {}
        rows.append(
            {
                "question_id": record["id"],
                "question": record["question"],
                "old_answer": record.get("stale_answer"),
                "current_answer": record.get("gold_answer"),
                "time_x": metadata.get("time_x"),
                "time_y": metadata.get("time_y"),
                "cohort_group": metadata.get("historical_cohort_group"),
                "historical_outcome_state": metadata.get("historical_outcome_state"),
                "historical_accuracy_stale": metadata.get("historical_accuracy_stale"),
                "historical_accuracy_current": metadata.get("historical_accuracy_current"),
                "historical_accuracy_drop": metadata.get("historical_accuracy_drop"),
                "natural_support_hit": intervention.get("natural_support_hit"),
                "natural_support_rank": intervention.get("natural_support_rank"),
                "p2_support_injected": intervention.get("p2_support_injected"),
                "p2_changed_from_p1": intervention.get("p2_changed_from_p1"),
                "p3_support_moved_to_rank1": intervention.get(
                    "p3_support_moved_to_rank1"
                ),
            }
        )
    return pd.DataFrame(rows)


def pivot_metric(metrics: pd.DataFrame, name: str) -> pd.DataFrame:
    table = metrics.pivot(index="question_id", columns="condition", values=name)
    table = table.reindex(columns=CONDITIONS)
    table.columns = [f"{name}__{condition}" for condition in table.columns]
    return table.reset_index()


def distribution_wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["question_id"])
    distances = pd.read_csv(path)
    pieces: list[pd.DataFrame] = []
    for condition in CONDITIONS[1:]:
        comparison = f"{condition}__vs__p1_natural"
        subset = distances[distances["comparison"] == comparison].copy()
        if subset.empty:
            continue
        keep = [
            "question_id",
            "swd",
            "mmd_rbf",
            "energy",
            "centroid_gap",
            "cluster_js",
        ]
        subset = subset[keep].rename(
            columns={name: f"{name}__{condition}" for name in keep if name != "question_id"}
        )
        pieces.append(subset)
    if not pieces:
        return pd.DataFrame(columns=["question_id"])
    output = pieces[0]
    for piece in pieces[1:]:
        output = output.merge(piece, on="question_id", how="outer")
    return output


def recovered(
    frame: pd.DataFrame,
    condition: str,
    delta_threshold: float,
    accuracy_threshold: float,
) -> pd.Series:
    baseline = pd.to_numeric(frame["accuracy__p1_natural"], errors="coerce")
    current = pd.to_numeric(frame[f"accuracy__{condition}"], errors="coerce")
    return ((current - baseline) >= delta_threshold) & (current >= accuracy_threshold)


def assign_mechanism(row: pd.Series, recovery_columns: dict[str, str]) -> tuple[str, str]:
    if float(row.get("accuracy__p1_natural", 0.0)) >= 0.5:
        return ("p1_natural", "no_failure_on_rerun")
    if bool(row.get(recovery_columns["p2_support_presence"], False)) and bool(
        row.get("p2_support_injected", False)
    ):
        return ("p2_support_presence", "retrieval_coverage_failure")
    if bool(row.get(recovery_columns["p3_support_first"], False)) and bool(
        row.get("p3_support_moved_to_rank1", False)
    ):
        return ("p3_support_first", "ranking_or_position_sensitivity")
    if bool(row.get(recovery_columns["p4_evidence_only"], False)):
        return (
            "p4_evidence_only",
            "evidence_extraction_or_context_complexity_failure",
        )
    if bool(row.get(recovery_columns["p5_fact_card"], False)):
        return (
            "p5_fact_card",
            "evidence_utilization_or_answer_realization_failure",
        )
    ineffective_recovery = any(
        bool(row.get(recovery_columns[condition], False))
        for condition in ("p2_support_presence", "p3_support_first")
    )
    if ineffective_recovery:
        return ("none", "stochastic_recovery_without_context_change")
    return ("none", "persistent_after_explicit_fact")


def condition_summary(
    frame: pd.DataFrame, rounds: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in COHORT_ORDER:
        subset = frame[frame["cohort_group"] == group]
        if subset.empty:
            continue
        baseline = pd.to_numeric(subset["accuracy__p1_natural"], errors="coerce")
        for index, condition in enumerate(CONDITIONS):
            accuracy = pd.to_numeric(subset[f"accuracy__{condition}"], errors="coerce")
            delta = accuracy - baseline
            low, high = bootstrap_mean_ci(delta.to_numpy(float), rounds, seed + index)
            rows.append(
                {
                    "cohort_group": group,
                    "condition": condition,
                    "n_questions": int(len(subset)),
                    "mean_accuracy": float(accuracy.mean()),
                    "mean_delta_vs_p1": float(delta.mean()),
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "mean_semantic_entropy": float(
                        pd.to_numeric(
                            subset[f"semantic_entropy__{condition}"], errors="coerce"
                        ).mean()
                    ),
                    "mean_semantic_volume": float(
                        pd.to_numeric(
                            subset[f"semantic_volume__{condition}"], errors="coerce"
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for delta_threshold in (0.10, 0.25, 0.50):
        for accuracy_threshold in (0.50, 0.75):
            for group in COHORT_ORDER:
                subset = frame[frame["cohort_group"] == group]
                if subset.empty:
                    continue
                any_recovery = pd.Series(False, index=subset.index)
                for condition in CONDITIONS[1:]:
                    any_recovery |= recovered(
                        subset, condition, delta_threshold, accuracy_threshold
                    )
                rows.append(
                    {
                        "delta_threshold": delta_threshold,
                        "accuracy_threshold": accuracy_threshold,
                        "cohort_group": group,
                        "n_questions": int(len(subset)),
                        "recovered_questions": int(any_recovery.sum()),
                        "recovery_rate": float(any_recovery.mean()),
                    }
                )
    return pd.DataFrame(rows)


def make_figures(frame: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "new_degradation": "#c94c4c",
        "persistent_failure": "#7f3c8d",
        "adaptive_control": "#2f7f72",
        "normal_control": "#4c78a8",
    }
    fig, axis = plt.subplots(figsize=(10.8, 5.8))
    for group in COHORT_ORDER:
        subset = summary[summary["cohort_group"] == group]
        if subset.empty:
            continue
        subset = subset.set_index("condition").reindex(CONDITIONS)
        axis.plot(
            range(len(CONDITIONS)),
            subset["mean_accuracy"],
            marker="o",
            linewidth=2.2,
            label=group.replace("_", " "),
            color=colors[group],
        )
    axis.set_xticks(range(len(CONDITIONS)), [SHORT[value] for value in CONDITIONS])
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Mean current-answer accuracy")
    axis.set_title("CLARK diagnostic probe: accuracy by evidence intervention")
    axis.grid(axis="y", color="#d9dee7", linewidth=0.8)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_probe_stage.png", dpi=180)
    plt.close(fig)

    stage_order = (*CONDITIONS, "none")
    counts = pd.crosstab(frame["cohort_group"], frame["earliest_recovery_stage"])
    counts = counts.reindex(index=COHORT_ORDER, columns=stage_order, fill_value=0)
    fig, axis = plt.subplots(figsize=(10.8, 5.8))
    bottom = np.zeros(len(counts), dtype=float)
    stage_colors = ["#7aa6c2", "#2f7f72", "#e3a02b", "#d87535", "#c94c4c", "#6b7280"]
    for stage, color in zip(stage_order, stage_colors, strict=True):
        values = counts[stage].to_numpy(float)
        axis.bar(
            [label.replace("_", " ") for label in counts.index],
            values,
            bottom=bottom,
            label=SHORT.get(stage, stage),
            color=color,
        )
        bottom += values
    axis.set_ylabel("Questions")
    axis.set_title("Earliest recovery stage by historical outcome")
    axis.legend(frameon=False, ncol=3, fontsize=8)
    axis.grid(axis="y", color="#d9dee7", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "earliest_recovery_stage.png", dpi=180)
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    delta_threshold: float,
    accuracy_threshold: float,
) -> None:
    group_rows: list[str] = []
    for group in COHORT_ORDER:
        subset = frame[frame["cohort_group"] == group]
        if subset.empty:
            continue
        p1_values = pd.to_numeric(subset["accuracy__p1_natural"], errors="coerce")
        p1 = float(p1_values.mean())
        p5 = float(pd.to_numeric(subset["accuracy__p5_fact_card"], errors="coerce").mean())
        p1_failure = p1_values < 0.5
        recovered_after_p1 = subset["earliest_recovery_stage"].isin(CONDITIONS[1:])
        recovered_count = int((p1_failure & recovered_after_p1).sum())
        p1_failure_count = int(p1_failure.sum())
        group_rows.append(
            f"| {group} | {len(subset)} | {p1:.3f} | {p5:.3f} | "
            f"{p1_failure_count} | {recovered_count}/{p1_failure_count} |"
        )
    mechanism_counts = Counter(frame["mechanism"].astype(str))
    mechanism_lines = [f"- `{name}`: {count}" for name, count in mechanism_counts.most_common()]
    historical_failure = frame[frame["cohort_group"].isin(["new_degradation", "persistent_failure"])]
    rerun_failure_rate = float(
        (pd.to_numeric(historical_failure["accuracy__p1_natural"], errors="coerce") < 0.5).mean()
    )
    lines = [
        "# CLARK 과거 실패 문항 진단 Probe 결과",
        "",
        "## 판정 규칙",
        "",
        f"- 회복: P1 대비 accuracy가 `{delta_threshold:.2f}` 이상 증가하고, 해당 조건 accuracy가 `{accuracy_threshold:.2f}` 이상",
        "- P1~P5 조건명은 모델 입력에 노출하지 않고 결과 메타데이터에서만 사용",
        "- 모든 조건의 질문, 평가 시점, 시스템 프롬프트, 샘플링 설정은 동일",
        "",
        "## 핵심 요약",
        "",
        f"- 전체 문항: {len(frame)}",
        f"- 과거 실패군의 P1 재실행 실패 재현율: {rerun_failure_rate:.3f}",
        "",
        "| 과거 상태 | N | P1 accuracy | P5 accuracy | P1 실패 수 | P1 실패 후 회복 |",
        "|---|---:|---:|---:|---:|---:|",
        *group_rows,
        "",
        "## 최초 회복 단계 기반 진단",
        "",
        *mechanism_lines,
        "",
        "- P2에서 회복: 자연 top-k에 빠진 최신 근거를 넣으면 회복하는 retrieval coverage 문제",
        "- P3에서 회복: 같은 근거를 맨 앞으로 옮겼을 때 회복하는 ranking/position 문제",
        "- P4에서 회복: 핵심 evidence만 남겼을 때 회복하는 extraction 또는 context complexity 문제",
        "- P5에서 회복: 정답 사실을 명시해야 회복하는 evidence utilization 또는 answer realization 문제",
        "- P5에서도 미회복: 모델 지시 준수, 평가기, 데이터 연결을 추가 점검해야 하는 지속 실패",
        "",
        "## 산출물",
        "",
        "- `per_question_probe_results.csv`: 문항별 P1~P5 정확도, uncertainty, shift, 최초 회복 단계",
        "- `condition_summary.csv`: 코호트·조건별 평균과 bootstrap CI",
        "- `recovery_threshold_sensitivity.csv`: 회복 임계값 민감도",
        "- `mechanism_counts.csv`: 코호트별 진단 결과 수",
        "- `accuracy_by_probe_stage.png`, `earliest_recovery_stage.png`: 논문용 요약 그림",
    ]
    (output_dir / "report_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "probe_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / str(config["dataset"]["path"])
    metadata = dataset_metadata(dataset_path)
    metrics = pd.read_csv(run_dir / "metrics" / "question_level_metrics.csv")
    metrics = metrics[metrics["embedding_model_role"] == "primary"].copy()
    frame = metadata.copy()
    for metric_name in (
        "accuracy",
        "semantic_entropy",
        "semantic_volume",
        "centroid_shift",
        "current_answer_rate",
        "stale_answer_rate",
        "harmful_other_rate",
        "contradiction_rate",
    ):
        frame = frame.merge(pivot_metric(metrics, metric_name), on="question_id", how="left")
    frame = frame.merge(
        distribution_wide(
            run_dir / "distribution_shift" / "per_question_distribution_shift.csv"
        ),
        on="question_id",
        how="left",
    )

    recovery_columns: dict[str, str] = {}
    for condition in CONDITIONS[1:]:
        column = f"recovered__{condition}"
        frame[column] = recovered(
            frame, condition, args.recovery_delta, args.recovery_accuracy
        )
        recovery_columns[condition] = column
        frame[f"accuracy_delta__{condition}"] = (
            pd.to_numeric(frame[f"accuracy__{condition}"], errors="coerce")
            - pd.to_numeric(frame["accuracy__p1_natural"], errors="coerce")
        )
        frame[f"entropy_delta__{condition}"] = (
            pd.to_numeric(frame[f"semantic_entropy__{condition}"], errors="coerce")
            - pd.to_numeric(frame["semantic_entropy__p1_natural"], errors="coerce")
        )
        frame[f"volume_delta__{condition}"] = (
            pd.to_numeric(frame[f"semantic_volume__{condition}"], errors="coerce")
            - pd.to_numeric(frame["semantic_volume__p1_natural"], errors="coerce")
        )
    assigned = frame.apply(
        lambda row: assign_mechanism(row, recovery_columns), axis=1, result_type="expand"
    )
    assigned.columns = ["earliest_recovery_stage", "mechanism"]
    frame = pd.concat([frame, assigned], axis=1)

    summary = condition_summary(frame, args.bootstrap_rounds, args.seed)
    sensitivity = sensitivity_summary(frame)
    mechanism_counts = (
        frame.groupby(["cohort_group", "earliest_recovery_stage", "mechanism"])
        .size()
        .reset_index(name="n_questions")
    )
    frame.to_csv(output_dir / "per_question_probe_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "condition_summary.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(
        output_dir / "recovery_threshold_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    mechanism_counts.to_csv(
        output_dir / "mechanism_counts.csv", index=False, encoding="utf-8-sig"
    )
    make_figures(frame, summary, output_dir)
    write_report(
        frame,
        summary,
        output_dir,
        args.recovery_delta,
        args.recovery_accuracy,
    )
    print(output_dir / "report_ko.md")


if __name__ == "__main__":
    main()
