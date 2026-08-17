"""Analyze frozen-detector screening and P1-P5 localization as one workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_clark_historical_failure_probe import CONDITIONS


DEFAULT_RUN_DIR = ROOT / "outputs" / "runs" / "clark_detector_linked_probe_luna"
DEFAULT_PREDICTIONS = (
    ROOT
    / "outputs"
    / "runs"
    / "clark_t0_temporal_transfer_luna"
    / "core4_only_multimodel_transfer"
    / "frozen_future_predictions.csv"
)
GROUP_ORDER = (
    "detector_true_positive",
    "detector_false_negative",
    "detector_false_positive",
    "detector_true_negative_control",
)
MECHANISM_ORDER = (
    "retrieval_coverage_failure",
    "ranking_or_position_sensitivity",
    "evidence_extraction_or_context_complexity_failure",
    "evidence_utilization_or_answer_realization_failure",
    "persistent_after_explicit_fact",
    "stochastic_recovery_without_context_change",
    "no_degradation_on_probe_rerun",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--detector", default="additive_gam")
    parser.add_argument("--degradation-delta", type=float, default=0.10)
    parser.add_argument("--recovery-gain", type=float, default=0.10)
    parser.add_argument(
        "--recovery-tolerance",
        type=float,
        default=0.0625,
        help="Allowed residual loss versus pre-update accuracy; 1/16 by default.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metadata_frame(dataset_path: Path) -> pd.DataFrame:
    rows = []
    for record in read_jsonl(dataset_path):
        metadata = record.get("metadata") or {}
        intervention = metadata.get("probe_intervention") or {}
        rows.append(
            {
                "question_id": record["id"],
                "question": record["question"],
                "confusion_group": metadata.get("detector_confusion_group"),
                "transition_id": metadata.get("transition_id"),
                "time_x": metadata.get("time_x"),
                "time_y": metadata.get("time_y"),
                "historical_accuracy_stale": metadata.get("historical_accuracy_stale"),
                "historical_accuracy_current": metadata.get("historical_accuracy_current"),
                "historical_accuracy_drop": metadata.get("historical_accuracy_drop"),
                "target_new_degradation": int(bool(metadata.get("target_new_degradation"))),
                "detector_risk": metadata.get("detector_risk"),
                "detector_alarm": int(bool(metadata.get("detector_alarm"))),
                "natural_support_hit": intervention.get("natural_support_hit"),
                "natural_support_rank": intervention.get("natural_support_rank"),
                "p2_support_injected": intervention.get("p2_support_injected"),
                "p2_changed_from_p1": intervention.get("p2_changed_from_p1"),
                "p3_support_moved_to_rank1": intervention.get("p3_support_moved_to_rank1"),
            }
        )
    return pd.DataFrame(rows)


def pivot_metric(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = metrics.pivot(index="question_id", columns="condition", values=metric)
    table = table.reindex(columns=CONDITIONS)
    table.columns = [f"{metric}__{condition}" for condition in CONDITIONS]
    return table.reset_index()


def add_recovery_labels(
    frame: pd.DataFrame,
    degradation_delta: float,
    recovery_gain: float,
    recovery_tolerance: float,
) -> pd.DataFrame:
    output = frame.copy()
    old_accuracy = pd.to_numeric(output["historical_accuracy_stale"], errors="coerce")
    p1 = pd.to_numeric(output["accuracy__p1_natural"], errors="coerce")
    output["probe_accuracy_drop"] = old_accuracy - p1
    output["degradation_reproduced"] = output["probe_accuracy_drop"] >= degradation_delta
    output["recovery_target_accuracy"] = np.maximum(
        0.0, old_accuracy - recovery_tolerance
    )
    recovery_columns: dict[str, str] = {}
    for condition in CONDITIONS[1:]:
        accuracy = pd.to_numeric(output[f"accuracy__{condition}"], errors="coerce")
        column = f"recovered__{condition}"
        output[column] = (
            output["degradation_reproduced"]
            & ((accuracy - p1) >= recovery_gain)
            & (accuracy >= output["recovery_target_accuracy"])
        )
        output[f"accuracy_gain__{condition}"] = accuracy - p1
        output[f"residual_drop__{condition}"] = old_accuracy - accuracy
        recovery_columns[condition] = column

    def assign(row: pd.Series) -> tuple[str, str]:
        if not bool(row["degradation_reproduced"]):
            return "p1_natural", "no_degradation_on_probe_rerun"
        if bool(row[recovery_columns["p2_support_presence"]]):
            if bool(row.get("p2_support_injected", False)):
                return "p2_support_presence", "retrieval_coverage_failure"
            return "p2_support_presence", "stochastic_recovery_without_context_change"
        if bool(row[recovery_columns["p3_support_first"]]):
            if bool(row.get("p3_support_moved_to_rank1", False)):
                return "p3_support_first", "ranking_or_position_sensitivity"
            return "p3_support_first", "stochastic_recovery_without_context_change"
        if bool(row[recovery_columns["p4_evidence_only"]]):
            return (
                "p4_evidence_only",
                "evidence_extraction_or_context_complexity_failure",
            )
        if bool(row[recovery_columns["p5_fact_card"]]):
            return (
                "p5_fact_card",
                "evidence_utilization_or_answer_realization_failure",
            )
        return "none", "persistent_after_explicit_fact"

    assigned = output.apply(assign, axis=1, result_type="expand")
    assigned.columns = ["earliest_recovery_stage", "mechanism_candidate"]
    return pd.concat([output, assigned], axis=1)


def screening_summary(predictions: pd.DataFrame, detector: str) -> pd.DataFrame:
    y = predictions["target_new_degradation"].astype(int).to_numpy()
    risk = predictions[f"risk__{detector}"].astype(float).to_numpy()
    alarm = predictions[f"alarm__{detector}"].astype(str).str.lower().eq("true").to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, alarm, labels=[0, 1]).ravel()
    prevalence = float(y.mean())
    alarm_rate = float(alarm.mean())
    positive_alarm_rate = float(y[alarm].mean()) if alarm.any() else 0.0
    return pd.DataFrame(
        [
            {
                "detector": detector,
                "future_n": len(y),
                "future_positive": int(y.sum()),
                "prevalence": prevalence,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "auroc": float(roc_auc_score(y, risk)),
                "auprc": float(average_precision_score(y, risk)),
                "precision": float(precision_score(y, alarm, zero_division=0)),
                "recall": float(recall_score(y, alarm, zero_division=0)),
                "f1": float(f1_score(y, alarm, zero_division=0)),
                "alarm_rate": alarm_rate,
                "risk_lift": positive_alarm_rate / prevalence if prevalence else float("nan"),
            }
        ]
    )


def make_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    short = {
        "p1_natural": "P1 natural",
        "p2_support_presence": "P2 support",
        "p3_support_first": "P3 rank 1",
        "p4_evidence_only": "P4 evidence",
        "p5_fact_card": "P5 fact card",
    }
    colors = {
        "detector_true_positive": "#C74440",
        "detector_false_negative": "#E39C34",
        "detector_false_positive": "#6E59A5",
        "detector_true_negative_control": "#247A6B",
    }
    figure, axis = plt.subplots(figsize=(10.8, 5.9))
    for group in GROUP_ORDER:
        part = frame[frame["confusion_group"] == group]
        if part.empty:
            continue
        values = [
            pd.to_numeric(part[f"accuracy__{condition}"], errors="coerce").mean()
            for condition in CONDITIONS
        ]
        axis.plot(
            range(len(CONDITIONS)),
            values,
            marker="o",
            linewidth=2.2,
            label=group.replace("detector_", "").replace("_", " "),
            color=colors[group],
        )
    axis.set_xticks(range(len(CONDITIONS)), [short[value] for value in CONDITIONS])
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Mean current-answer accuracy")
    axis.set_title("Detector-linked CLARK probe: accuracy by intervention")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_by_detector_group_and_probe.png", dpi=220)
    plt.close(figure)

    counts = pd.crosstab(frame["confusion_group"], frame["mechanism_candidate"])
    counts = counts.reindex(index=GROUP_ORDER, columns=MECHANISM_ORDER, fill_value=0)
    figure, axis = plt.subplots(figsize=(12.5, 6.2))
    bottom = np.zeros(len(counts), dtype=float)
    palette = ["#2C7FB8", "#41AB5D", "#F0A43A", "#D95F59", "#7A5195", "#A0A0A0", "#D9D9D9"]
    for mechanism, color in zip(MECHANISM_ORDER, palette, strict=True):
        values = counts[mechanism].to_numpy(dtype=float)
        axis.bar(
            [name.replace("detector_", "").replace("_", " ") for name in counts.index],
            values,
            bottom=bottom,
            label=mechanism.replace("_", " "),
            color=color,
        )
        bottom += values
    axis.set_ylabel("Probe events")
    axis.set_title("Candidate failure location by detector outcome")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    figure.tight_layout()
    figure.savefig(output_dir / "mechanism_by_detector_group.png", dpi=220)
    plt.close(figure)


def write_report(
    frame: pd.DataFrame,
    screening: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    score = screening.iloc[0]
    group_rows = []
    for group in GROUP_ORDER:
        part = frame[frame["confusion_group"] == group]
        if part.empty:
            continue
        reproduced = int(part["degradation_reproduced"].sum())
        localized = int(
            part["earliest_recovery_stage"].isin(CONDITIONS[1:]).sum()
        )
        group_rows.append(
            f"| {group} | {len(part)} | {reproduced} | {localized} | "
            f"{pd.to_numeric(part['accuracy__p1_natural'], errors='coerce').mean():.3f} | "
            f"{pd.to_numeric(part['accuracy__p5_fact_card'], errors='coerce').mean():.3f} |"
        )
    counts = (
        frame.groupby(["confusion_group", "mechanism_candidate"])
        .size()
        .reset_index(name="n")
    )
    mechanism_lines = [
        f"- `{row.confusion_group}` / `{row.mechanism_candidate}`: {int(row.n)}"
        for row in counts.itertuples(index=False)
    ]
    tp = frame[frame["confusion_group"] == "detector_true_positive"]
    tp_localized = int(tp["earliest_recovery_stage"].isin(CONDITIONS[1:]).sum())
    end_to_end_recall = tp_localized / float(score["future_positive"])
    lines = [
        "# CLARK Detector-Linked P1-P5 진단 결과",
        "",
        "## 질문",
        "",
        "T0에서 동결한 Core4 detector가 미래 DB 업데이트에서 new degradation을 탐지한 뒤,",
        "동일 문항에 evidence intervention을 적용해 실패 위치 후보까지 연결할 수 있는가?",
        "",
        "## 1. 전체 미래 341개 screening 성능",
        "",
        f"- AUROC: {score['auroc']:.3f}",
        f"- AUPRC: {score['auprc']:.3f}",
        f"- Precision / Recall / F1: {score['precision']:.3f} / {score['recall']:.3f} / {score['f1']:.3f}",
        f"- TP / FP / FN / TN: {int(score['tp'])} / {int(score['fp'])} / {int(score['fn'])} / {int(score['tn'])}",
        f"- Risk lift: {score['risk_lift']:.3f}x",
        "",
        "## 2. Probe cohort",
        "",
        "- 실제 new degradation 60개 전체(TP 48, FN 12)",
        "- detector false positive 42개 전체",
        "- transition, 질문 유형, 답변 형식과 risk가 가까운 true negative 42개",
        "- 합계 144개; screening 성능은 이 선택 표본이 아니라 전체 341개에서 계산",
        "",
        "| Detector 결과 | N | P1에서 하락 재현 | P2-P5에서 위치 후보 배정 | P1 accuracy | P5 accuracy |",
        "|---|---:|---:|---:|---:|---:|",
        *group_rows,
        "",
        "## 3. 회복 및 진단 규칙",
        "",
        f"- P1 하락 재현: pre-update accuracy - P1 accuracy >= {args.degradation_delta:.3f}",
        f"- 회복: P1 대비 accuracy gain >= {args.recovery_gain:.3f}이고 pre-update accuracy와의 잔여 차이 <= {args.recovery_tolerance:.4f}",
        "- P2 회복 + support 주입: retrieval coverage failure 후보",
        "- P3 회복 + rank 1 이동: ranking/position sensitivity 후보",
        "- P4 회복: evidence extraction 또는 context complexity 후보",
        "- P5 회복: evidence utilization 또는 answer realization 후보",
        "- P5 미회복: model instruction, evaluator 또는 linkage를 포함한 persistent failure 후보",
        "",
        "## 4. Detector부터 진단까지의 end-to-end 측정",
        "",
        f"- 미래 실제 하락 60개 중 detector가 경보한 문항: {int(score['tp'])}",
        f"- 그중 probe에서 회복 단계가 확인된 문항: {tp_localized}",
        f"- 전체 실제 하락 대비 탐지+위치후보 연결 비율: {end_to_end_recall:.3f}",
        "",
        "## 5. 문항별 위치 후보 수",
        "",
        *mechanism_lines,
        "",
        "## 해석 한계",
        "",
        "- 회복 단계는 intervention 기반 원인 후보이며, 인과적 root cause 확정이 아니다.",
        "- P5 fact card는 gold를 포함하는 oracle upper bound이며 운영 조건이 아니다.",
        "- P1에서 과거 하락이 재현되지 않으면 stochastic non-replication으로 분리한다.",
        "- FN probe는 offline 분석에는 유용하지만 label-free 운영에서 detector가 자동 호출하지 못한 사례다.",
    ]
    (output_dir / "report_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / str(config["dataset"]["path"])
    frame = metadata_frame(dataset_path)
    metrics = pd.read_csv(run_dir / "metrics" / "question_level_metrics.csv")
    metrics = metrics[metrics["embedding_model_role"] == "primary"].copy()
    for metric in ("accuracy", "semantic_entropy", "semantic_volume", "current_answer_rate", "stale_answer_rate", "contradiction_rate"):
        frame = frame.merge(pivot_metric(metrics, metric), on="question_id", how="left")
    frame = add_recovery_labels(
        frame,
        args.degradation_delta,
        args.recovery_gain,
        args.recovery_tolerance,
    )
    screening = screening_summary(pd.read_csv(args.predictions), args.detector)
    output_dir = run_dir / "linked_probe_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_question_linked_probe.csv", index=False, encoding="utf-8-sig")
    screening.to_csv(output_dir / "screening_performance.csv", index=False, encoding="utf-8-sig")
    mechanism_counts = (
        frame.groupby(["confusion_group", "earliest_recovery_stage", "mechanism_candidate"])
        .size()
        .reset_index(name="n_events")
    )
    mechanism_counts.to_csv(output_dir / "mechanism_counts.csv", index=False, encoding="utf-8-sig")
    condition_rows: list[dict[str, Any]] = []
    for group, part in frame.groupby("confusion_group", sort=False):
        for condition in CONDITIONS:
            condition_rows.append(
                {
                    "confusion_group": group,
                    "condition": condition,
                    "n_events": len(part),
                    "mean_accuracy": pd.to_numeric(
                        part[f"accuracy__{condition}"], errors="coerce"
                    ).mean(),
                    "mean_semantic_entropy": pd.to_numeric(
                        part[f"semantic_entropy__{condition}"], errors="coerce"
                    ).mean(),
                    "mean_semantic_volume": pd.to_numeric(
                        part[f"semantic_volume__{condition}"], errors="coerce"
                    ).mean(),
                }
            )
    pd.DataFrame(condition_rows).to_csv(
        output_dir / "condition_summary.csv", index=False, encoding="utf-8-sig"
    )
    group_rows: list[dict[str, Any]] = []
    for group, part in frame.groupby("confusion_group", sort=False):
        group_rows.append(
            {
                "confusion_group": group,
                "n_events": len(part),
                "degradation_reproduced": int(part["degradation_reproduced"].sum()),
                "localized_p2_to_p5": int(
                    part["earliest_recovery_stage"].isin(CONDITIONS[1:]).sum()
                ),
                "mean_p1_accuracy": pd.to_numeric(
                    part["accuracy__p1_natural"], errors="coerce"
                ).mean(),
                "mean_p5_accuracy": pd.to_numeric(
                    part["accuracy__p5_fact_card"], errors="coerce"
                ).mean(),
            }
        )
    pd.DataFrame(group_rows).to_csv(
        output_dir / "probe_group_summary.csv", index=False, encoding="utf-8-sig"
    )
    make_figures(frame, output_dir)
    write_report(frame, screening, output_dir, args)
    print(output_dir / "report_ko.md")


if __name__ == "__main__":
    main()
