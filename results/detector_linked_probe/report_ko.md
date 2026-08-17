# CLARK Detector-Linked P1-P5 진단 결과

## 질문

T0에서 동결한 Core4 detector가 미래 DB 업데이트에서 new degradation을 탐지한 뒤,
동일 문항에 evidence intervention을 적용해 실패 위치 후보까지 연결할 수 있는가?

## 1. 전체 미래 341개 screening 성능

- AUROC: 0.853
- AUPRC: 0.531
- Precision / Recall / F1: 0.533 / 0.800 / 0.640
- TP / FP / FN / TN: 48 / 42 / 12 / 239
- Risk lift: 3.031x

## 2. Probe cohort

- 실제 new degradation 60개 전체(TP 48, FN 12)
- detector false positive 42개 전체
- transition, 질문 유형, 답변 형식과 risk가 가까운 true negative 42개
- 합계 144개; screening 성능은 이 선택 표본이 아니라 전체 341개에서 계산

| Detector 결과 | N | P1에서 하락 재현 | P2-P5에서 위치 후보 배정 | P1 accuracy | P5 accuracy |
|---|---:|---:|---:|---:|---:|
| detector_true_positive | 48 | 43 | 43 | 0.254 | 0.999 |
| detector_false_negative | 12 | 12 | 12 | 0.073 | 1.000 |
| detector_false_positive | 42 | 1 | 1 | 0.676 | 0.981 |
| detector_true_negative_control | 42 | 3 | 3 | 0.885 | 0.961 |

## 3. 회복 및 진단 규칙

- P1 하락 재현: pre-update accuracy - P1 accuracy >= 0.100
- 회복: P1 대비 accuracy gain >= 0.100이고 pre-update accuracy와의 잔여 차이 <= 0.0625
- P2 회복 + support 주입: retrieval coverage failure 후보
- P3 회복 + rank 1 이동: ranking/position sensitivity 후보
- P4 회복: evidence extraction 또는 context complexity 후보
- P5 회복: evidence utilization 또는 answer realization 후보
- P5 미회복: model instruction, evaluator 또는 linkage를 포함한 persistent failure 후보

## 4. Detector부터 진단까지의 end-to-end 측정

- 미래 실제 하락 60개 중 detector가 경보한 문항: 48
- 그중 probe에서 회복 단계가 확인된 문항: 43
- 전체 실제 하락 대비 탐지+위치후보 연결 비율: 0.717
- 단, 43개에는 gold fact를 직접 주는 P5 회복이 포함되므로 0.717은 oracle 진단 상한이다.
- P5를 제외하고 P2-P4에서 회복한 실제 하락은 11/60(0.183)이다.
- 실제 하락 60개 중 52개는 natural top-k에 최신 support가 이미 존재했다.

## 5. 문항별 위치 후보 수

- `detector_false_negative` / `evidence_extraction_or_context_complexity_failure`: 4
- `detector_false_negative` / `evidence_utilization_or_answer_realization_failure`: 6
- `detector_false_negative` / `retrieval_coverage_failure`: 2
- `detector_false_positive` / `evidence_utilization_or_answer_realization_failure`: 1
- `detector_false_positive` / `no_degradation_on_probe_rerun`: 41
- `detector_true_negative_control` / `evidence_extraction_or_context_complexity_failure`: 2
- `detector_true_negative_control` / `evidence_utilization_or_answer_realization_failure`: 1
- `detector_true_negative_control` / `no_degradation_on_probe_rerun`: 39
- `detector_true_positive` / `evidence_extraction_or_context_complexity_failure`: 8
- `detector_true_positive` / `evidence_utilization_or_answer_realization_failure`: 31
- `detector_true_positive` / `no_degradation_on_probe_rerun`: 5
- `detector_true_positive` / `ranking_or_position_sensitivity`: 1
- `detector_true_positive` / `retrieval_coverage_failure`: 2
- `detector_true_positive` / `stochastic_recovery_without_context_change`: 1

## 해석 한계

- 회복 단계는 intervention 기반 원인 후보이며, 인과적 root cause 확정이 아니다.
- P5 fact card는 gold를 포함하는 oracle upper bound이며 운영 조건이 아니다.
- P1에서 과거 하락이 재현되지 않으면 stochastic non-replication으로 분리한다.
- FN probe는 offline 분석에는 유용하지만 label-free 운영에서 detector가 자동 호출하지 못한 사례다.
