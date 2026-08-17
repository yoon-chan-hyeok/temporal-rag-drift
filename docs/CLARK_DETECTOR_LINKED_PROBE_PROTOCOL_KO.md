# CLARK 동결 Detector 연계 Probe 실험

## 목적

이 실험은 두 단계를 연결한다.

1. T0에서 동결한 Core4 detector가 미래 DB 업데이트의 `new degradation`을 선별한다.
2. 선별 결과에 P1-P5 evidence intervention을 적용해 성능 저하의 후보 위치를 좁힌다.

Detector의 탐지 성능과 probe의 진단 성능은 분리해 평가한다. Detector 성능은 미래 구간 전체 341개 문항으로 계산하며, probe 결과는 진단 대상으로 선정한 144개 문항에서만 계산한다.

## 입력과 동결 조건

- 전체 미래 평가 사건: 341개
- 실제 new degradation: 60개
- Detector: T0에서 학습하고 동결한 Additive GAM
- Detector 입력: Core4 변화량
  - Energy distance
  - JS divergence
  - Semantic entropy 변화량
  - Semantic volume 변화량
- 미래 구간에서는 모델 재학습이나 임계값 재조정을 하지 않는다.

## Probe 코호트

동결 detector의 미래 예측을 기준으로 다음 144개를 선정한다.

| 그룹 | 의미 | 문항 수 |
|---|---|---:|
| TP | 실제 하락이며 detector가 경보 | 48 |
| FN | 실제 하락이나 detector가 놓침 | 12 |
| FP | 하락하지 않았지만 detector가 경보 | 42 |
| Matched TN | 하락하지 않았고 경보도 없음 | 42 |

TN은 시간 구간, 질문 관계, binary/open 형식, detector risk가 가능한 한 비슷하도록 FP와 매칭한다. 이 구성은 detector가 잡은 실패만 분석하는 선택 편향을 줄이고, 놓친 실패와 오경보의 성질도 비교하기 위한 것이다.

## P1-P5 intervention

프롬프트에는 현재 probe 단계명을 노출하지 않는다. 모든 단계는 동일한 질문과 동일한 시점을 사용하고 evidence 구성만 바꾼다.

| 단계 | 입력 evidence | 진단 의미 |
|---|---|---|
| P1 Natural | 실제 top-k retrieval | 운영 상태 재현 |
| P2 Support | 정답 지지 문서를 top-k에 추가 | retrieval coverage 후보 |
| P3 Support-first | 지지 문서를 1순위로 이동 | ranking/position 후보 |
| P4 Evidence-only | 관계에 직접 필요한 evidence만 제공 | chunking/context complexity 후보 |
| P5 Fact-card | 현재 사실을 짧고 명시적으로 제공 | evidence utilization의 진단 상한 |

P5에는 현재 정답이 포함되므로 운영 detector가 아니라 oracle 진단 조건이다.

## 회복과 후보 위치 판정

과거 DB 정확도를 `A_pre`, P1 정확도를 `A_P1`, 각 probe 정확도를 `A_Pk`라고 한다.

- P1에서 하락 재현: `A_pre - A_P1 >= 0.10`
- probe 회복: `A_Pk - A_P1 >= 0.10`이며 `A_Pk >= A_pre - 0.0625`

최초 회복 단계로 후보 위치를 정한다.

- P2에서 실제 support가 추가되고 회복: retrieval coverage 후보
- P3에서 실제 순위가 바뀌고 회복: ranking/position sensitivity 후보
- P4에서 회복: extraction/context complexity 후보
- P5에서만 회복: evidence utilization/answer realization 후보
- P5에서도 미회복: explicit fact 이후에도 지속되는 generation/utilization failure 후보
- P1에서 과거 하락이 재현되지 않음: stochastic generation 또는 비재현 사례

이 분류는 intervention 기반의 후보 위치 진단이다. 단일 실행만으로 인과적 root cause를 확정한다고 주장하지 않는다.

## 주요 평가값

Detector screening은 미래 전체 341개에서 다음을 보고한다.

- AUROC, AUPRC
- precision, recall, F1
- risk lift
- TP, FN, FP, TN

Probe 진단은 144개에서 다음을 보고한다.

- 그룹별 P1 하락 재현율
- 단계별 평균 정확도와 회복률
- 후보 failure mechanism 분포
- 실제 미래 하락 60개 중 `detector가 탐지하고 probe가 위치 후보까지 제시한 비율`

마지막 값은 end-to-end coverage이며 detector 성능과 별도로 해석한다.
특히 P5는 gold fact를 직접 제공하는 oracle 상한이므로 P2-P4 회복률과
분리해서 보고한다.

## 실행

비용과 코호트 확인:

```powershell
cd "<repository-root>"
.\run_clark_detector_linked_probe_luna.cmd --stage cost
```

전체 실험:

```powershell
cd "<repository-root>"
.\run_clark_detector_linked_probe_luna.cmd --stage all --confirm-api-cost --gpu 0
```

현재 산출된 비용 계획은 11,520회 요청, 예상 약 $6.43, 최대 출력 기준 약 $7.31이다.

## 출력

- 실행 루트: `outputs/runs/clark_detector_linked_probe_luna`
- 코호트 감사표: `data/processed/clark_detector_linked_probe/cohort_audit.csv`
- 문항별 결과: `linked_probe_analysis/per_question_linked_probe.csv`
- Detector 전체 성능: `linked_probe_analysis/screening_performance.csv`
- 후보 원인 집계: `linked_probe_analysis/mechanism_counts.csv`
- 한국어 보고서: `linked_probe_analysis/report_ko.md`
- 시각화: `linked_probe_analysis/*.png`

## 포트폴리오에서의 위치

기존 결과만으로도 `DB 업데이트 후 answer-distribution 신호로 위험 문항을 선별한다`는 detector 프로젝트는 제시할 수 있다. 이 추가 실험은 완성의 필수 조건은 아니지만, `경보 이후 무엇을 할 것인가`에 답한다. 따라서 포트폴리오의 이야기를 탐지에서 진단까지 확장하는 강한 보강 실험이다.
