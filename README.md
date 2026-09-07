<div align="center">

[한국어](README.md) | [English](README_EN.md)

# Temporal RAG Failure Detection & Diagnosis

**DB 업데이트 뒤 답변이 나빠졌을 가능성이 큰 질문을 먼저 고르고, 어떤 RAG 단계를 점검할지 좁히는 연구입니다.**

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Dataset CLARK](https://img.shields.io/badge/Dataset-CLARK--News-0F766E)
![Protocol Frozen Transfer](https://img.shields.io/badge/Protocol-Frozen%20Temporal%20Transfer-7C3AED)
![Tests](https://github.com/yoon-chan-hyeok/temporal-rag-drift/actions/workflows/tests.yml/badge.svg)

[Problem](#1-problem-and-operating-setting) · [Method](#4-method) · [Results](#6-results-frozen-temporal-transfer) · [Diagnosis](#7-diagnosis-intervention-based-probing) · [Quick start](#9-quick-start) · [발표자료](TEMPORAL%20RAG%20FAILURE%20DETECTION.pptx)

</div>

뉴스·정책 문서를 갱신할 때마다 모든 질문의 최신 정답을 다시 만들기는 어렵습니다. 같은 질문에 대한 업데이트 전후의 답변 분포를 비교해, 정답 검수가 준비되기 전에 확인할 질문의 순서를 정했습니다. 탐지한 사례에는 검색 근거를 바꿔 넣는 intervention-based probing을 적용해 회복되는 단계를 살폈습니다.

첫 업데이트에서 선택하고 고정한 **Core4 + Extra Trees는 미래 평가 341건 중 89건에 경보를 냈고, 그 안에 실제 성능 저하 60건 중 47건이 포함됐습니다.** 모든 질문을 같은 우선순위로 보는 대신 위험 사례를 모아 검토할 수 있는지 확인한 결과입니다. 근거 개입 실험은 별도의 Additive GAM을 사용했으며, 최신 정답 근거를 활용한 탐색적 진단입니다.

## 1. Problem and operating setting

문서를 최신화해도 RAG가 새 사실을 제대로 쓰지 못할 수 있습니다. 과거 문서가 남아 있는 DB에서는 최신 evidence와 outdated evidence가 함께 검색되기 때문입니다. 같은 query의 답변이 여러 후보로 흩어질 수도 있고, 예전 답으로 계속 수렴할 수도 있습니다. 단순히 답이 바뀌었는지 또는 confidence가 낮은지만 확인해서는 두 경우를 충분히 구분하기 어렵다고 봤습니다.

![Gold answer가 없는 RAG update monitoring 문제](assets/ppt_problem_setting.png)

이 프로젝트는 다음과 같은 배치형 운영 환경을 가정했습니다.

| 운영 조건 | 이 프로젝트의 가정 |
|---|---|
| DB 업데이트 | `Ky`는 새 DB로 교체한 상태가 아니라 `Kx`에 새 문서가 누적된 스냅샷입니다. |
| RAG 구성 | DB 업데이트의 영향만 보기 위해 검색기, 프롬프트와 생성 모델을 고정합니다. |
| 점검 질문 | 업데이트 전후에 같은 질문을 다시 실행할 수 있는 모니터링 질문 또는 회귀 테스트 묶음이 있습니다. |
| 최신 정답 | 운영 시점에는 모든 질문의 최신 gold answer가 없습니다. 과거 label은 초기 detector selection과 offline evaluation에만 사용합니다. |
| 관측 단위 | 질문별로 각 스냅샷에서 답변을 16회 생성해 한 번의 응답이 아니라 답변 분포를 비교합니다. |

따라서 이 detector는 처음 들어온 질문 하나의 정오를 실시간으로 판정하지 않습니다. DB 업데이트 전후를 모두 관측할 수 있는 질문을 대상으로, 어떤 질문부터 사람이 확인해야 하는지 순서를 정합니다. 여기서 `label-free`는 detector가 무학습이라는 뜻이 아니라, future update를 감시할 때 새 gold answer를 입력으로 쓰지 않는다는 뜻입니다.

## 2. 적용을 가정한 운영 흐름

| 적용 장면 | 이 프로젝트가 제공하는 정보 |
|---|---|
| 뉴스, 정책, 규정 DB의 정기 업데이트 | 업데이트 뒤 위험도가 커진 질문을 정렬해 전체 질문 대신 상위 사례부터 검수할 수 있습니다. |
| 새 DB snapshot의 배포 전 점검 | alarm이 몰린 질문과 변화 양상을 보고 배포 전에 추가 확인할 범위를 정할 수 있습니다. |
| 장애 또는 품질 저하 triage | alarm query에 evidence를 단계적으로 바꿔 넣어 retrieval coverage, ranking, context complexity, evidence utilization 중 어디를 먼저 살펴볼지 좁힐 수 있습니다. |

운영에 적용한다면 DB ingestion이 끝난 뒤 monitoring query set을 `Kx`와 `Ky`에 다시 실행합니다. Detector는 degradation risk 순으로 review queue를 만들고, alarm case만 intervention probe로 넘깁니다.

출력은 질문별 상대 위험도와 근거 개입에서 처음 회복한 단계입니다. 이를 검수 목록과 담당 단계별 점검에 연결하는 것이 예상 활용 방식입니다. 실제 서비스에서의 검수 시간 절감이나 자동 롤백·수정은 검증하지 않았습니다. 특히 현재 probe는 benchmark의 최신 정답 근거를 사용하므로, 운영 로그만으로 같은 진단을 수행하려면 개입 근거를 마련하는 과정이 추가로 필요합니다.

## 3. Design rationale

설계의 출발점은 **답변이 어디로 이동했는지와 얼마나 흩어졌는지를 함께 보자**는 것이었습니다. 새 정답으로 안정적으로 바뀐 경우와 자신 있게 오답을 반복하는 경우는 모두 큰 shift를 보일 수 있습니다. 반대로 평균적인 답은 비슷해도 불확실성이 커질 수 있습니다. 그래서 shift와 uncertainty change를 별개의 정보로 남겼습니다.

운영 조건은 시간 분할에도 반영했습니다. 첫 업데이트 `T0`에서 detector와 threshold를 정한 뒤, 이후 업데이트에는 새 정답을 보고 조정하지 않고 적용했습니다. 위험 점수만으로는 확인할 RAG 단계를 알 수 없으므로, 탐지 이후에 evidence intervention을 붙였습니다.

## 4. Method

CLARK의 time-valid QA를 누적 news snapshot으로 재구성하고, 같은 query를 `Kx`와 `Ky`에서 반복 실행합니다. 각 snapshot의 answer distribution에서 Core4 feature를 계산한 뒤 `T0`에서 detector와 threshold를 선택합니다. 이후 `T1`부터 `T4`까지는 refitting 없이 그대로 적용합니다.

![Temporal RAG failure detection and diagnosis pipeline](assets/ppt_framework.png)

### 4.1 Answer distribution sampling

생성 모델의 답은 같은 입력에서도 달라질 수 있습니다. 한 번의 응답만 비교하면 sampling noise를 실제 drift로 오해할 수 있어, snapshot마다 같은 query에 16개의 답변을 생성했습니다. 아래 사례처럼 DB 시점이 바뀐 뒤 정답률뿐 아니라 답변이 어느 후보로 모이는지도 달라집니다.

![같은 query의 DB snapshot별 answer distribution 변화](assets/ppt_answer_distribution_example.png)

### 4.2 Core4 features

Core4는 단순 confidence score가 아닙니다. Embedding geometry와 semantic cluster probability를 함께 사용해 answer distribution의 이동과 불확실성 변화를 측정합니다.

```text
Shift               = Energy distance + cluster JS divergence
Uncertainty change  = Δ semantic entropy + Δ semantic volume
```

![Core4: shift와 uncertainty change를 함께 보는 이유](assets/ppt_core4_features.png)

처음에는 shift와 uncertainty를 높고 낮음으로 나눈 단순 rule을 검토했습니다. 실제 데이터에는 크게 이동했지만 한쪽 오답으로 수렴해 uncertainty가 낮아지는 `confident-wrong` case도 있었습니다. 그래서 네 feature를 이분화하지 않고 그대로 사용해 여러 classifier family를 비교했습니다.

## 5. Experimental protocol

`T0`는 deployment 전 calibration 또는 초기 운영 구간에 해당합니다. T0에서 normalization, model family, hyperparameter와 alarm threshold를 선택한 뒤 모두 freeze했습니다. 미래 update인 `T1`부터 `T4`에는 새 label을 보지 않고 동일한 detector를 적용했습니다.

![T0에서 고정한 detector를 미래 update에 적용하는 protocol](assets/ppt_frozen_transfer_protocol.png)

| 항목 | 설정 |
|---|---|
| 데이터 | CLARK 자연어 시간 질의와 정답 유효 기간 |
| DB 스냅샷 | 해당 시점까지 발행된 뉴스 기사 누적 |
| 검색 | SQLite FTS5 BM25 + `BAAI/bge-large-en-v1.5` + RRF |
| 답변 생성 | `gpt-5.6-luna`, temperature 0.8, top-p 0.95, 조건별 16회 |
| 의미 분석 | BGE 답변 임베딩 + `microsoft/deberta-large-mnli` 군집화 |
| 성능 저하 정의 | 업데이트 뒤 정확도가 10%p 이상 하락한 사례. 두 시점 모두 정확도가 50% 미만인 persistent failure는 양성에서 제외 |

CLARK를 사용한 이유는 시간에 따라 정답이 달라지는 질문과 뉴스 근거가 연결돼 있어, 동일한 규칙으로 과거 `Kx`와 누적 업데이트 뒤 `Ky`를 구성할 수 있기 때문입니다. 실제 서비스 DB를 공개하거나 과거 상태로 되돌리지 않고도 temporal update를 통제해 볼 수 있는 대리 환경입니다. 따라서 이 결과를 다른 도메인의 운영 성능으로 바로 일반화하지 않습니다. 데이터 구성 과정은 [CLARK 데이터 파이프라인](docs/CLARK_DATA_PIPELINE.md)에 정리했습니다.

## 6. Results: frozen temporal transfer

Model family, representation, hyperparameter와 alarm threshold는 첫 업데이트 `T0`만 보고 정했습니다. 이후 네 update에는 retraining이나 threshold tuning 없이 적용했습니다.

| 구간 | DB 업데이트 | 사례 수 | 새 성능 저하 |
|---|---|---:|---:|
| T0 선택 구간 | 2021-12-22 → 2022-08-31 | 167 | 24 |
| T1 평가 | 2022-08-31 → 2023-01-29 | 123 | 22 |
| T2 평가 | 2023-01-29 → 2023-07-31 | 92 | 16 |
| T3 평가 | 2023-07-31 → 2023-11-21 | 70 | 11 |
| T4 평가 | 2023-11-21 → 2024-04-19 | 56 | 11 |

T1부터 T4까지는 질문과 업데이트의 조합 341건이며, 고유 질문은 329개입니다. 341건 전체에서 고정 전이 성능을 평가했고, 이 중 성능 저하는 60건이었습니다. 비교한 9개 분류기 계열 가운데 T0 selection winner, 미래 지표별 상위 모델과 diagnosis에 사용한 모델을 표에 제시했습니다.

| 모델 | T0에서 선택된 정규화 | 미래 AUROC | AUPRC | 재현율 | F1 | 위험도 향상 |
|---|---|---:|---:|---:|---:|---:|
| Extra Trees, T0 선택 | robust-z | 0.867 | 0.534 | 0.783 | 0.631 | 3.00배 |
| Elastic Net | robust-z | **0.886** | 0.642 | 0.833 | 0.633 | 2.90배 |
| L2 로지스틱 | robust-z | 0.883 | 0.617 | 0.833 | 0.645 | 2.99배 |
| 2차 로지스틱 | robust-z | 0.860 | 0.558 | 0.817 | **0.649** | **3.06배** |
| Additive GAM | robust-z | 0.853 | 0.531 | 0.800 | 0.640 | 3.03배 |
| RBF-SVM | ECDF | 0.865 | **0.664** | 0.733 | 0.599 | 2.87배 |

위험도 향상(risk lift)은 경보 사례에서 성능 저하가 나타난 비율을 전체 평가 집합의 성능 저하 비율로 나눈 값입니다. Extra Trees의 경우 전체 341건 중 60건(17.6%)이 성능 저하였고, 경보 89건 중에서는 47건(52.8%)이었습니다. 경보 밖으로 놓친 성능 저하 13건과 거짓 경보 42건도 남습니다.

T0 F1 기준의 공식 selection winner는 Extra Trees입니다. 미래 구간을 사후 비교하면 Elastic Net의 AUROC, RBF-SVM의 AUPRC와 2차 로지스틱의 F1이 각각 가장 높았습니다. 미래 label을 보고 배포 모델을 다시 고른 것은 아닙니다. GAM과 L2 로지스틱·2차 로지스틱·Extra Trees 사이의 F1 차이는 question-cluster bootstrap으로 검토했으며, 세 비교 모두 95% 구간에 0을 포함했습니다. 이어지는 diagnosis experiment에는 feature별 risk curve를 확인할 수 있고 frozen transfer 성능도 비슷한 Additive GAM을 post-hoc으로 선택했습니다. 전체 후보와 T0 선택 결과는 [frozen_future_model_summary.csv](results/core4_ml/frozen_future_model_summary.csv), 비교 구간은 [paired_model_bootstrap.csv](results/core4_ml/paired_model_bootstrap.csv)에서 확인할 수 있습니다.

![고정 Core4 Additive GAM 3차원 위험 표면](assets/clark_core4_gam_3d_direct_surface.png)

3차원 표면은 4차원 GAM을 2개의 해석 축으로 보인 직접 단면입니다. 이동량 축은
robust-z Energy와 JS를, 불확실성 변화 축은 robust-z 엔트로피 변화와 부피
변화를 같이 움직입니다. 색 표면은 T0의 변수 쌍 내 중앙 차이를 고정해 계산했고,
점의 높이는 각 질문의 실제 4차원 GAM 위험 확률입니다.

![Additive GAM의 구간별 고정 Core4 전이](assets/clark_core4_gam_robust_z_transfer.png)

구간별 그림의 빨간 점은 새 성능 저하, 청록색 점은 그 밖의 사례,
검은 테두리는 실제 4차원 경보를 뜻합니다. [L2 로지스틱](assets/clark_core4_l2_robust_z_transfer.png)과
[2차 로지스틱](assets/clark_core4_quadratic_robust_z_transfer.png) 단면도 함께 공개했습니다.

## 7. Diagnosis: intervention-based probing

Detector가 찾은 위험 질문만으로는 failure mechanism을 알 수 없습니다. 같은 query에 들어가는 evidence를 P1부터 P5까지 한 단계씩 바꾸고, 어느 intervention에서 answer가 처음 회복되는지 확인했습니다. 이 방식은 root cause를 확정하는 causal diagnosis가 아니라, 다음에 확인할 RAG stage를 좁히는 triage입니다.

![P1부터 P5까지 evidence intervention ladder](assets/ppt_intervention_probe.png)

Additive GAM의 탐지 결과를 기준으로 양성 60건 전체와 거짓 경보 42건, 크기를 맞춘 정상 대조군 42건을 다시 실행했습니다. 경보에 잡히지 않은 양성도 포함해 탐지 누락 이후의 회복 양상을 함께 확인했습니다. 총 144건에서 11,520개의 답변을 생성했습니다.

Screening confusion matrix와 probe 집계는 [screening_performance.csv](results/detector_linked_probe/screening_performance.csv)와 [진단 보고서](results/detector_linked_probe/report_ko.md)에 공개했습니다.

| 단계 | 근거를 바꾼 방법 | 이 단계에서 회복할 때 먼저 의심할 부분 |
|---|---|---|
| P1 | 원래 검색 결과 | 기준 상태 |
| P2 | 최신 정답 근거가 상위 10개 안에 들도록 보장 | 검색 범위 |
| P3 | 최신 정답 근거를 1순위로 이동 | 순위와 위치 |
| P4 | 결정적인 근거만 제공 | 정보 추출 또는 복잡한 문맥 |
| P5 | 현재 사실을 짧은 카드로 제공 | 근거 활용과 답변 생성 |

![탐지 결과 그룹별 근거 개입 정확도](assets/clark_detector_linked_probe_accuracy.png)

실제 성능 저하 60건 중 48건이 경보에 걸렸습니다. 이 가운데 5건은 P1에서 실패가 다시 나타나지 않았고, 나머지 43건은 경보에 걸린 뒤 어느 단계에서든 회복했습니다. P5에는 현재 정답 사실이 직접 들어가므로 `43/60`, 71.7%는 진단 가능성의 상한으로 해석했습니다. 운영 환경의 위치 추정률로 사용할 수는 없습니다. P5 이전의 P2부터 P4에서 회복한 사례는 11건이었습니다.

![탐지 결과별 후보 실패 원인](assets/clark_detector_linked_probe_mechanisms.png)

최신 정답 근거가 원래 검색 상위 10개에 있었던 양성 사례가 60건 중 52건이었지만, 재현된 성능 저하는 주로 P4 또는 P5에서 처음 회복했습니다. 이 결과는 이 조건에서 근거 추출과 활용을 먼저 살펴볼 필요가 있음을 보여줍니다. 개입 단계는 점검 후보를 좁히는 방법이며 하나의 원인을 인과적으로 증명하지는 않습니다.

## 8. Repository structure

```text
assets/                     탐지기 단면과 근거 개입 결과 그림
configs/actual/             민감정보를 뺀 실제 실험 설정
data/                       합성 예제와 CLARK 준비 안내
docs/                       방법, 데이터 파이프라인, 결과와 한계
results/clark_t0/           초기 2축 기준 실험
results/core4_ml/           Core4 모델 선택과 고정 전이 결과
results/detector_linked_probe/ 탐지 연계 근거 개입 집계
scripts/                    검색, 생성, 탐지와 근거 개입 실행 코드
src/                        공통 RAG, 임베딩, 군집화와 지표 모듈
tests/                      단위 테스트와 합성 파이프라인 테스트
```

CLARK 원문 질문과 기사, 사례별 예측, 답변 로그, 모델 가중치, API 인증 정보와 SQLite 인덱스는 공개하지 않았습니다.

## 9. Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\mini_temporal_mock.yaml
.\.venv\Scripts\python.exe scripts\check_public_artifact.py
```

이 실행은 직접 만든 합성 데이터와 모의 구성 요소로 파이프라인 연결을 확인합니다. 위 연구 결과를 재현하는 실행은 아닙니다. 전체 재현에는 라이선스를 지켜 준비한 CLARK 원자료와 로컬 DB 스냅샷이 필요합니다. 자세한 내용은 [재현 안내](docs/REPRODUCIBILITY.md)와 [CLARK 데이터 파이프라인](docs/CLARK_DATA_PIPELINE.md)에 정리했습니다. 실험을 읽는 순서는 [Methods](docs/METHODS.md), [결과 해석](docs/RESULTS.md), [집계 파일 안내](results/README.md)입니다.

## 10. Limitations

- 미래 탐지기는 최신 정답을 입력으로 사용하지 않습니다. 정답은 T0 학습·선택과 오프라인 평가, 그리고 별도 진단 실험의 개입 근거 구성에 사용했습니다.
- Core4 평가에는 정답이 시간에 따라 바뀌는 changed question만 포함했습니다. 정답이 유지되는 질문까지 섞인 운영 검수 목록에서의 성능은 추가 확인이 필요합니다.
- Core4 미래 341건과 초기 186건 실험은 서로 다른 평가 집합이므로 직접적인 성능 향상으로 비교하지 않습니다.
- 탐지기는 업데이트 전후의 상대 위험을 추정합니다. 처음 보는 답변 하나의 정오를 판정하지 않습니다.
- 결과는 이 저장소에서 사용한 CLARK, 검색기, 생성 모델과 프롬프트 조건에서 확인했습니다.
- 근거 개입에서의 회복은 점검 후보를 제시할 뿐, 하나의 원인을 확정하지 않습니다.

CLARK 출처: [Language Modeling with Editable External Knowledge](https://aclanthology.org/2025.findings-naacl.168/)
