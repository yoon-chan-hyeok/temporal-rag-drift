# CLARK Core4 기반 New Degradation Detector 연구노트

작성 기준일: 2026-08-17
실험 상태: Core4-only T0 모델 선정 및 T1-T4 동결 전이 평가 완료

## 1. 연구 질문

뉴스가 누적되며 RAG DB가 업데이트될 때, 동일 질문에 대한 답변 정확도가 새롭게 하락하는 `new degradation` 문항을 정답 label 없이 관측 가능한 답변분포 변화만으로 우선 탐지할 수 있는가?

이 단계에서 detector가 사용하는 정보는 다음 네 개뿐이다.

- 분포 이동: Energy distance, cluster JS divergence
- 불확실성 변화: delta semantic entropy, delta semantic volume

정답 label은 detector 입력으로 사용하지 않고, detector의 사후 성능 평가와 T0 calibration에만 사용한다.

## 2. 이번 실험에서 고정한 범위

- 데이터셋: CLARK 누적 뉴스 기반 temporal QA 실험 데이터
- 분석 대상: changed question만 사용
- Stable question은 이번 분류기의 학습, 정규화 및 평가에 사용하지 않음
- 비교 조건: `stale_only`와 `current_only`
- Mixed 조건은 사용하지 않음
- 모델 절대 상태를 나타내는 과거/현재 Entropy와 Volume은 입력에서 제외
- Core4 변화 지표만 사용
- T0에서 정규화, 모델, hyperparameter와 threshold를 선택한 뒤 T1-T4에서는 변경하지 않음

## 3. 시간 분할과 표본

분석 단위는 `질문 x DB 업데이트 이벤트`이다.

| 구분 | DB 변화 구간 | 관측 수 | New degradation |
|---|---|---:|---:|
| T0 calibration | 2021-12-22 -> 2022-08-31 | 167 | 24 |
| T1 frozen test | 2022-08-31 -> 2023-01-29 | 123 | 22 |
| T2 frozen test | 2023-01-29 -> 2023-07-31 | 92 | 16 |
| T3 frozen test | 2023-07-31 -> 2023-11-21 | 70 | 11 |
| T4 frozen test | 2023-11-21 -> 2024-04-19 | 56 | 11 |
| Future 전체 | T1-T4 | 341 | 60 |

- Future에는 329개의 고유 질문이 존재한다. 일부 질문은 서로 다른 업데이트 이벤트에 반복 관측된다.
- T0 질문과 future 질문은 분리되어 있다.
- Future bootstrap은 동일 질문의 반복 관측을 하나의 cluster로 취급한다.

## 4. 답변 생성 및 의미분포 구성

기존에 생성된 CLARK Luna 응답을 재사용했다. 이번 Core4 ML 분석에서는 API를 새로 호출하지 않았다.

| 항목 | 설정 |
|---|---|
| 생성 모델 | `gpt-5.6-luna` |
| backend | OpenAI-compatible API |
| 조건 | `stale_only`, `current_only` |
| 조건별 sampling | 16회 |
| 이벤트당 총 답변 | 32개 |
| temperature | 0.8 |
| top_p | 0.95 |
| max_new_tokens | 96 |
| generation rule | 제공된 retrieval context와 명시된 시점만 사용 |
| insufficient context | 고정된 insufficient 문구 출력 |

답변 embedding 및 semantic clustering 설정은 다음과 같다.

| 항목 | 설정 |
|---|---|
| Embedding | `BAAI/bge-large-en-v1.5` |
| Embedding normalization | 사용 |
| Volume용 PCA | 10차원 |
| Semantic clustering | NLI |
| NLI model | `microsoft/deberta-large-mnli` |
| Equivalence | bidirectional entailment |
| Numeric mismatch | 별도 cluster로 분리 |

## 5. 성능 저하 label 정의

정답률은 각 시점의 gold answer와 생성 답변을 NLI 및 lexical rule로 비교해 산출했다.

- `accuracy_x`: 과거 DB에서 과거 정답 기준 정확도
- `accuracy_y`: 업데이트 DB에서 현재 정답 기준 정확도
- Accuracy drop: `accuracy_x - accuracy_y`

운영 상태는 다음 규칙으로 분류했다.

- Persistent failure: `accuracy_x < 0.5`이고 `accuracy_y < 0.5`
- New degradation: persistent failure가 아니면서 `accuracy_x - accuracy_y >= 0.10`
- 그 밖의 recovery, adaptive success 및 정상 상태: 이번 binary endpoint에서 0

최종 binary target은 다음과 같다.

- `1`: new degradation
- `0`: 그 밖의 모든 operational outcome

따라서 이번 실험은 "현재 정확도가 낮은가"가 아니라 "DB 업데이트 후 새롭게 정확도가 0.10 이상 하락했는가"를 탐지한다.

## 6. Core4 특징

### 6.1 Shift

- `Energy`: stale 답변 embedding 분포와 current 답변 embedding 분포 사이 Energy distance
- `Cluster JS`: stale/current 의미 cluster 확률분포 사이 Jensen-Shannon divergence

### 6.2 Uncertainty change

- `Delta Entropy = SemanticEntropy_current - SemanticEntropy_stale`
- `Delta Volume = SemanticVolume_current - SemanticVolume_stale`

Core4를 택한 이유는 detector가 절대적으로 어려운 질문을 찾는 것이 아니라, DB 업데이트 전후에 발생한 분포 이동과 불확실성 변화를 탐지한다는 가설과 직접 대응하기 때문이다.

## 7. 정규화

모든 정규화 기준은 T0에서 계산한 뒤 future에 그대로 적용했다. Future 분포로 재정규화하지 않았다.

### 7.1 T0-ECDF

`ECDF(x) = (T0 rank(x) - 0.5) / N_T0`

- 출력 범위는 대략 0과 1 사이
- 단위와 왜도에 강함
- 값의 실제 간격보다 순위 정보를 사용

### 7.2 Rank-Gaussian

`RankGaussian(x) = Phi^-1(ECDF(x))`

- percentile을 표준정규 분위수로 변환
- 순위는 보존하지만 원래 값의 간격은 보존하지 않음

### 7.3 Robust-z

`RobustZ(x) = (x - median_T0) / (IQR_T0 / 1.349)`

- 평균과 표준편차 대신 중앙값과 IQR 사용
- 이상치에 강하면서 변화량의 상대적 크기를 유지
- 결과는 `[-8, 8]`로 clipping

Additive GAM에는 Core4 robust-z가 선택됐다. GAM spline이 각 변화량의 크기에 따른 비선형 위험 변화를 학습할 수 있다는 점에서 연구 가설과 가장 잘 맞는다.

## 8. 비교한 ML detector

- L2 Logistic
- Elastic Net
- Quadratic Logistic
- Additive GAM
- RBF-SVM
- Extra Trees
- HistGradientBoosting
- XGBoost
- MLP

모델 9종과 정규화 3종을 조합한 총 27개 Core4 후보를 비교했다.

- 모든 후보는 동일한 repeated stratified outer/inner fold를 사용
- tuned model은 repeated nested OOF 3회와 inner average precision으로 hyperparameter 선택
- class imbalance를 고려한 class-balanced loss 또는 sample weight 사용
- XGBoost는 `scale_pos_weight` binary log loss 사용
- 최종 alarm threshold는 T0 OOF F1 최대점으로 선택
- Future label은 모델, 정규화, hyperparameter 또는 threshold 선택에 사용하지 않음

단, XGBoost와 MLP는 이번 스크립트에서 사전 고정된 baseline configuration을 사용했으며 다른 모델과 동일한 크기의 hyperparameter grid를 탐색하지 않았다.

## 9. Core4-only 모델 결과

| 모델 | T0 선택 정규화 | T0 F1 | Future AUROC | Future AUPRC | Precision | Recall | Future F1 | Risk lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Quadratic Logistic | robust-z | 0.692 | 0.860 | 0.558 | 0.538 | 0.817 | **0.649** | 3.060x |
| L2 Logistic | robust-z | 0.691 | **0.883** | 0.617 | 0.526 | 0.833 | 0.645 | 2.991x |
| Additive GAM | robust-z | 0.679 | 0.853 | 0.531 | 0.533 | 0.800 | 0.640 | 3.031x |
| Elastic Net | robust-z | 0.679 | 0.886 | 0.642 | 0.510 | 0.833 | 0.633 | 2.900x |
| Extra Trees | robust-z | **0.706** | 0.867 | 0.534 | 0.528 | 0.783 | 0.631 | 3.001x |
| RBF-SVM | ECDF | 0.654 | 0.865 | **0.664** | 0.506 | 0.733 | 0.599 | 2.874x |
| HistGradientBoosting | rank-Gaussian | 0.667 | 0.861 | 0.506 | 0.467 | 0.817 | 0.594 | 2.652x |
| XGBoost | ECDF | 0.694 | 0.868 | 0.519 | 0.533 | 0.667 | 0.593 | 3.031x |
| MLP | robust-z | 0.640 | 0.860 | 0.563 | 0.445 | **0.883** | 0.592 | 2.531x |

T0 F1 기준 승자는 Extra Trees다. Future F1 기준 사후 최고는 Quadratic Logistic이다. Future F1은 T0 model selection에 사용하지 않았으므로 Quadratic Logistic을 이 실험의 사전 선정 승자로 바꿀 수는 없다.

## 10. Additive GAM 결과

선택 representation: `core4_robust_z`
T0 OOF threshold: `0.5879487749`

### 10.1 T0

- AUROC: 0.851
- AUPRC: 0.486
- Precision: 0.621
- Recall: 0.750
- F1: 0.679

### 10.2 Future 전체

- TP: 48
- FP: 42
- FN: 12
- TN: 239
- AUROC: 0.853
- AUPRC: 0.531
- Precision: 0.533
- Recall: 0.800
- F1: 0.640
- Risk lift: 3.031x

Question-cluster bootstrap 95% interval:

- AUROC: [0.799, 0.900]
- AUPRC: [0.408, 0.651]
- Recall: [0.694, 0.897]
- F1: [0.536, 0.723]
- Risk lift: [2.572, 3.664]

### 10.3 시기별 동결 전이

| 구간 | AUROC | F1 | Risk lift |
|---|---:|---:|---:|
| T1 | 0.847 | 0.655 | 2.95x |
| T2 | 0.861 | 0.683 | 3.22x |
| T3 | 0.851 | 0.615 | 3.39x |
| T4 | 0.821 | 0.560 | 2.55x |

T1-T3에서는 비교적 안정적이지만 T4에서 AUROC와 F1이 함께 하락했다. 따라서 GAM detector가 시간에 완전히 불변한다고 주장할 수는 없다.

## 11. 상위 모델 간 차이

동일 질문 cluster bootstrap 5,000회를 사용했다.

| 비교 | F1 차이 | 95% interval | 해석 |
|---|---:|---:|---|
| GAM - Quadratic Logistic | -0.0093 | [-0.0341, 0.0090] | 유의한 차이 확인 불가 |
| GAM - L2 Logistic | -0.0057 | [-0.0409, 0.0256] | 유의한 차이 확인 불가 |
| GAM - Extra Trees | 0.0091 | [-0.0133, 0.0349] | 유의한 차이 확인 불가 |

현재 표본에서는 상위 모델의 F1 차이가 작고 모든 interval이 0을 포함한다. 따라서 "GAM이 성능상 최고"라는 결론은 지지되지 않는다. 더 타당한 해석은 Core4 signal이 여러 분류기에서 유사하게 작동하며, GAM은 그중 해석 가능성이 높은 선택이라는 것이다.

## 12. Surface 해석

모델은 실제로 네 개 Core4 변수를 사용하지만 그림은 다음 두 축으로 투영한다.

전이 surface의 상위 3개 모델은 future 전체 F1을 본 뒤 설명 목적으로 고른 것이며, detector의 사전 model selection 결과로 해석하지 않는다. 각 그림의 모델과 threshold 및 위험면은 T0에서 적합한 상태로 고정하고 T1-T4의 점만 교체했다.

- Shift axis: `mean(T0-ECDF(Energy), T0-ECDF(JS))`
- Uncertainty-change axis: `mean(T0-ECDF(Delta Entropy), T0-ECDF(Delta Volume))`

배경색은 4D 모델 risk의 2D surrogate projection이고, 검은 선은 T0에서 동결한 alarm boundary다.

- 빨간 점: new degradation
- 청록 점: no new degradation
- 흰 영역: T0 관측점에서 멀어 외삽을 금지한 영역

상위 모델 surface projection OOF R2:

- Additive GAM: 0.962
- L2 Logistic: 0.960
- Quadratic Logistic: 0.927

따라서 이 세 모델은 2D shift x uncertainty 그림이 실제 4D 위험함수를 비교적 충실하게 표현한다. Surface 자체가 detector 입력은 아니며 설명용 근사다.

## 13. 연구적으로 지지되는 결론

1. Energy, JS, delta Entropy, delta Volume만으로 future new degradation을 평균 이상의 수준으로 순위화할 수 있다.
2. 여러 Core4 모델이 future AUROC 약 0.85 이상을 보였다.
3. 주요 모델의 경보 문항은 전체 평균보다 약 3배 높은 new degradation 비율을 보였다.
4. 절대 Entropy/Volume 상태를 추가하지 않아도 detector signal이 유지됐다.
5. 복잡한 비선형 모델이 단순한 L2 또는 GAM을 확실하게 이기지 못했다.
6. Core4 Additive GAM은 최고 성능 모델은 아니지만 해석 가능성과 future 성능을 함께 고려한 주모델 후보로 합리적이다.

## 14. 아직 지지되지 않는 주장

1. Core4 Additive GAM이 모든 모델보다 통계적으로 우수하다는 주장
2. CLARK에서 선택한 GAM이 다른 DB, retriever, 질문 유형과 LLM에서도 그대로 최적이라는 주장
3. Detector risk가 실제 오류 확률로 완전히 calibration됐다는 주장
4. Gold label 없이 개별 답변이 틀렸다고 확정할 수 있다는 주장
5. T4 이후에도 detector 성능이 유지된다는 주장
6. Shift와 uncertainty만으로 retrieval failure와 generation utilization failure를 완전히 구분할 수 있다는 주장

## 15. 방법론적 한계

### 15.1 작은 T0 양성 수

T0 new degradation은 24개뿐이다. 27개 후보를 비교했기 때문에 T0 winner에는 model-selection optimism이 존재할 수 있다.

### 15.2 GAM 선택은 현재 시점에서 post-hoc

GAM은 T0 F1 승자가 아니다. 이번 결과를 본 뒤 해석 가능성을 이유로 GAM을 주모델로 정한 것이므로, 현재 결과만으로 "GAM을 사전 선택했다"고 서술하면 안 된다. GAM을 지금 동결하고 이후의 새로운 dataset/model/time update에서 검증해야 prospective evidence가 된다.

### 15.3 OOF 정규화

현재 정규화 reference는 label을 사용하지 않지만 T0 전체에서 먼저 계산된 뒤 OOF 평가에 사용됐다. Future에는 누수가 없지만, 논문 최종판의 T0 OOF 추정에서는 각 training fold 내부에서 normalization reference를 fit해야 한다.

### 15.4 Label 정의 민감도

New degradation은 accuracy drop 0.10, persistent threshold 0.50에 의존한다. 0.05, 0.15, 0.20 등의 threshold sensitivity가 필요하다.

### 15.5 단일 운용 regime

현재 핵심 수치는 CLARK, Luna, 현재 retrieval/prompt 구성에 대한 결과다. 다른 LLM과 retriever에서 재현되어야 일반화 주장이 가능하다.

### 15.6 평가 label의 필요성

운영 시 detector 입력은 label-free지만, detector를 calibration하고 성능을 검증하는 단계에는 gold-derived accuracy가 필요하다. 따라서 본 방법은 "정답 없이 실패를 확정"하는 방법이 아니라, probe에서 학습한 risk signature를 이용해 운영 중 검토 우선순위를 제공하는 방법이다.

## 16. 현재 연구 수준 평가

### 실험적 수준

- 시간 순서를 지킨 T0 calibration과 T1-T4 locked transfer가 존재한다.
- Future label을 모델 선택에 사용하지 않았다는 점은 강점이다.
- Question-cluster bootstrap과 전이별 결과를 제공한다.
- Core4-only 결과는 연구 가설과 입력 특징의 의미가 일치한다.

현재 결과는 강한 proof-of-concept이며 workshop 또는 본 논문의 핵심 실험 한 축으로 사용할 수 있다. 하지만 이 결과 하나만으로 top conference 수준의 일반화 주장을 하기에는 부족하다.

### 가장 강한 수치

- Core4 미래 AUROC: 최고 0.886, 주요 모델 약 0.85 이상
- Core4 미래 F1: 최고 0.649
- Additive GAM 미래 F1: 0.640
- Additive GAM Recall: 0.800
- Additive GAM Risk lift: 3.031x
- Additive GAM surface projection R2: 0.962

### 핵심 해석

가장 중요한 결과는 특정 모델이 1등이라는 사실보다, 네 개의 변화 지표만으로 여러 모델에서 반복적으로 약 3배의 risk enrichment가 나타났다는 점이다. 이는 `shift x uncertainty`가 new degradation risk를 나타내는 운용 signature가 될 수 있다는 가설을 지지한다.

## 17. 다음 동결 결정과 필수 후속 실험

현재 시점 이후의 confirmatory experiment에서는 다음을 사전 고정하는 것이 적절하다.

- Primary detector: Core4 robust-z Additive GAM
- Features: Energy, cluster JS, delta Entropy, delta Volume
- Alarm threshold: 0.5879487749
- Secondary baseline: Core4 robust-z L2 Logistic
- Primary surface axis: mean robust-z(Energy, JS) x mean robust-z(delta Entropy, delta Volume)
- Primary surface definition: 4D detector에서 T0의 Energy-JS 및 delta Entropy-delta Volume 중앙 대비를 고정한 직접 단면

필수 후속 실험:

1. 새로운 LLM 또는 새로운 CLARK 시간 구간에 GAM을 재선정 없이 적용
2. 별도 실제 temporal RAG dataset에 동결 전이
3. Fold-wise normalization으로 T0 OOF 재평가
4. Accuracy-drop 및 persistent threshold sensitivity
5. Shift-only, uncertainty-only, high-high rule과 동일 locked test 비교
6. Brier score, calibration curve와 ECE를 이용한 절대 risk calibration
7. P2-P4 evidence intervention을 운영 로그만으로 자동 구성하는 방법

## 18. 산출물

### 분석 코드

- `scripts/clark_score_features.py`
- `scripts/clark_detector_models.py`
- `scripts/prepare_clark_detector_linked_probe.py`
- `scripts/analyze_clark_detector_linked_probe.py`

### 결과표

- `results/core4_ml/t0_selected_per_model.csv`
- `results/core4_ml/frozen_future_model_summary.csv`
- `results/core4_ml/frozen_transition_summary.csv`
- `results/core4_ml/paired_model_bootstrap.csv`
- `results/core4_ml/robust_z_direct_surfaces.csv`

### 그림

- Additive GAM: `assets/clark_core4_gam_robust_z_transfer.png`
- L2 Logistic: `assets/clark_core4_l2_robust_z_transfer.png`
- Quadratic Logistic: `assets/clark_core4_quadratic_robust_z_transfer.png`

### Surface 해석 정정

기존 `model_surfaces`와 `top3_transfer_surfaces` 그림은 4D detector의 출력을
`mean T0-ECDF(Energy, JS)`와 `mean T0-ECDF(delta Entropy, delta Volume)`에 다시
적합한 KernelRidge surrogate projection이다. 모델 입력 자체가 ECDF였다는 뜻이
아니다. 따라서 이 그림은 설명용 투영으로만 사용해야 한다.

`robust_z_direct_surfaces`는 별도 surrogate를 사용하지 않는다. 동결된 4D 모델에
robust-z 좌표를 직접 넣되, 한 장에 표시하기 위해 두 shift 변수와 두 uncertainty
변수의 T0 중앙 대비를 고정한 2D 단면이다. 점의 검은 경보 테두리는 단면값이 아니라
각 문항의 실제 4D 입력에 대한 예측이다.

과거 PPT의 `changed-development CDF` 그림은 별도의 2D quadratic logistic detector다.
당시 입력은 T0 changed empirical CDF로 정규화한 shift score와 uncertainty score였고,
shift score는 SWD, MMD, Energy, cluster JS, centroid gap의 평균이었다. 따라서 0~1
정사각형과 0.5 사분면이 깔끔하게 보였지만, 이는 현재 Core4 robust-z GAM의 결정면과
동일한 그림이 아니다.

## 19. 최종 한 문장

CLARK 누적 뉴스 RAG에서 T0 probe로 학습한 Core4 shift x uncertainty detector는 미래 DB 업데이트에서 새롭게 정확도가 하락하는 질문을 약 3배 높은 위험도로 우선순위화했으며, Additive GAM은 최고 성능 모델은 아니지만 해석 가능성과 전이 성능을 함께 갖춘 동결 detector 후보로 확인됐다.
