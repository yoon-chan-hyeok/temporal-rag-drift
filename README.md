![Temporal RAG Drift project hero](assets/project-hero.svg)

<div align="center">

**DB 업데이트 이후 새로 위험해진 RAG 질문을 라벨 없이 우선순위화하고, evidence intervention으로 조사할 실패 구간을 좁히는 평가 프레임워크**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Protocol](https://img.shields.io/badge/Protocol-Frozen%20Temporal%20Transfer-7C3AED)
![Tests](https://img.shields.io/badge/Tests-16%20passing-15803D)
![Status](https://img.shields.io/badge/Status-Research%20Artifact-D97706)

[핵심 결과](#핵심-결과) · [진단 프로브](#탐지-후-진단) · [빠른 검증](#빠른-검증) · [재현 문서](docs/REPRODUCIBILITY.md)

</div>

---

## 문제

RAG의 지식베이스가 업데이트되면 새 정보를 답할 수 있지만, 기존 질문의 검색 결과와 문맥도 함께 바뀝니다. 운영 시점에는 현재 질문의 정답 라벨이 바로 없기 때문에 업데이트가 실제 성능 저하를 만들었는지 즉시 확인하기 어렵습니다.

이 프로젝트는 업데이트 전후의 답변 분포와 불확실성 변화를 이용해 검토할 질문의 우선순위를 정합니다. 위험 사례에는 P1-P5 evidence intervention을 적용해 retrieval coverage, ranking, context complexity와 evidence utilization 중 어디를 먼저 조사할지 제안합니다.

## 방법

```mermaid
flowchart LR
    A["Cumulative knowledge<br/>snapshots Kx and Ky"] --> B["Fixed hybrid<br/>retrieval"]
    B --> C["16 answer samples<br/>per condition"]
    C --> D["Shift and uncertainty<br/>features"]
    D --> E["T0-frozen<br/>detector"]
    E --> F["Future risk<br/>ranking"]
    F --> G["P1-P5 evidence<br/>probe"]
```

- retrieval: SQLite FTS5 BM25와 BGE dense retrieval을 reciprocal-rank fusion으로 결합
- monitoring signal: SWD, RBF-MMD, Energy distance, semantic-cluster JS, centroid gap, semantic entropy와 volume 변화
- evaluation: T0에서 detector와 threshold를 고정한 뒤 질문이 겹치지 않는 미래 업데이트에 적용
- diagnosis: evidence 제공 조건만 단계적으로 바꿔 최초 복구 지점을 기록

세부 정의와 실험 설정은 [Methods](docs/METHODS.md)에 정리했습니다.

## 핵심 결과

주요 주장은 T0 이후 처음 등장한 질문 186건으로 구성한 confirmatory cohort 결과입니다. Detector와 threshold는 미래 구간에서 다시 맞추지 않았습니다.

| Cohort | N | New degradation | AUROC | Recall | F1 | Risk lift |
|---|---:|---:|---:|---:|---:|---:|
| **Future question-disjoint confirmatory** | 186 | 28 | **0.854** | **0.714** | **0.615** | **3.59×** |

이는 완전한 오류 판정기가 아니라 검토 우선순위를 만드는 결과입니다. 업데이트 구간별 편차가 있었고, 가장 약한 구간의 AUROC는 `0.658`이었습니다. 전체 표와 bootstrap interval은 [Results](docs/RESULTS.md)에서 확인할 수 있습니다.

![Frozen transfer surfaces](assets/clark_t0_temporal_transfer_surface.png)

## 탐지 후 진단

탐지된 과거 실패를 같은 질문과 sampling 조건에서 다시 실행하고 evidence 전달만 바꿨습니다.

```text
P1  natural top-k
P2  guarantee current support is present
P3  move current support to rank 1
P4  decisive evidence only
P5  compact fact card
```

New-degradation 22건 중 P1에서 다시 실패한 18건은 P5까지 모두 복구됐습니다. 최초 복구 단계는 확정 원인이 아니라 다음 조사 대상을 정하는 진단 가설입니다.

![Probe accuracy](assets/clark_probe_accuracy_by_stage.png)

## 구현과 재현

- temporal data linkage와 cumulative snapshot 생성
- hybrid retrieval과 고정 sampling
- 분포·불확실성 feature와 frozen detector
- question-disjoint evaluation, aggregate report와 P1-P5 probe
- synthetic smoke fixture와 16개 unit test

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\mini_temporal_mock.yaml
```

Smoke run은 배선과 실행 절차만 확인합니다. CLARK 실험을 다시 실행하려면 라이선스가 허용된 원천 데이터와 외부 모델·API가 필요합니다. 전체 순서는 [Reproducibility](docs/REPRODUCIBILITY.md)와 [Data pipeline](docs/CLARK_DATA_PIPELINE.md)에 있습니다.

## 해석 범위

- risk score는 개별 답변의 정답 확률이 아닙니다.
- CLARK 밖의 데이터셋, 모델, prompt와 retriever로 일반화된다고 주장하지 않습니다.
- probe의 최초 복구 단계는 인과적으로 증명된 root cause가 아닙니다.
- CLARK 원천 파일, 기사 본문, API 응답, 모델 가중치와 local index는 공개하지 않습니다.

## 기여

연구 질문, temporal protocol, 평가 기준, detector, failure probe, 결과 감사와 해석 범위를 설계했습니다. 공개 저장소에는 실행 코드, 테스트와 핵심 집계표를 함께 두었습니다.

[Methods](docs/METHODS.md) · [Results](docs/RESULTS.md) · [Limitations](docs/LIMITATIONS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)
