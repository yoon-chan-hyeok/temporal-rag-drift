# CLARK 과거 실패 문항 진단 Probe 결과

## 판정 규칙

- 회복: P1 대비 accuracy가 `0.25` 이상 증가하고, 해당 조건 accuracy가 `0.50` 이상
- P1~P5 조건명은 모델 입력에 노출하지 않고 결과 메타데이터에서만 사용
- 모든 조건의 질문, 평가 시점, 시스템 프롬프트, 샘플링 설정은 동일

## 핵심 요약

- 전체 문항: 84
- 과거 실패군의 P1 재실행 실패 재현율: 0.825

| 과거 상태 | N | P1 accuracy | P5 accuracy | P1 실패 수 | P1 실패 후 회복 |
|---|---:|---:|---:|---:|---:|
| new_degradation | 22 | 0.219 | 1.000 | 18 | 18/18 |
| persistent_failure | 18 | 0.160 | 0.944 | 15 | 15/15 |
| adaptive_control | 22 | 0.977 | 1.000 | 0 | 0/0 |
| normal_control | 22 | 0.972 | 1.000 | 0 | 0/0 |

## 최초 회복 단계 기반 진단

- `no_failure_on_rerun`: 51
- `evidence_utilization_or_answer_realization_failure`: 18
- `evidence_extraction_or_context_complexity_failure`: 12
- `retrieval_coverage_failure`: 2
- `ranking_or_position_sensitivity`: 1

- P2에서 회복: 자연 top-k에 빠진 최신 근거를 넣으면 회복하는 retrieval coverage 문제
- P3에서 회복: 같은 근거를 맨 앞으로 옮겼을 때 회복하는 ranking/position 문제
- P4에서 회복: 핵심 evidence만 남겼을 때 회복하는 extraction 또는 context complexity 문제
- P5에서 회복: 정답 사실을 명시해야 회복하는 evidence utilization 또는 answer realization 문제
- P5에서도 미회복: 모델 지시 준수, 평가기, 데이터 연결을 추가 점검해야 하는 지속 실패

## 산출물

- `per_question_probe_results.csv`: 문항별 P1~P5 정확도, uncertainty, shift, 최초 회복 단계
- `condition_summary.csv`: 코호트·조건별 평균과 bootstrap CI
- `recovery_threshold_sensitivity.csv`: 회복 임계값 민감도
- `mechanism_counts.csv`: 코호트별 진단 결과 수
- `accuracy_by_probe_stage.png`, `earliest_recovery_stage.png`: 논문용 요약 그림
