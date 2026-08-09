# Learning & Engineering Roadmap

이 문서는 연구 아이디어를 면접에서 코드와 운영 관점까지 설명할 수 있는 포트폴리오로 강화하기 위한 계획입니다.

## 1. 데이터·검색 기반기

- SQL: 실행 이력, 문서 버전, 질문, 검색 결과를 정규화한 스키마로 설계
- PostgreSQL/pgvector: 벡터 검색과 메타데이터 필터를 한 저장소에서 재현
- 데이터 버전 관리: 지식 스냅샷과 평가 세트의 해시를 모든 실행에 기록

**완료 증거:** ERD, 마이그레이션 파일, 대표 SQL 10개, 인덱스 전후 실행 계획 비교

## 2. 평가 신뢰성

- temporal/question-disjoint split으로 누수 방지
- bootstrap confidence interval과 McNemar 검정 구현
- calibration curve, AUROC, F1, Precision@k, Risk Lift 비교
- `변화 감지`와 `오류 탐지`를 분리한 단위 테스트

**완료 증거:** 고정 seed 실험, 자동 생성되는 평가 리포트, 실패 케이스 분석표

## 3. 운영화

- 배치 평가 워크플로와 재시도 가능한 task 설계
- Docker Compose로 API·DB·worker 실행
- OpenTelemetry trace와 대시보드 연결
- 임계값 초과 시 검토 큐 또는 rollback 후보를 만드는 정책

**완료 증거:** 한 명령 재현, CI 통과 배지, 장애 시나리오별 runbook

