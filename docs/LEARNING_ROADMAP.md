# 미구현 확장 계획

이 문서는 현재 완료된 기능이 아니라 이후 학습과 구현 후보를 기록합니다. README의 실험 결과에는 아래 항목이 포함되지 않습니다.

## 데이터와 검색

- 실행 이력, 문서 버전, 질문과 검색 결과를 관리할 SQL schema
- PostgreSQL과 pgvector 기반 검색 재현
- 지식 snapshot과 평가 세트의 version hash 관리

## 운영화

- 재시도 가능한 batch evaluation workflow
- API, DB와 worker를 분리한 Docker Compose 환경
- trace, dashboard, review queue와 rollback 후보 정책

완료 기준은 schema와 migration, 한 명령 재현, CI, dashboard와 장애 runbook입니다.
