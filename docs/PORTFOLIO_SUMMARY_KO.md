# 포트폴리오 요약

## 프로젝트

**CLARK 누적 뉴스 DB 업데이트에 따른 RAG 성능 저하 위험 탐지**

뉴스 DB가 과거 시점 `Kx`에서 누적 업데이트된 `Ky`로 바뀔 때, 동일한
RAG 시스템의 답변 분포 변화를 이용해 새롭게 정답률이 하락하는 질문을
우선 탐지하는 연구형 프로젝트입니다.

## 수행 내용

- CLARK 질문, 시점별 정답, 외부 뉴스 근거의 로컬 연결 구조 점검
- SQLite FTS5 BM25와 BGE dense retrieval을 결합한 공통 hybrid retriever 구성
- 시점별 누적 뉴스 snapshot과 top-k 10 문맥 생성
- 질문과 시점마다 답변 16개를 생성해 답변 분포 구성
- SWD, MMD, Energy, Cluster-JS, centroid gap으로 shift 측정
- semantic entropy와 semantic volume 변화로 uncertainty 측정
- T0에서 detector와 threshold를 동결하고 미래 4개 DB 업데이트에 전이
- P1-P5 evidence intervention으로 failure 위치 후보 진단
- retrieval support, 시점 cutoff, 중복 기사, 실행 비용과 로그 오류 감사

## 핵심 결과

T0 detector를 미래 질문 비중복 confirmatory 186문항에 적용했습니다.

- new degradation: 28문항
- AUROC: 0.854
- recall: 0.714
- F1: 0.615
- risk lift: 3.59배

전체 미래 primary 320건에서는 AUROC 0.870, F1 0.662, risk lift 3.16배를
기록했습니다.

## 원인 probe

84문항을 natural top-k부터 compact fact card까지 다섯 단계로 재실행했습니다.
new degradation 중 P1에서 실패한 18문항은 P5에서 모두 회복했습니다.
이를 통해 retrieval coverage, ranking, context complexity, evidence utilization
문제를 구분하는 진단 후보를 만들었습니다.

## 기술 구성

Python, PyTorch, Transformers, SentenceTransformers, scikit-learn, SQLite FTS5,
BM25, BGE, reciprocal-rank fusion, DeBERTa NLI, OpenAI-compatible API,
quadratic logistic regression을 사용했습니다.

## 주장 범위

이 프로젝트는 한 답변의 절대 정답 여부를 판정하지 않습니다. DB 업데이트
전후 답변 분포를 비교해 **새로운 성능 저하 위험을 순위화하고 전문가 검토
대상을 줄이는 것**이 목표입니다.

## 역할 한 문장

시간에 따라 누적되는 외부 지식 DB에서 RAG failure risk를 탐지하기 위해
temporal retrieval, 반복 답변 분포, 동결 detector, 미래 전이 평가와
evidence probe를 설계하고 실행·감사한 연구 프로젝트입니다.
