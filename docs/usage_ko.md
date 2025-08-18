# Reasoning Model Evaluation Pipeline 사용 가이드

이 문서는 파이프라인을 설치하고 실행하는 기본 절차를 설명합니다.

## 1. 환경 준비
- Python 3.10 이상을 권장합니다.
- `math_verify` 기반 수학 평가가 필요하다면 `pip install math-verify`를 추가로 실행합니다.

## 2. 설정 파일 편집
`config.yaml` 파일 하나로 실험을 제어합니다. 주요 항목은 다음과 같습니다.

- `database`: SQLite 또는 PostgreSQL 등의 연결 정보를 지정합니다.
- `dataset`: Hugging Face 데이터셋 이름과 split을 지정하거나, 예시처럼 `records` 항목에 직접 데이터를 적을 수 있습니다. 모든 레코드는 `question`, `gold`, `category` 컬럼을 포함해야 합니다.
- `models`: 평가할 모델 목록을 나열합니다.
- `generation_worker`: 응답 생성과 관련된 설정과 추론 on/off 제어 파라미터를 포함합니다.
- `evaluation_worker`: 실행할 평가 모듈을 `evaluations_to_run` 리스트로 지정합니다.
  - `mcqa`: 객관식 정답 비교
  - `math_verify`: 수학 답안 검증
  - `llm_judge_comparative`: Base vs Reasoning 상대 비교
  - `llm_judge_individual`: 개별 응답 평가
  - 각 Judge 설정은 `comparative_judge`와 `individual_judge` 블록에서 모델 이름, 프롬프트 템플릿, 루브릭 경로 등을 지정합니다.

## 3. 태스크 생성
```bash
python orchestrator.py --config config.yaml --setup-only
```
데이터셋과 모델 조합으로 구성된 모든 태스크가 DB에 `PENDING_GENERATION` 상태로 등록됩니다.

## 4. 응답 생성
```bash
python generation_worker.py --config config.yaml
```
생성 워커는 `PENDING_GENERATION` 태스크를 처리하여 응답을 저장하고 상태를 `GENERATION_COMPLETE`로 변경합니다.

## 5. 평가 수행
```bash
python evaluation_worker.py --config config.yaml
```
평가 워커는 `evaluations_to_run`에 명시된 모듈을 적용하여 정답 여부, 판정 결과 등을 기록하고 상태를 `COMPLETE`로 업데이트합니다.

## 6. 결과 내보내기
```bash
python export.py --config config.yaml --format csv
```
`status`가 `COMPLETE`인 태스크를 CSV 또는 Parquet 형식으로 추출합니다.

## 7. 모니터링
추가적으로 `monitor.py`와 같은 도구를 구현하여 태스크 상태 분포를 확인할 수 있습니다.

---
이 가이드는 기본 동작을 설명한 것으로, 실제 연구 환경에 맞추어 워커 수나 평가 로직 등을 확장하여 사용할 수 있습니다.
