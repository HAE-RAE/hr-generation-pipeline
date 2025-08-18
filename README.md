# Reasoning Model Evaluation Pipeline

이 프로젝트는 **question/gold/category** 스키마를 따르는 데이터셋을 받아 언어 모델의 Base/Reasoning 응답을 생성하고 평가하는 실험 파이프라인입니다. 모든 동작은 하나의 `config.yaml`로 제어되며 워커들은 DB 상태를 통해 독립적으로 동작합니다.

## 구성 요소
- `orchestrator.py`: 데이터셋을 로드하고 카테고리별 프롬프트를 생성하여 태스크(DB)에 등록합니다.
- `generation_worker.py`: 각 태스크에 대해 `base_response`와 `reasoning_response`를 생성합니다.
- `evaluation_worker.py`: 선택된 평가 모듈을 실행하여 점수와 판정 결과를 기록합니다.
  - MCQA 정답 비교
  - `math_verify` 기반 수학 검증
  - LLM-as-Judge (개별/상대 비교) 평가
- `export.py`: 완료된 태스크를 CSV 또는 Parquet 형식으로 추출합니다.
- `prompts/`: 판정 프롬프트와 루브릭 YAML(`comparative_judge_template.yaml`, `general_rubric.yaml`)이 위치합니다.

## 빠른 시작
1. `config.yaml`의 `dataset`과 `evaluation_worker.evaluations_to_run`을 실험에 맞게 수정합니다.
2. 태스크 등록
   ```bash
   python orchestrator.py --config config.yaml --setup-only
   ```
3. 응답 생성
   ```bash
   python generation_worker.py --config config.yaml
   ```
4. 응답 평가
   ```bash
   python evaluation_worker.py --config config.yaml
   ```
5. 결과 내보내기
   ```bash
   python export.py --config config.yaml --format csv
   ```

수학 검증을 사용하려면 추가로 `pip install math-verify`를 실행하세요.

자세한 사용 방법은 [`docs/usage_ko.md`](docs/usage_ko.md) 문서를 참고하십시오.
