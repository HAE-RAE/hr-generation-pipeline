# Reasoning Model Evaluation Pipeline

이 프로젝트는 다양한 언어 모델의 **Reasoning(추론) 모드**를 정량적으로 비교하기 위한 실험 파이프라인입니다. 모든 구성은 하나의 `config.yaml` 파일로 제어되며, 워커들은 상태 기반으로 독립적으로 동작합니다.

## 구성 요소
- `orchestrator.py`: 데이터셋과 모델 조합으로 태스크를 생성하여 DB에 저장합니다.
- `generation_worker.py`: 모델 응답을 생성하고 DB 상태를 업데이트합니다.
- `evaluation_worker.py`: 생성된 응답을 판정 프롬프트와 루브릭을 이용해 평가합니다.
- `export.py`: 완료된 태스크를 CSV 또는 Parquet 파일로 추출합니다.
- `prompts/`: YAML 형식의 판정 프롬프트(`judge_template.yaml`)와 루브릭(`general_rubric.yaml`)이 위치합니다.

## 빠른 시작
1. `config.yaml` 내용을 환경에 맞게 수정합니다.
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

자세한 사용 방법은 [`docs/usage_ko.md`](docs/usage_ko.md) 문서를 참고하십시오.
