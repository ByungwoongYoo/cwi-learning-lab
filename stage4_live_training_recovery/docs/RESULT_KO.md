# Stage4 실제 검증 결과

최종 builder 고정시나리오: 12.580초, PASS. 12개 unittest PASS(내부 parameterized 사례 포함, 독립검사와 중복 합산 금지).

- 실제 학습 자식 5개를 생성하고 각 프로세스를 정해진 코드지점에서 SIGKILL했다. 초기1회와 재시작4회이다.
- 시작 상태: 0 → 32 → 32 → 32 → 64. 종료 후 선택된 정상 저장:32 → 32 → 32 → 64 → 128.
- 최초128회 연속 실행과 최종 checkpoint의 전체 metadata, 모델4배열, momentum4배열, torch RNG, sampler permutation 모두 정확히 같다.
- 실제 평가 예측도 동일하다. terminal128에서 supervisor 재호출 시 학습 프로세스를 0개 생성했다.
- 장애 위치를 기록한 이벤트의 PID가 학습 step 이벤트 및 supervisor의 생성한 직접 자식 PID와 같다. 별도 완료상태 복사 writer만 종료한 이전 Stage3와 다르다.

참조/각 attempt/결과는 runs/builder_final에 포함되어 있다. 실제 결과가 없는 경우 성공으로 표시하지 않는다. 검사 원자료와 재현 명령은 README 및 독립 감사문을 따른다.

## 해석 제한
정해진 barrier에서 pause한 실제 training process를 종료한 시험이다. optimizer 내부 연산 임의시점 종료, supervisor자체 crash복구, 전원장애·디스크고장·동시writer·GPU·분산은 시험하지 않았다. 단일필렛 TRAIN 표본 양성1개이고 최종 전부음성 예측이므로 탐지 성능 향상으로 해석하면 안 된다.
