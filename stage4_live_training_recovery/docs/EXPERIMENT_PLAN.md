# 사전 고정 계획

PLAN 상수는 실행 전 코드에 고정되며, 각 verify가 참조/장애 실행 전에 PLAN.json으로 저장합니다. 바꿔가며 성공 사례를 선택하지 않습니다.

- 기준: 같은 config/data/runtime의 Stage2 uninterrupted128.
- 변경: 학습 루프 내32step 저장, child PID/token 검사, bounded supervisor 자동 재시작.
- 장애: backward37 / optimizer48 / before_commit64 / array_closed96 / published128.
- 예상 resume: 0,32,32,32,64. 각 종료 후 선택:32,32,32,64,128.
- 성공: metadata 및 모든 배열, 예측 np.array_equal. terminal128 후 launch0.
- 중단: 계획 부정합, reference failure, state mismatch, identity mismatch, 모든 checkpoint 무효, restart/no-progress/time cap, 외부 환경 제약.
- 외부 호출·비용·GPU·추가 데이터·test/validation 이미지 사용 없음.

손상 파일 거부는 전체 후보가 무효일 때의 명시 실패를 시험합니다. 손상 후보와 정상 후보를 함께 둔 fallback은 기존 Stage3에서 시험한 별도 범위입니다.
