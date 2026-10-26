# Stage4 독립 감사 — PASS

실제 최종 소스를 읽고 별도의 새 run에서 실행했습니다. 새 학습 프로세스 5개가 각각 실제 optimizer 호출을 수행한 뒤 고정 barrier에서 대기했고, 부모가 직접 생성한 해당 자식만 SIGKILL하여 종료코드 -9를 확인했습니다. events의 PID/token/start state를 supervisor 기록과 별도로 대조했습니다.

- 독립 경계 사례45개 통과: 입력/marker identity, 모든 저장본 무효, 실제 restart cap/no-progress cap, symlink 및 경로 거부.
- 요청한 수정: checkpoint 저장 hook이32배수가 아니면 실행 전에 거부. 수정 후9개 불가능한 일정이 자식 생성 없이 거부됨을 확인했습니다.
- 최종 builder 단위검사12개 재실행 통과. 위 사례와 중복될 수 있어 합산하지 않습니다.
- 최종 독립 전체 반복장애 실험12.095초 PASS. 재개 지점[0,32,32,32,64], 종료 후 정상 저장본[32,32,32,64,128]. 모델·momentum·RNG·sampler·손실/진행 metadata·실제 예측 모두 참조와 정확히 일치했습니다.
- terminal128을 다시 검사할 때 자식 생성0회. 원본 Stage1/2/3/public release의 감사 시작 시점 파일 해시 불변.

## 제한
살아 있는 supervisor가 학습 자식을 재시작하는 시험입니다. supervisor 자체 재시작은 지원하지 않습니다. 정해진 코드 지점에서 대기한 뒤 SIGKILL했으며 임의 시점/optimizer 내부 연산 중 장애를 표본화하지 않았습니다. 정전/fsync 내구성·동시 writer·GPU·분산·운영 보안 인증이 아닙니다. 기존 Stage3 파서 공격검사 전체를 다시 수행한 것도 아닙니다. tiny TRAIN 표본의 탐지 성능은 여전히 실패했으며 새로운 알고리즘·논문 성과·사용자 독립 역량을 주장하지 않습니다.

소스 SHA256 및 정확한 실행 기록은 INDEPENDENT_AUDIT.json에 있습니다. 남은 필수 코드 결함은 발견되지 않았습니다.
