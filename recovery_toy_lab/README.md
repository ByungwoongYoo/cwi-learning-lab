# Recovery Toy Lab 0.1.0

작은 CPU 학습을 고정된 다섯 지점에서 중단하고, 정상 저장본을 선택해 새 학습 프로세스로 자동 재시작하는 **재현용 도구**입니다. CWI 이미지 다운로드는 필요 없습니다.

## 무엇이 달라졌나

이전 프로젝트의 학습·복구 구조를 자체 패키지로 정리했습니다. 설치 경로에 기록하지 않고 사용자가 지정한 새 출력 폴더에 결과를 남깁니다. 7→16→1 PyTorch MLP, SGD+momentum, dropout, 128 updates의 **한 가지 합성 데이터 adapter**만 지원합니다. 범용 checkpoint 프레임워크나 새로운 복구 알고리즘이 아닙니다.

## 요구 환경

검증 환경: Linux, Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.3.5. 패키지는 Python ≥3.10, torch ≥2.6, NumPy ≥1.26을 선언하지만 다른 버전 조합은 실행 검증하지 않았습니다. Windows/macOS는 지원 검증하지 않았으며 CLI에서 거부합니다. GPU·네트워크·외부 데이터는 사용하지 않습니다. PyTorch와 NumPy는 별도로 설치되어 있어야 합니다. 의존성이 없는 도구가 아닙니다.

이미 의존성이 있는 가상환경에서 로컬 wheel을 설치합니다:

    python -m pip install --no-index --no-deps dist/recovery_toy_lab-0.1.0-py3-none-any.whl
    python -m recovery_tool doctor

설치 뒤 아래 **한 명령**으로 전체 데모를 재현합니다. 출력 상위 폴더는 이미 있어야 하고 마지막 폴더는 새 이름이어야 합니다.

    python -m recovery_tool demo --output /your/existing/folder/recovery_demo

콘솔 명령 `recovery-toy`도 같은 CLI입니다. source 사용은 프로젝트 루트에서 `python -m recovery_tool`을 실행합니다. 기존 결과를 덮어쓰지 않습니다. 읽기 검사는 다음과 같습니다.

    python -m recovery_tool inspect --output /your/existing/folder/recovery_demo

결과 폴더의 RESULT.json, REPORT.txt, PLAN.json, SUPERVISOR_PROGRESS.json 및 각 학습 프로세스 이벤트를 확인하세요. PASS는 정확히 해당 환경과 고정 계획을 통과했다는 뜻입니다. CLI 종료코드: 0 성공, 1 실행/입력 실패, 2 명령행 구문 오류, 3 의존성 누락/지원하지 않는 플랫폼.

## 고정 실험

- 합성 입력 8,192행·7특성. 양성 4,096, 음성 4,096. NumPy PCG64 seed 20261026; 선형 점수의 중앙 순위로 라벨을 나눕니다.
- 모델 seed 1701, sampler seed 1702. 모델·optimizer·torch RNG·sampler RNG·순열·진행상태·손실기록을 저장합니다.
- backward37, optimizer48, commit전64, 배열저장96, 최종게시128의 barrier에서 **직접 생성한 학습 자식 프로세스**만 SIGKILL합니다.
- 연속 참조 학습과 복구 학습의 전체 상태 및 예측이 정확히 같은지 검사합니다.
- 완료 저장본이 있으면 불필요한 추가 학습을 시작하지 않습니다.

## 한계

재시작 관리 프로세스(supervisor)는 살아 있어야 합니다. 정해진 barrier 시험이며 임의 시점 장애의 통계적 검증이 아닙니다. 정전 durability, 여러 writer, GPU, 분산 시스템, 원격 저장소는 보장하지 않습니다. 미완성 디렉터리는 삭제하지 않습니다. 합성 TRAIN 재현이므로 실세계 탐지 성능·일반화·사용자 숙련도·취업 성공을 보여주는 증거가 아닙니다. AI-assisted 구현과 검증입니다.

## 라이선스와 공개 상태

원래 Stage2–4의 자체 작성 MIT 코드를 재구성했습니다. MIT 고지와 출처는 LICENSE 및 ATTRIBUTION.md에 보존했습니다. 외부 데이터나 PyTorch/NumPy 바이너리는 배포하지 않습니다. 이 패키지는 로컬로 제작한 산출물이며 PyPI 또는 새 외부 서비스에 게시하지 않았습니다.
