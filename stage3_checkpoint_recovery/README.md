# Stage 3 — 체크포인트 저장 실패·손상 후 복구

**AI-assisted · 작은 단일 writer CPU 실험 · 공개 소스 · NOT_DEPLOYED · USER_COMPETENCY_NOT_ASSESSED**

## 한 줄 설명

별도 checkpoint 저장 writer가 종료되거나 최신 저장본이 손상됐을 때, 마지막 정상 저장본을 자동으로 골라 실제 학습을 이어가고 연속 참조와 비교하는 작은 실험실입니다. 학습 프로세스 전체를 강제 종료하거나 정전을 재현한 시험은 아닙니다.

### 실제 성공한 것

- 모델 가중치만이 아니라 최적화 모멘텀, 난수, 데이터 순서와 진행 기록을 함께 복구합니다.
- 단계 번호로 세대를 관리하며 동일 번호 덮어쓰기를 거절합니다.
- 최신 저장본을 검증하고 손상·미완료 상태를 건너뛰어 이전 정상 세대를 자동 선택합니다.
- 정상 상태가 하나도 없으면 조용히 처음부터 다시 시작하지 않고 명시적으로 실패합니다.
- 저장 중 두 번, 게시 직후 한 번 실제 하위 프로세스를 강제 종료하고, 다른 저장본 한 개는 고의로 손상시켰습니다.
- 최종 128회 학습의 모델·optimizer·RNG·sampler·손실/진행 상태와 실제 예측이 중단 없는 참조와 정확히 같았습니다.

이것은 새로운 알고리즘이나 산업용 제품이 아닙니다. 145개 파라미터의 작은 모델이며 GPU·대형 모델·여러 노드·정전 내구성을 검증하지 않았습니다. 탐지 성능은 개선하지 않았습니다.

## 실행

Python 3.12, PyTorch 2.6.0+cpu 및 vendor/stage2/requirements.txt에 기록된 기존 의존 환경에서 검증했습니다. 환경·패키지는 ZIP에 넣지 않았습니다. 새 환경 설치 방법은 원래 Stage2 README를 참고하되 이번 작업에서 설치·다운로드는 수행하지 않았습니다.

프로젝트 폴더에서:

```bash
python -m unittest discover -s tests -v
python -m recovery_lab verify --run my_first_replay
python -m recovery_lab inspect --run my_first_replay
python -m recovery_lab cleanup-dry-run --run my_first_replay
```

run 이름은 영문/숫자로 시작하는 영문·숫자·밑줄·하이픈 최대48자입니다. 경로를 입력하지 않습니다. 결과는 이 프로젝트의 runs/<이름>에만 생깁니다. 같은 run이 있으면 덮어쓰지 않습니다. 기존 실행 근거 frozen_probe는 보존하고, 재현할 때마다 새 이름을 씁니다.

cleanup-dry-run은 후보 목록만 보여줍니다. 파일을 삭제·이동하지 않습니다. 중단된 임시 저장본과 손상본은 그대로 남아 조사할 수 있습니다.

## 증거 읽기

- docs/EXPERIMENT_PLAN.md: 실행 전에 고정한 시나리오.
- runs/final_probe/RESULT.json: 최종 소스의 실제 반복 장애, 자동 선택, 전체 상태 비교와 실행 명령/종료코드.
- docs/unit_final_stderr.txt: 최종 builder 단위검사 결과. 독립 검사는 별도 문서로 구분합니다.
- docs/RESULTS_KO.md: 결과 해설과 제한.
- recovery_lab/store.py: 제한된 저장/검증/선택 관리자.
- recovery_lab/experiment.py: 고정 비교 실험. 임의 타이밍 대신 파일 훅 도달을 확인하고 자기 하위 프로세스만 종료합니다.
- docs/INDEPENDENT_AUDIT.md: 실제 완료된 독립 검토와 재현 결과, 검사 중복 및 한계.

## 기존 작업과의 관계

vendor/stage2는 이전 공개 프로젝트의 보존된 코드·데이터·결과 복제입니다. 기존 공개 Stage1/2 169파일은 바이트 그대로 보존하고 이 Stage3 디렉터리만 추가했습니다. 새 관리자는 Stage2가 만든 정상 checkpoint를 검증하고 세대별로 게시합니다. 강제 종료 대상 writer는 학습 결과를 저장소에 복사·게시하는 별도 프로세스이며, 학습 연산 중간·원래 저장 함수의 임의 바이트 쓰기 지점을 죽이는 시험은 아닙니다.

데이터는 공개 CWI TRAIN Original004/006 두 이미지 쌍으로, 독립 표본 두 개가 아니라 한 물리적 Chicken1에서 나왔습니다. 고정 8,192픽셀 중 양성은1개, 기존 모델은 전부 음성으로 예측했습니다. 이번 작업은 그 탐지 실패를 고치거나 숨기는 작업이 아닙니다.

## 안전 경계

단일 사용자·단일 writer가 관리하는 로컬 폴더를 전제로 합니다. 경로/심볼릭 링크/하드링크·파일 크기·형식·상태·해시 오류를 검사하지만 악의적인 동시 파일 교체(TOCTOU), 동일 저장소 동시 writer, 모든 해시와 내용의 동시 위조를 막는 보안 경계는 아닙니다. 네트워크 업로드/모델 pickle 실행 기능은 없습니다. 자세한 제한은 docs/RESULTS_KO.md를 읽어 주세요.

코드/문서는 MIT, 공개 CWI 데이터 및 파생 결과의 출처·CC BY4.0 조건은 ATTRIBUTION.md와 vendor/stage2의 원래 문서를 보존했습니다. AI의 도움으로 구현·검증했으며 사용자 개인의 독립 숙련도를 평가한 인증은 아닙니다.

## 패킷 구성 안내

runs/frozen_probe와 runs/final_probe에는 고의로 손상시킨 g000064와 미완료 .pending-* 폴더가 **실험 근거로** 포함됩니다. 정상 checkpoint가 아닌 것을 관리자가 건너뛰는지 확인하는 자료이며, 임의로 모델 파일로 불러오지 않습니다. 감사/단위검사의 악성 symlink·hardlink·초대형 fixture는 ZIP에서 제외했습니다. 독립 검사 결과는 docs에 선별한 script/JSON/감사문으로 포함했습니다.

## 공개 복사본 검증

원본 ZIP의 선별된 262파일을 바탕으로 공개 상태·설명 문서만 정리했습니다. 코드·테스트·vendor·수치 모델/배열·이미지·실행 근거는 원본과 동일합니다. 변경 내역과 개인정보/경로 검사는 [PUBLIC_RELEASE_PREPARATION.json](PUBLIC_RELEASE_PREPARATION.json), 이 공개 디렉터리의 자체 제외 해시는 [PACKAGE_SHA256SUMS.json](PACKAGE_SHA256SUMS.json)에 기록했습니다. 소스 공개는 실행 중인 웹 데모나 Hugging Face 배포를 의미하지 않습니다.
