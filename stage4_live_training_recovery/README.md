# Live CPU training restart lab — Stage 4

**PUBLIC_SOURCE_ADDITION · AI_ASSISTED · NOT_DEPLOYED · USER_COMPETENCY_NOT_ASSESSED**

실제 학습 프로세스가 중단됐을 때, 살아 있는 supervisor가 정상 체크포인트를 고르고 **새 프로세스에서 학습을 재시작하는 작은 재현 실험**입니다. AI-assisted engineering portfolio이며 새로운 알고리즘·논문 결과나 운영 시스템 인증이 아닙니다. Stage3의 별도 저장 writer 종료 시험에서 범위를 한 단계 확장했습니다.

## 실행

기존 환경: Python 3.12, PyTorch 2.6.0+cpu, NumPy 등은 `vendor/stage2/requirements.txt` 및 기존 문서를 참고하세요. 환경·패키지는 ZIP에 포함하지 않습니다. 설치나 네트워크 요청은 실행 코드에 없습니다. 동일 라이브러리 버전의 Linux CPU 환경을 사용해야 합니다.

프로젝트 루트에서:

```
python -m live_recovery verify --run my_run
python -m live_recovery inspect --run my_run
python -m unittest discover -s tests -v
```

run 이름은 단순 영숫자/밑줄/하이픈 1–48자이며 새 이름이어야 합니다. 결과는 `runs/my_run/RESULT.json`, 사전 계획은 `PLAN.json`, 각 자식의 실제 PID·시작 상태·학습 step·중단 위치는 `work/attempt*/events.jsonl`에 남습니다. 삭제 기능은 없습니다. 별도 임시/손상 저장본도 자동 삭제하지 않습니다.

## 고정 시험

145개 파라미터의 CPU MLP가 8,192개의 고정 TRAIN 픽셀로 128 optimizer updates를 수행합니다. 32회마다 저장합니다.

1. step37 backward 후, optimizer 전 종료 → 정상 저장32에서 재시작.
2. step48 optimizer 후, loss/progress 갱신 전 종료 → 정상 저장32에서 재시작.
3. step64 progress 갱신 후, checkpoint 생성 전 종료 → 정상 저장32에서 재시작.
4. step96 checkpoint 게시 중 배열 복사 후 종료 → 정상 저장64에서 재시작.
5. step128 checkpoint 게시 직후 종료 → 완성된128을 인식하고 추가 학습 없이 완료.

각 장애는 **실제 학습과 optimizer를 실행한 같은 자식 PID**가 정확한 코드 지점에서 대기한 다음, 부모가 해당 Popen 자식만 SIGKILL하는 방식입니다. 무작위 시점의 장애나 optimizer 내부 C++ 연산 한가운데 종료를 시험한 것은 아닙니다. 모든 신호 대상은 자신이 만든 단일 직접 자식입니다.

## 성공 판정

중단 없는 참조와 모델 가중치·SGD momentum·torch RNG·sampler RNG/순서·loss history·epoch/cursor/step가 정확히 일치해야 합니다. 실제 예측도 비교합니다. 최종 완료 후 supervisor를 다시 호출할 때 새 학습을 시작하지 않는지 확인합니다.

`tests/`에는 모든 후보 손상/미완성만 존재하는 경우의 fail-closed, 실제 재시작 횟수 제한, 실제 무진행 반복 제한, marker PID/token/attempt 검증, 계획·경로 거부가 포함됩니다. 기존 Stage3 파일 파서/안전 경계 검증 전체를 다시 했다는 뜻은 아닙니다.

## 한계와 정직한 사용

- 단일 writer, 동일 CPU 환경, 작은 고정 데이터만 검증합니다. GPU·분산·정전/fsync 내구성·동시 writer는 검증하지 않았습니다.
- 살아 있는 supervisor가 자식 학습 프로세스를 재시작합니다. **supervisor 자체 종료 후 비완료 run을 이어가는 기능은 없습니다.** 그런 run은 보존되며 같은 이름 재실행은 거부됩니다.
- 최초 실행을 포함한 자식 실행8회(재시작최대7회)·무진행4회·자식40초·감시루프540초 상한이 있습니다. 고정 failpoint는 검사 가능한 테스트 장치이며 외부 공격자에 대한 보안 인증이 아닙니다.
- 데이터는 Chicken1 한 필렛의 두 영상쌍입니다. 표본 양성은1픽셀이고 모델은 전부 음성으로 예측합니다. **탐지 성능 개선·일반화·식품 안전 검증이 아닙니다.**
- AI_ASSISTED; 사용자의 독립 구현/설명 능력은 평가하지 않았습니다.
- 기존 공개 저장소의 Stage1/2/3 및 root README를 포함한 432파일은 바이트 그대로 보존하고 이 Stage4 디렉터리만 추가했습니다.

출처·라이선스는 `ATTRIBUTION.md`, 구현 범위는 `docs/EXPERIMENT_PLAN.md`, 실제 결과는 `docs/RESULT_KO.md`를 참고하세요.

## 공개 복사본 검증

원본 ZIP의 225파일을 모두 유지하며 공개 상태와 비개인 설명만 정리했습니다. 코드·테스트·vendor·모델/수치배열·이미지·실행기록은 원본과 같습니다. [PUBLIC_RELEASE_PREPARATION.json](PUBLIC_RELEASE_PREPARATION.json)에 변경 내역과 경로/개인정보 검사를, [PACKAGE_SHA256SUMS.json](PACKAGE_SHA256SUMS.json)에 이 공개 디렉터리의 자체 제외 해시를 기록했습니다. 공개 소스 게시이며 실행 중인 웹 데모나 Hugging Face 배포가 아닙니다.
