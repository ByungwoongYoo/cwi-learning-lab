# Stage 2 — 실제 CPU 신경망 학습 + 중단/재개 검증

**완료한 것은 작은 PyTorch 학습 루프와 체크포인트 복구의 기술 검증입니다. 탐지 성능 개선이나 사용자 숙련도 검증이 아닙니다.**

- 공식 CPU wheel의 **PyTorch 2.6.0+cpu**를 사용했습니다(이 환경의 검증 버전이며 최신 버전이라는 뜻이 아닙니다).
- MLP `7 → 16 → 1`, tanh, dropout 0.1, **145개 파라미터**. CPU 1스레드, float64, SGD + momentum.
- 공개 CWI **TRAIN Original004/006 두 이미지 쌍만** 사용. 두 쌍 모두 Chicken1, 즉 **물리적 닭가슴살 표본 1개**에 속합니다. 고정된 8,192개 픽셀, 4 epoch / 128 update.
- 37 update에서 저장하고 프로세스를 종료한 뒤 **새 OS 프로세스에서 91 update**를 이어갔습니다.
- 128 update 연속 실행과 최종 파라미터·모멘텀·예측·손실 이력·RNG·샘플러 상태가 **완전히 일치**했습니다.
- JSON/NPZ 자체 형식. `torch.load`, pickle 사용 없음. 손상·해시·스키마 등 실패 사례는 검증 결과에 기록했습니다.

## 가장 중요한 한계

고정 무작위 표본의 양성 라벨은 **8,192개 중 1개**입니다. 최종 임계값 0.5의 예측은 **전부 음성**입니다: `TP=0, FN=1, FP=0, TN=8191`.

학습 손실이 약 `0.83306 → 0.005708`로 낮아졌지만, 이 결과는 심한 클래스 불균형의 **훈련 데이터 재대입(resubstitution)** 결과일 뿐입니다. 뼈 탐지에 성공했다는 뜻이 아니며 정확도 숫자로 포장하지 않습니다. 이를 보고 표본/시드/학습률을 바꾸는 탐색은 하지 않았습니다.

**불포함:** 전체 87개 학습, validation/purged/test/Dataset2 사용, 기존 모델 갱신, 대형 모델·GPU·다중노드 경험, 과학적 성능 향상, 사용자 독립 구현/면접 능력 입증.

## 실행 — Linux x86_64 / Python 3.12 검증

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu -r requirements-torch-cpu.txt
python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m training_lab.verify --out artifacts/my_reproduction
```

새 결과 폴더가 이미 있으면 덮어쓰지 않고 거절합니다. 기본 검증은 번들 TRAIN TIFF만 읽으며 네트워크, UI, GPU가 필요 없습니다. dependency 설치에만 인터넷이 필요합니다. `.venv`와 wheel은 ZIP에 넣지 않았습니다.

`verify` 명령이 합성 데이터 기울기 검사 → 표본 준비 → 연속 학습 → 중단 학습 → 새 프로세스 재개 → 정확한 상태 비교 → 잘못된 체크포인트 거절 검사를 실행합니다. 별도 세부 명령은 [재현 가이드](docs/REPRODUCING.md)에 있습니다.

## 먼저 볼 파일

| 파일 | 역할 |
|---|---|
| `artifacts/technical_smoke/verification.json` | 실제 학습·재개·음성 테스트 결과 |
| `artifacts/technical_smoke/process_commands.json` | 세 개 프로세스 실행 명령/종료코드 |
| `artifacts/technical_smoke/*/summary.json` | PID, CPU 시간, RSS, 손실, 표본 내 혼동행렬 |
| `config.json`, `docs/config_freeze.json` | 학습 전에 고정한 설정과 소스 해시 |
| `artifacts/technical_smoke/pretraining_freeze.json` | 첫 update 전 고정한 데이터/표본 해시 |
| `docs/WORKED_EXPLANATION_KO.md` | 질문/숙제 없는 코드 해설 |
| `docs/INDEPENDENT_AUDIT.md` | 독립 AI 감사 결과 |
| `ATTRIBUTION.md`, `licenses/CC-BY-4.0-NOTICE.md` | 데이터 출처·변형·라이선스 구분 |

코드와 문서는 **AI_ASSISTED**, 사용자 능력은 **NOT_ASSESSED**입니다. 이 결과물을 사용자의 단독 작성물이나 이미 검증된 직무 능력으로 기재하면 안 됩니다. 별도 원본 Stage1 ZIP/모델은 수정하지 않았습니다.
