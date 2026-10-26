# Evidence and explicit limits

## Stage 1 — fixed CPU inference

- [`pytest.txt`](stage1_inference/artifacts/pytest.txt): 41 builder tests passed. [`independent_audit.json`](stage1_inference/docs/independent_audit.json): 65 separate AI checks. These counts can overlap; **do not sum them as 106 independent tests**.
- [`http_smoke.json`](stage1_inference/artifacts/http_smoke.json): real local HTTP page/config/API responses plus synthetic inference and multipart TIFF upload/inference. **UI_NOT_RUN**: no rendered browser layout, interaction or accessibility QA. **NOT_DEPLOYED**. Internal browser-probe details are not part of this snapshot.
- [`BASELINE_RESULT.md`](stage1_inference/docs/BASELINE_RESULT.md): historical development-validation text, not a new evaluation. Both models alarmed on 10/10 normal images; those validation examples came from only two correlated fillets. This is distinct from the bundled TRAIN pair set below.

## Stage 2 — training/restart correctness only

- [`verification.json`](stage2_training/artifacts/technical_smoke/verification.json): actual PyTorch 2.6.0+cpu, float64 CPU MLP `7 → 16 → 1`, **145 parameters**.
- Two TRAIN image pairs, Original004/006, from **one physical Chicken1 fillet**. A fixed sample of **8,192 pixels contains one positive and 8,191 negatives**. Final 0.5-threshold predictions are all negative: **TP 0, FN 1, FP 0, TN 8191**. Training loss reduction is not good detection or held-out performance.
- 128 uninterrupted optimizer updates exactly matched 37 + 91 updates across a fresh-process restart: parameters, momentum, predictions, losses, RNG, sampler/permutation and progress. This is same-runtime checkpoint correctness, not cross-version/device reproducibility.
- Original report: 14 unittest tests; 20 built-in malformed-checkpoint rejections. Independent AI audit adds 35 data/artifact checks, 29 malformed cases, 145 dropout-on finite differences and a manual two-step SGD/momentum comparison. See [`INDEPENDENT_AUDIT.md`](stage2_training/docs/INDEPENDENT_AUDIT.md).
- Earlier failed attempt and repeated engineering proof are retained. No hyperparameter or sample retuning after seeing the one-positive/all-negative outcome.

## Public-snapshot preparation

The local preparation report records fresh package/model/data checks separately from historical execution receipts. Stage 2's unchanged legacy sibling-folder test may skip after the directory rename; the same Stage 1 model hash is checked directly during snapshot preparation. This does not turn the historical 14-test result into a newly executed 14-pass result.

**AI_ASSISTED; user competency NOT_ASSESSED.** No fabricated personal experience, human certification, production validation, public deployment, CUDA, WebGL or multi-node claim. No private data or personal conversation is included. Checksums detect changes, not authorship or cryptographic authenticity.
