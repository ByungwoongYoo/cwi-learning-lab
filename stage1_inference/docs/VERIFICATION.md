# Verification receipt — 2026-10-26

## Passed on the build executor

- Python 3.12; pinned NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0, Gradio 6.30.0; pytest 9.0.2. `runtime_versions.json` records tested package versions and environment scope.
- **41 pytest tests passed** after the current code changes (`artifacts/pytest.txt`). Cases cover signed TIFF preservation, malformed bytes, size/dimension/dtype/multipage/NaN/Inf rejection, paired shape, frozen-model hash, exact feature/score arithmetic, 8-connectivity/area boundary, exploratory status, repeat callbacks, stale-output clearing on callback failure, no outbound socket during inference, and pre-cache transport URL/path rejection with a successful real-upload-path roundtrip.
- `python -m compileall -q app.py cwi_lab tests scripts`: passed.
- `python -m pip check`: no broken requirements found.
- Actual CLI synthetic, public TRAIN normal and public TRAIN bone runs wrote `artifacts/` receipts and PNGs. Normal Original006 dual run had 13,617 false-positive pixels. Bone Original004 dual run had TP133/FP14,639/FN232. These are **training-example diagnostics, not new evaluation**.
- Actual local HTTP server checked in one process: `/`, `/config`, `/gradio_api/info` returned 200. HTTP synthetic inference executed the dual model; a multipart float32 TIFF upload then executed the low model through the HTTP transport and pre-cache guard. Exact receipt: `artifacts/http_smoke.json`. Reproduce with `python scripts/http_smoke.py` (local port 7861).
- Public source example fetch: 8 exact allowlisted TRAIN members, **9,332,689 transferred bytes** (under 30 MiB cap); all CRC/size/SHA checks passed. No held-out payload fetched. Data-derived normal review preview was visually inspected.
- Bounded CPU profiling: one warmup + five timed frozen-inference calls per model on TRAIN Original006. Median low 32.429 ms (31.068–34.014), dual 57.879 ms (53.422–59.434); observed OpenBLAS pools each used one thread. Full timing region/environment/limits: `artifacts/cpu_profile.json`. Not a production latency or speedup claim.
- No new fitting, GPU use, external deployment, repository push or GitHub Actions run.

## Not established

- **UI_NOT_RUN**: rendered browser interaction, layout and accessibility were not verified. Actual HTTP/API tests are not a substitute for browser QA. Internal browser-attempt details are not included.
- **NOT_DEPLOYED / HF_NOT_DEPLOYED / ACCOUNT_ELIGIBILITY_UNKNOWN**: no remote build or hosting availability tested.
- This executor used an isolated venv with already installed system-site numerical libraries. A fresh clean-room OS/container install is not claimed. Python version support beyond 3.12 is not claimed.
- No authentication/penetration/supply-chain security audit; no clinical or production model validation; no generalization claim.
- **AI_ASSISTED / USER_COMPETENCY_NOT_ASSESSED**: implementation and verification are AI-performed. User PC installation or independent skill mastery is not established.

Independent AI review is recorded separately in `INDEPENDENT_AUDIT.md`; it is not a human peer-review or certification.
