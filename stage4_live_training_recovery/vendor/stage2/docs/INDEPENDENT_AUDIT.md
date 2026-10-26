# Independent Stage 2 audit — 2026-10-26

**PASS: tiny CPU training and same-runtime checkpoint/restart correctness only.** This is an independent AI implementation review, not human certification, model validation, production approval, or assessment of the user's competence. Authorship is **AI_ASSISTED**; user competency is **NOT_ASSESSED**.

## Independently verified

- Observed config SHA-256 `5646af441474857f99387c1282b38f3c83cd66aaaa443c3a0603a36e064804cf` before real training artifacts existed. The final config is unchanged. The recorded data/index freeze also precedes the subprocess training runs.
- All eight public source TIFFs match Stage 1 bytes, fetch-receipt SHA-256, allowlist size/CRC32, and original frozen TRAIN split metadata for **Original_ID 004 and 006 only** (two image pairs from the same physical Chicken1 fillet). Source NoBone/000.tiff corresponds to Original006, not Original000. No validation, purged, test, Dataset2, or private KAMP payload was accessed by this audit.
- Independently reconstructed all **8,192 fixed pixel indices**, sampled labels, seven features, sample-only means/scales, and standardized matrix. They match exactly. The scaler uses only the selected training rows; no held-out rows exist in this experiment.
- Checked every one of the **145 trainable parameters** with central finite differences on separate synthetic inputs while holding the training-mode dropout RNG fixed. Maximum absolute gradient error was **1.1544e-10** (tolerance 1e-8). Two synthetic SGD/momentum steps match the manual update formula to **6.94e-18** in parameters and exactly in momentum buffers. Gradients are cleared between steps, and parameters really change.
- Re-read the final proof artifacts independently: uninterrupted **128** steps and fresh-process **37 + 91** steps have byte-identical parameters, momentum buffers, final predictions, Torch RNG, and current sample permutation. JSON loss history, PCG64 sampler state, progress, and complete checkpoint metadata also match exactly. The saved interruption is genuinely mid-epoch: epoch 1, cursor 1280, step 37. Final progress is epoch 4, cursor 0, step 128.
- Independently loaded the final parameter arrays into a newly created model: all four parameter arrays differ from the frozen initialization, all four momentum buffers are nonzero, and recomputed predictions and training loss exactly match the saved outputs. No optimizer update was performed in this check.
- Reviewed subprocess code and three distinct process-ID receipts. This is fresh **OS-process** restart evidence in one recorded CPU software environment, not a new machine/container or cross-version reproducibility claim.
- **35 artifact/data checks** passed. **29 additional independent malformed-format cases** were all rejected, including boolean/numeric schema aliases, invalid RNG/progress/loss values, object dtype, Fortran layout, wrong dtype, nonfinite parameters/momentum, ZIP traversal/duplicates/extra members, encoded and decoded size bounds, truncated array bodies, and incorrect data counts/status. Detailed cases and audited source hashes are in [`independent_audit.json`](independent_audit.json).

## Persistence and resource boundary

The closed checkpoint format is JSON plus NPZ with `allow_pickle=False`, bounded JSON and archive sizes, bounded NPY headers, exact keys/shapes/dtypes, finite numeric checks, configuration/data hashes, strict progress consistency, and RNG/permutation validation. It is not an arbitrary PyTorch/pickle loader. Hashes detect inconsistency/corruption but are **not authenticity signatures** against an attacker replacing both files and their manifests. This audit is not a general parser penetration test.

The verified backend is **PyTorch 2.6.0+cpu**, installed from an official CPU wheel. CUDA is unavailable and no GPU is used. The final training subprocesses report a maximum **287,492 KiB peak RSS** (about 281 MiB), well under 2 GiB. The auditor performed no extra real-data optimizer run, no model search, no threshold tuning, and no external upload.

## Important negative result

The prospectively fixed uniform sample contains **one positive and 8,191 negative pixels**. At the frozen 0.5 threshold the final model predicts **all negative: TP 0, FN 1, FP 0, TN 8191**. Training loss falling from about 0.8331 to 0.005708 is **not successful bone detection**; extreme imbalance makes low loss compatible with missing the only positive. All displayed losses/counts are training resubstitution on these two examples, not held-out performance or full 87-example training. No resampling or tuning was used to hide this result, and this does not repair or supersede Stage 1's prior detection failure.

## Recorded implementation repair

The first attempt failed before an optimizer update because NumPy 2.3.5 no longer exposed a private NPY-header helper. The builder switched to public header-reading functions and preserved the failed logs. A later metadata-type hardening change was followed by the same frozen-config proof, not a new configuration search. This audit tested the final source hashes and final `artifacts/technical_smoke` evidence.

The package provides worked explanations and AI-assisted implementation evidence, not a quiz, a claim of independent personal mastery, or a deployed service.

Final documentation review found local Markdown links valid, source/derived-data attribution retained, and the blocked HTTP 403 retrieval of the full CC BY legal text explicitly recorded with official links instead of a false bundled-text claim. No workaround was attempted.
