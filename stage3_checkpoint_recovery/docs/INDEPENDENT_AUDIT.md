# Independent audit — 2026-10-26

## Verdict

**PASS within the stated single-writer, same-runtime CPU toy scope.** No remaining required defect found in the inspected implementation. This is an engineering portfolio validation, not a new algorithm, industrial certification, detection-performance improvement, or assessment of the user's independent programming ability.

## Actual independent work

- Read the Stage2 loader, Stage3 store/CLI/experiment, frozen experiment plan, actual receipts and explanatory documentation.
- Executed **35 independent malformed/selection/conflict cases** on the final implementation: schema and type errors; progress inconsistency; wrong configuration/data/sample/runtime/optimizer bindings; loss and RNG errors; nonfinite JSON/arrays; duplicate JSON; extra NPZ members; object/shape/dtype errors; invalid permutation; compressed decoded-size overflow; archive duplicate/path entries; file symlinks; missing/extra members; filename/step mismatch; validated progress rather than modification time; corrupted-latest fallback; all-invalid refusal; same-step conflicting commit refusal with old bytes preserved.
- Executed two additional malformed ZIP compression/header cases. Both were recorded as invalid instead of selected. This is not an exhaustive ZIP-format claim.
- Independently reran the builder's **24 unit-test methods**, all passing. These overlap with the independent cases and must not be added into a unique-test total.
- Independently reran the full actual interruption/corruption trajectory after the final source changes: **19.505 seconds**. Earlier independent replay also passed, before two audit-suggested guard improvements.
- Final replay selected steps **32, 32, 32, 32, 96**; three owned generation-writer child processes ended by SIGKILL. States visible after the three failpoints were **32, 32, 96**. Final checkpoint step was 128.
- Final reference/recovery parameters, optimizer momentum, Torch RNG, sample permutation, all manifest metadata including sampler RNG/loss/progress, and directly evaluated predictions were exactly equal.
- Rechecked all **104 files of the public Stage2 snapshot** against the prior snapshot hash record and against `vendor/stage2`: no differences. The original private Stage2's log text differs from the previously sanitized public snapshot; those historic sanitation differences are not new Stage3 modifications.

## Improvements applied and rechecked

The builder added explicit assertions for the expected fallback sequence, post-kill visibility, and final step. Directory listing now reads only the allowed boundary plus one sentinel rather than eagerly materializing arbitrarily many entries. Final independent malformed tests and full replay passed after these changes. Exact audited source hashes appear in `INDEPENDENT_AUDIT.json`.

## Critical interpretation

The killed writer copies and publishes an already produced checkpoint generation. It is not an arbitrary interruption inside a training instruction or the original checkpoint's byte-level serialization. The recovery trajectory does perform actual further training from automatically selected valid states. Incomplete/corrupt generations remain on disk, and cleanup is dry-run only.

Canonical `gNNNNNN` names must agree with validated saved progress. Alternate spellings are ignored, and committing to an existing canonical step is refused even for otherwise valid conflicting content. This does not claim multi-writer locking or concurrent-write conflict resolution.

Path, link, shape, finite-value, member-count, hash and size checks are useful local input guards. They are not cryptographic authenticity, an operating-system sandbox, protection against malicious concurrent replacement, or a proof that every possible trailing ZIP/NPY representation is rejected. Power loss, fsync durability, filesystem failure, GPU, multiple nodes and cross-version/device parity were not tested.

## Reproduction evidence

- `INDEPENDENT_AUDIT.json`: verdict, scope and exact source hashes.
- `independent_adversarial_results.json`: individual 35-case results.
- `independent_adversarial_tests.py`: executable independent case definitions. Run from the project using the same existing dependency environment. It creates fresh `audit_*` directories within `runs` and does not delete them.
- `independent_replay_result.json`: actual final independent process/selection/parity receipt.

The additional malformed ZIP header checks were separate bounded probes; their outcome is described here and not presented as part of the 35-case result file. No network access, external publication, dependency installation or unrelated-process termination was used for this audit.
