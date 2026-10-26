# Frozen engineering plan — 2026-10-26

Scope: original fixed Stage2 145-parameter CPU model and 8192 sampled TRAIN pixels, same runtime. No detection tuning, validation/test data, GPU, networking or external writes.

Run a fresh 128-update reference. Commit generation32. Automatically select latest valid generation before each continuation:
1. Continue to37, kill only own writer after state file closes. Expected fallback32.
2. Continue to48, kill only own writer after complete candidate validation but before publication. Expected fallback32.
3. Continue to64, commit, then deliberately flip one byte of this run's state archive. Expected fallback32 without deletion.
4. Continue to96, kill only own writer after publication. Expected selection96.
5. Continue to128 and commit. Compare complete checkpoint metadata, all parameter/optimizer/RNG/permutation arrays and actual evaluation scores with reference.

Every hook has a file handshake; only the Popen child owned by the test is killed. Per-child40s waits; experiment540s guard. Existing public originals must remain unchanged. Unit cases cover malformed members, bindings, paths, symlink/hardlink, count/size bounds, duplicate generation refusal and no-valid-state failure. One canonical folder name per step prevents numeric alias selection. Conflicting content at a previously committed step is refused rather than overwritten.

Completion requires the frozen trajectory and unit tests passing, independent review and any necessary repairs/retest. Failure is retained, not converted into success. Concurrency, power loss, filesystem durability, cross-runtime/device parity and malicious replacement of all hash-bound content are excluded. Resource caps are scope limits, not production isolation guarantees.
