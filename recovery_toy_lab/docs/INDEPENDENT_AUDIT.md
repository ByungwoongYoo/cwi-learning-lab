# Independent packaging and reproducibility audit

Verdict: PASS for the fixed synthetic CPU adapter.

## Executed independently

- Installed the final wheel into a fresh temporary target with `pip --no-index --no-deps`; ran from an unrelated working directory with that installed target on Python’s module path. This reused the existing Python environment and dependencies, not a fresh machine.
- 16 CLI/install checks passed: help, doctor, invalid command, missing argument, missing parent, invalid basename, symlink parent, parent traversal, full demo, existing-output refusal without byte changes, inspect, console entry point, and dependency-free help / missing-dependency doctor and demo.
- Final full installed demonstration: 10.958 seconds. Five distinct actual training subprocesses exited with SIGKILL at the fixed barriers. Metadata, every state array and predictions exactly matched the uninterrupted reference. A terminal check launched no additional training.
- Synthetic data had exactly 8,192 rows and 4,096 examples of each class.
- Independently extracted the sdist after path/link checks and built its wheel locally without network; all packaged Python source bytes matched the final wheel. The wheel’s packaged source bytes also matched the final source tree.
- Reran the 12 builder unit tests successfully. Do not sum these with other checks as unique tests.
- Verified the stored manifests of the original Stage1/2 public snapshot, Stage3 public snapshot and Stage4 release. Listed source files remained unchanged.

## Review

README and API_SCOPE correctly limit support to one synthetic MLP/SGD/dropout trainer, Linux and the recorded environment. Dependencies are declared and are not bundled. Output lives outside the installed package and existing output is refused. No CWI image download is needed. Source inspection shows only the newly spawned direct child process is targeted, with identity and marker checks.

## Limits

This is not a clean-OS installation test, general checkpoint framework, security certification or real-world model-performance result. The supervisor must remain alive. Failure injection uses deterministic barriers rather than arbitrary-time faults. Power loss, concurrent writers, GPU, distributed execution, other dependency versions, OS-level network isolation and human competency remain unverified. No public publishing was performed by this audit.

Machine-readable checks, source hashes and artifact hashes: INDEPENDENT_AUDIT.json.
