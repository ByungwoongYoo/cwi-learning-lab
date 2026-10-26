# Supported interface

Stable for this artifact: `python -m recovery_tool doctor`, `demo --output NEW_DIRECTORY`, `inspect --output EXISTING_DIRECTORY`, and equivalent `recovery-toy` console entry point.
The internal modules `store`, `worker`, `supervisor`, `data`, `train`, and `common` implement one fixed adapter. They are not a public arbitrary-model plugin API. To use another model, dataset shape, optimizer, schedule or backend, implementation changes and fresh validation are required. `_worker` is internal, not a user workflow.

Output must have an existing, writable parent, a fresh simple basename of at most 48 characters, and no symlink ancestors. Runs never overwrite existing directories and never automatically delete failed staging directories. The environment variable RECOVERY_TOOL_RUNS is an internal child transport set by the public CLI, not a promise of isolated execution.

No network client or downloads are implemented in the tool. Dependencies are preinstalled; installing missing dependencies is a separate action. Local tests used no-index/no-deps installation. We do not claim OS-level network sandboxing or adversarial multi-user filesystem race protection.
