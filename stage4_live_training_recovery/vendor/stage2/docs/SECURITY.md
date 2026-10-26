# Local persistence boundary

This is a local, fixed-data lab, not a checkpoint upload service. It reads only its own generated checkpoints. Do not load arbitrary third-party files merely because they use `.npz` or this schema.

No `torch.load`, `pickle.load`, `eval`, dynamically imported checkpoint classes, URL fetch or shell expansion is used in checkpoint loading. `np.load(..., allow_pickle=False)` is preceded by SHA-256 verification, exact ZIP member checks, NPY header shape/dtype checks, compressed and aggregate decoded byte bounds (4 MiB), and a 4 KiB header bound. JSON is bounded (128 KiB), duplicate keys and NaN/Infinity are rejected. Parameters must be finite float64; RNG/permutation arrays have exact types/shapes. Progress, metadata, optimizer settings, data/config hashes and runtime must match. The sample permutation must contain every index exactly once.

A hash detects accidental or uncoordinated corruption; it is not a signature, authentication mechanism, or protection if an attacker rewrites both artifact and manifest. Config/package hashes have the same limitation. Dependency/parser vulnerabilities, symlink races, hostile concurrency and filesystem failure recovery are not comprehensively audited. Temporary-directory publication avoids exposing a partially written checkpoint in ordinary local use, but fsync/power-loss durability and distributed filesystems are out of scope.

Training scripts never start a listener or send data externally. The dependency installer is a separate explicit step. No person/patient information or private files are included. These images are chicken-fillet inspection data, not human medical data.
