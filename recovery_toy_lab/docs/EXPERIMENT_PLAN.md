# Fixed reproduction plan

Before the first package demo: preserve Stage4's five fixed fault hooks and 128-update reference trajectory, replace CWI-derived samples with deterministic PCG64 synthetic 8192x7 features and balanced labels, and package only source plus receipts. No model search or detector-quality experiment.

Primary outcome: uninterrupted and restarted checkpoint metadata, model parameters, SGD momentum, torch RNG, sampler RNG/permutation, progress, loss history and predictions exactly equal within the same runtime. Expected resumed steps [0,32,32,32,64], expected selected-after steps [32,32,32,64,128]. Terminal recheck launches zero workers.

Boundaries: direct child process only, deterministic pause markers, eight-launch/four-stagnant default bounds, no supervisor-crash support. Reject invalid inputs without overwriting prior output. Stop on permission or environment errors, no alternative-process kill.

Packaging outcome: wheel/sdist build using already installed tooling, no-index/no-deps local target install, help/doctor/entry point, fresh working directory full run, dependency-missing exit3, bad syntax exit2, bad execution/input exit1. Python 3.12/Linux only actually tested. Separate independent audit and final extracted-package rerun required before release.
