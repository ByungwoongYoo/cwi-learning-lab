# Reproducing the bounded experiment

Use the README's explicit venv/install commands. Tested: Linux x86_64, Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0. PyTorch 2.6.0+cpu is the tested pin, not a claim of newest release. The CPU wheel came from `download.pytorch.org/whl/cpu`; installation logs are in `artifacts/torch_install*.txt`. About 187 MB of wheel payload was downloaded, well below 1 GiB. The temporary environment is outside the deliverable.

`python -m training_lab.verify --out artifacts/my_reproduction` is the complete pipeline. The output path must not exist. It runs three child processes with one CPU thread, one fixed architecture/config, 8,192 samples and total 256 optimizer updates across the comparison (128 continuous versus 37+91 restarted), not a model search. Each child has a five-minute timeout. Actual observed compute is far below ten minutes and peak RSS far below 2 GiB; timings are observations, not portable guarantees.

For manual inspection using the bundled prepared data:

```bash
python -m training_lab.train --data artifacts/technical_smoke/data --out artifacts/manual_continuous
python -m training_lab.train --data artifacts/technical_smoke/data --out artifacts/manual_partial --stop-step 37
python -m training_lab.train --data artifacts/technical_smoke/data --out artifacts/manual_resumed --resume artifacts/manual_partial/checkpoint
```

Do not change `config.json` and then present the run as this frozen experiment. Config/source hashes are checked. Runtime versions must match when resuming; exact recovery is demonstrated only in the observed environment, not across versions/devices.

## Execution history, including recoverable failure

1. Before training, fixed `config.json` and all eight input SHA-256 values. An independent auditor observed the same config hash before optimizer execution.
2. Initial command failed before any model update because NumPy 2.3.5 does not expose a private array-header helper. Replaced that call with the public v1/v2 NPY header readers. Logs are preserved under `attempt01_*`.
3. First completed proof is preserved under `artifacts/attempt02_first_complete`. No configuration/seed/sample/hyperparameter changed.
4. Metadata type hardening and additional rejection tests were added after review; final identical-configuration proof is `artifacts/technical_smoke`.

Only these engineering repairs were made. No failed detection metric was used to retune training.

## Reference documentation

- [Official PyTorch CPU installation](https://pytorch.org/get-started/locally/)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html): reproducibility is environment-dependent.
- [PyTorch checkpoint concepts](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): restoring optimizer state matters. This lab deliberately implements its own bounded JSON/NPZ serialization rather than pickle-based model loading.

Package hashes detect accidental changes but are not cryptographic signatures. `scripts/verify_package.py` verifies the included manifest after extraction.
