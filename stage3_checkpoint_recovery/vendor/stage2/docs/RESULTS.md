# Observed verification result — 2026-10-26

**PASS: small CPU neural training and same-runtime fresh-process checkpoint recovery. NOT a detection-performance success.**

- Official CPU PyTorch 2.6.0+cpu, 145 parameters, 8,192 fixed training pixels from Original004/006 only: two image pairs from one physical fillet, Chicken1.
- 128 uninterrupted optimizer updates exactly equal 37+91 fresh-process updates in parameters, momentum, predictions, loss history, Torch RNG, PCG64 sampler state, permutation and progress. Maximum parameter/prediction difference: 0.
- 20 built-in malformed-checkpoint cases rejected; 14 unittest tests passed.
- Independent AI audit: 35 data/artifact checks; 29 extra malformed cases; 145 dropout-on finite differences; two-step manual SGD/momentum comparison. See INDEPENDENT_AUDIT.md.
- Initial train loss 0.8330589738; final train loss 0.0057082381. All losses finite.
- **Only 1 positive versus 8191 negative labels; final predictions all negative: TP0, FN1, FP0, TN8191.** No generalization, successful bone detection or improvement claim.

## Actual resource observations

| Process | Updates in process | Training process CPU s | Training wall s | Peak process RSS KiB |
|---|---:|---:|---:|---:|
| interrupted | 37 | 0.016078 | 0.016083 | 287488 |
| resumed | 91 | 0.042830 | 0.042835 | 287428 |
| uninterrupted | 128 | 0.059214 | 0.059220 | 287492 |

CPU timing covers the update loop; process RSS covers the whole child process, including framework/data. Startup, feature preparation and dependency install are not the training-loop time. Full process wall times are in process_commands.json and summary.json. These are measured smoke-test observations, not throughput benchmarks. All three subprocesses use only CPU.

The frozen config and selected sample indices were unchanged after observing the one-positive/all-negative outcome. The original Stage1 model and Stage1 ZIP hashes remain intact. No remote upload/publication is performed by this artifact.

Authorship AI_ASSISTED; user competency NOT_ASSESSED.
