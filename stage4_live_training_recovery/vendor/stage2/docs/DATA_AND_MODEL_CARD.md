# Data/model card

- Source: Andriiashen/CWI 2024 dual-energy chicken-fillet dataset, CC BY 4.0; see ATTRIBUTION.md.
- Included source payload: exactly Original004 bone and Original006 normal, 8 unchanged TRAIN TIFFs copied from Stage1. Both image pairs belong to Chicken1: one physical fillet, not two independent specimens.
- Excluded: original validation, purged, test, Dataset2, private KAMP, all other TRAIN image payloads. Metadata can mention excluded partitions; no excluded pixels/labels are opened.
- Fixed population: 4,096 uniform unique pixel indices per example, seed 20261026. Sample-only scaler. The second mask page (zero-based page 1) is used as the supervised target.
- Observed labels: 1 positive / 8,191 negative. Final sampled prediction confusion at threshold 0.5: TP0, FN1, FP0, TN8191. Sampling is not a class-balanced research design.
- Fitted model: newly initialized CPU float64 7→16→1 tanh/dropout MLP; 145 parameters. Fixed SGD learning rate .03, momentum .9; 4 epochs / 128 updates. No search, early-stopping selection, validation threshold selection or calibration.
- Scope: software training/checkpoint smoke, not an image-level detector evaluation. All reported losses/confusion are resubstitution on those same 8,192 TRAIN pixels. No whole-image inference metrics are claimed in Stage2.
- Baseline relationship: this is not the prior 87-TRAIN-example linear model. The original frozen model is not overwritten, and its earlier development-validation failure is not repaired by this lab. `STAGE1_BASELINE_FREEZE.json` and `STAGE1_INPUT_HASHES.txt` preserve references. No side-by-side performance comparison is performed.
- Checkpoints are new derived model artifacts with CC BY source attribution, not replacements for Stage1. No deployment, safety, food inspection or medical claims.
- Authorship AI_ASSISTED; user competency NOT_ASSESSED. No quiz/homework was assigned.
