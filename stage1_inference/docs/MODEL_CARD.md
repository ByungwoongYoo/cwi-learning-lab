# Frozen model card

## Intended use
A private/local educational software artifact showing real input handling, inference, testability and honest false-alarm analysis. **Not** a food-safety, industrial inspection or medical system. No new scientific model search, tuning or training was performed in this project.

## Architecture and provenance
Two CPU logistic classifiers copied byte-for-byte from the frozen 2026-10-26 developmental baseline. `low` consumes three features; `dual` consumes seven. `models.json` SHA-256: `0b6bce8f46b677c834048994c3ea72cba26e4d6a4172cd88018190ed7cbb9416`.

Features, in order: L; L − uniform_filter(L,9,reflect); L − uniform_filter(L,31,reflect); H; H − uniform_filter(H,9,reflect); H − uniform_filter(H,31,reflect); L−H. Low uses the first three. Float32 input/features are preserved, then stored float64 scaler/coefficient arrays reproduce the original arithmetic. Logistic scores are clipped at logits ±700 for numerical stability, **not calibrated probabilities**.

Original fitting: training-sample StandardScaler and sklearn LogisticRegression(C=1, class_weight=balanced, lbfgs, max_iter=300, tol=1e-4, random_state=20261026). Sampling used up to 2048 positive and 2048 nonbone pixels per lineage. The frozen split had 87 training lineages (56 normal/31 positive), 20 developmental validation lineages and 47 purged bridge lineages. It was explicitly developmental, not pristine blind validation.

Default postprocessing is score ≥0.5, 8-connectivity and components ≥25 pixels. Entire image is scored. No ground-truth ROI is used. Changing thresholds in the app is labeled exploratory and does not change or reevaluate the stored validation claim.

## Known failure
Both low and dual alarmed on **10/10 normal validation images**. There were only two correlated fillets in that validation, so this is not an industrial false-positive-rate estimate. Positive-image any-alarm is not localized detection performance. No fresh validation/test inference was run for this app.

The bundled normal and bone examples were in training. Their overlays demonstrate debugging, not generalization. Synthetic input is not real X-ray data. No safety conclusion is possible from a demo alarm or lack of alarm.

## Limits
Input pairing is checked by shape, not physical registration/energy metadata. Same-shaped wrong pairs cannot be automatically detected. New devices, acquisition settings and anatomy/foods are out of scope. Display contrast is separate from inference. JSON coefficients are data, not executable serialized objects.

## Status
`NOT_DEPLOYED`; `AI_ASSISTED`; `USER_COMPETENCY_NOT_ASSESSED`. No CUDA, GPU training, distributed training, deployment availability or user mastery claim is made. Runtime licenses belong to the corresponding packages. Derived weights retain public-data attribution.
