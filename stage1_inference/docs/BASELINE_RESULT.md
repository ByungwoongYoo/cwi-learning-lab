# Prior frozen development-validation summary (excerpt)

This is a textual summary of the existing frozen baseline, **not a new evaluation**. No validation image, mask or prediction payload was loaded for the learning app. Twenty correlated lineages from only two fillets (10 positive/10 normal) were used in the prior development validation.

| Model | Positive images with any alarm | Normal images with any alarm | Pooled pixel Dice |
|---|---:|---:|---:|
| always_negative | 0/10 | 0/10 | 0.000000 |
| low | 10/10 | 10/10 | 0.012761 |
| dual | 10/10 | 10/10 | 0.003188 |

Low: TP 7,113 / FP 1,095,881 / FN 4,721. Dual: TP 8,467 / FP 5,290,699 / FN 3,367. Both models failed to suppress normal-image false alarms. Any alarm does not imply localization, and additional TP does not offset uncontrolled FP. No general conclusion against all dual-energy methods follows.

The validation was explicitly developmental, not pristine blind. No industrial false-positive-rate estimate, independence claim or confidence interval is warranted. Frozen settings were 0.5 threshold, 8-connectivity, area≥25. No setting was tuned in this learning-app project.

Textual provenance: prior frozen CPU baseline RESULT_KO.md and validation_summary.json, 2026-10-26. Coefficient freeze checksum is recorded in original_model_freeze.json. Original data: Andriiashen/CWI, https://doi.org/10.5281/zenodo.10579608, CC BY 4.0.
