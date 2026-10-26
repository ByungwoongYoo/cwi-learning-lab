# Data card

## Public source
Vladyslav Andriiashen (data collector), Centrum Wiskunde & Informatica, 2024. **Dual-energy X-ray inspection of chicken fillets containing rib bone fragments**, v1, DOI https://doi.org/10.5281/zenodo.10579608. Source https://zenodo.org/records/10579608. CWI rights record https://ir.cwi.nl/pub/33988 states **CC BY 4.0**, https://creativecommons.org/licenses/by/4.0/ (verified 2026-10-26).

These are nonmedical chicken-fillet X-ray projections, 956×760 pixels, with 40/90 kV paired acquisition. Source images have flat/dark-field and logarithmic preprocessing. Source review masks contain fillet and bone pages; the second page is used only after inference.

## Selection and exclusions
Before running this app's inference, select the lowest Original_ID within each Bone_Class among the prior **frozen training** allowlist. Result: Original_ID 006 (normal, Chicken1) and 004 (bone, Chicken1/Bone3). The exact source member names, offsets, compressed/decoded sizes and CRC are in `examples/allowlist.json`. Source filenames are not equivalent to Original_ID: normal Original006 is source NoBone/000.tiff.

Only their paired images and masks (8 files) are included. **No original validation20, purged47, test, Dataset2 image/mask/prediction payload, or private KAMP asset is included.** Metadata references to exclusions are not payload. Both examples are training inputs and cannot estimate held-out performance.

`examples/public_train/fetch_receipt.json` records exact downloaded members, byte counts and SHA-256. The optional fetcher requires the fixed allowlist hash, checks HTTP 206 and exact Content-Range before reading body, validates ZIP member names and CRC, and caps all reserved transfer bytes at 30 MiB. No runtime fetch is performed by the app.

## Transformations
Original TIFF bytes are unmodified. Review PNGs use 1st/99th percentile display scaling and added model/ground-truth overlay colors; these are adaptations. JSON metrics and model predictions are new analyses, not source-author results or endorsements. Signed values are retained for inference. Synthetic arrays are independently generated with a fixed RNG seed and explicitly labeled **SYNTHETIC_NOT_REAL_XRAY**.

## Privacy and use
No person/patient data are bundled. The app sends uploaded files only to its hosting server for local computation; it does not contact an external inference service. Do not upload personal, sensitive or nonauthorized data. UI caches are temporary, not a security guarantee of zero persistence. See SECURITY.md.
