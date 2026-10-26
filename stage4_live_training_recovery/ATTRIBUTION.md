# Attribution and license boundaries

## Public source data and derived arrays/weights
Vladyslav Andriiashen (data collector), Centrum Wiskunde & Informatica (2024), **Dual-energy X-ray inspection of chicken fillets containing rib bone fragments**, v1. DOI: https://doi.org/10.5281/zenodo.10579608. Record: https://zenodo.org/records/10579608. Rights source: https://ir.cwi.nl/pub/33988.

The public data are CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. The license notice and official full-text links are in `vendor/stage2/licenses/CC-BY-4.0-NOTICE.md`. Full legal text was not bundled because direct retrieval returned HTTP 403; the failed attempt is recorded in `vendor/stage2/licenses/retrieval_status.json`. No creator or CWI endorsement is implied.

Eight original TRAIN TIFFs are unchanged copies from the separately frozen Stage1 package, with original source-member mappings and checksums. Only Original004 (bone) and Original006 (normal) are included; both image pairs belong to the same physical fillet, Chicken1. Source file `NoBone/000.tiff` maps to Original006, not Original000.

Changes/derived material: seven numerical feature planes; 8,192 fixed pixel samples; train-sample centering/scaling; labels sampled from bone mask page 1 (zero-based); tiny fitted MLP weights and momentum; scalar losses and sampled predictions. These derived analyses retain the data attribution and are not source-author results. The separate Stage1 linear model was not copied into the training model and was not overwritten.

## Original code and docs
MIT for independently implemented AI-assisted original code/tests/docs, see `LICENSE`. This does not relicense public data or third-party libraries. The source author's separate GPL-3 code at https://github.com/vandriiashen/pod2settings was not copied or executed. Seven-feature logic was adapted from this project's own independently authored Stage1 implementation.

PyTorch, NumPy, SciPy, Pillow and transitive dependencies retain their own licenses. No dependency source/wheel or environment is bundled. `vendor/stage2/docs/STAGE1_ATTRIBUTION_REFERENCE.md` preserves the previous project attribution as historical context, not a statement that Stage2 uses the Stage1 UI dependencies.

Authorship: **AI_ASSISTED**. User independent competency: **NOT_ASSESSED**.

## Stage4 reuse
The complete public Stage2 vendor tree and Stage3 recovery_lab/store.py are reused unchanged. Stage4 adds its own live-training worker, bounded supervisor, tests and documentation. New run arrays remain derived from the attributed CWI data. No author endorsement or independent user competency is implied.
