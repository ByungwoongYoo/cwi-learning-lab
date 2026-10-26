# Independent implementation audit — 2026-10-26

**Result: implementation regression PASS; browser UI NOT RUN.** This is an independent AI code/test review, not human certification, production approval, model validation, or assessment of the user's competence.

## Tested evidence

**65 independent targeted checks passed:** 31 core checks, 19 parameter/component-metadata checks, and 15 pre-cache transport checks. Detailed names, results and audited code hashes are in [`independent_audit.json`](independent_audit.json).

- Frozen `models.json` is byte-identical to the restored baseline and matches SHA-256 `0b6bce8f46b677c834048994c3ea72cba26e4d6a4172cd88018190ed7cbb9416`.
- On three deterministic signed-float synthetic shapes (16×16, 31×33, 47×63), both low/dual paths exactly matched the original baseline's features, stored-scaler/logit/score arithmetic and cleaned masks. The original `features` and `cleaned` functions were extracted without importing or executing its training/evaluation entry points.
- Verified inclusive score ≥0.5, eight-connectivity, retention of area 25 and removal of area 24, diagonal connectivity, and empty/full cases. Component-list truncation is correct at 99, 100 and 101 retained components; those summary-edge tests use controlled masks, not performance experiments.
- Exact TIFF bounds: **8,388,608 bytes**, **1,048,576 pixels**, **16–1536 per dimension**, one-page classic TIFF, 2-D float32, finite values with absolute magnitude ≤10,000. Exact valid boundaries and one-step invalid boundaries were checked. Signed values survive decoding unchanged. UInt8/UInt16/RGB, multipage, malformed, NaN/±Inf, excessive values, path/URL-as-bytes and mismatched pairs were rejected. Both models reject a mismatched supplied pair.
- Invalid threshold/minimum-area values are safely rejected; fractional minimum areas are not silently rounded. Display contrast and ground truth do not enter model features, region selection or threshold choice.

## Security finding resolved before delivery

The initial `gr.File(type='binary')` callback alone did **not** prevent Gradio 6.30 from fetching a URL-valued `FileData.path` during framework preprocessing. A mock cache/downloader reproduced that path **without making a network request**.

The app now uses an app-scoped `LocalUploadBlocks` guard before any framework cache traversal. It requires a bounded existing file canonically inside the Gradio upload root; rejects remote/scheme-relative/file URIs, outside-root paths, symlink escapes, streams, malformed file values and nested file objects in scalar controls; and forwards only the verified local path and minimal file metadata.

All 15 independent transport tests ran with socket connection disabled. Rejected cases reached **zero cache calls**. Valid local uploads, a `FileData` object, and absolute local URL plus nested remote metadata all round-tripped as the original TIFF bytes; extra metadata was stripped rather than fetched. No pickle/joblib, caller-selected executable code, remote model download, public tunnel or external inference service is used.

This remains a **local learning prototype**, not a hardened multiuser upload service or penetration-tested TIFF parser. CLI file paths are supplied only by the local operator.

## Data and claims

All eight bundled TIFFs match allowlist CRC32/size and fetch-receipt SHA-256. Original_ID 004 and 006 match the frozen TRAIN partition. Total recorded transfer was **9,332,689 bytes**. No KAMP, validation20, purged47, original-test or Dataset2 payload was opened or inferred by this audit. No fitting, new model search or performance evaluation was performed.

Documentation makes the existing **normal-image alarm 10/10 failure**, AI assistance, **NOT_DEPLOYED**, and unassessed user competence explicit. The learning notes are optional worked explanations, not mandatory questions or quizzes. Local Markdown links resolve. No CPU demo is presented as CUDA, distributed-training, deployment or personal-mastery evidence. Hugging Face CPU Basic Gradio creation is subject to eligible plan requirements; CPU hourly pricing is not a free-account deployment promise. [Official Spaces documentation](https://huggingface.co/docs/hub/spaces-overview).

## Browser and runtime boundary

**UI_NOT_RUN**: rendered browser layout, accessibility, interactions and browser upload behavior were not verified. Internal browser-attempt details are omitted from this public snapshot.

Separately, the builder's reviewed [`http_smoke.json`](../artifacts/http_smoke.json) records successful same-process local HTTP page/config/info requests, actual synthetic dual inference and multipart TIFF upload followed by low-model inference. This is HTTP/API evidence, **not browser QA**. The builder reports 41 passing pytest cases in its separate suite. A fresh OS/container install, Windows/user-PC run, remote HF build and public availability are not established.
