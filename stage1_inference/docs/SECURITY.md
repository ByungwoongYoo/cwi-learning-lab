# Security and privacy scope

This is a local learning prototype, not a production-hardened upload service.

- Default bind: 127.0.0.1:7860; `share=False`; no public tunnel. A future authorized Hugging Face Space can bind 0.0.0.0 via its `SPACE_ID` environment.
- An app-scoped `LocalUploadBlocks` guard rejects remote/unknown file paths and nested FileData before Gradio cache preprocessing; actual uploads must resolve under its upload directory. The callback then accepts upload **bytes**, never arbitrary server paths/URLs. CLI paths are only supplied by the local operator. Fixed bundled paths do not derive from upload names.
- Maximum 8 MiB/file, 16–1536 pixels per dimension, 1,048,576 total pixels. Single-page classic TIFF, mode F float32, 2-D, finite, |value|≤10,000. Pair shape must match. File extension alone is not trusted.
- Gradio launch also caps upload size and allows one inference at a time, queue size 8. Component table capped at 100. Filesystem/code/model exposure is not intentionally enabled.
- Unknown pickle/joblib files are never loaded. Frozen JSON has a hardcoded SHA-256 integrity check.
- No requests/network calls in inference. Gradio analytics disabled; HF offline mode on. Dependencies may have their own behaviors; tests specifically block sockets during callback inference.
- Gradio may write temporary upload/output files. Cache cleanup is configured for a one-hour interval/age while running; this is not instantaneous secure deletion. Do not upload secrets. No user content is intentionally logged or retained by application code.
- Callback input rejection clears previous displayed outputs; transport-level rejection raises a Gradio error before inference (a previous result may remain visible, with its prior receipt). The fixed default is restored whenever exploratory mode is off. Pair identity/registration and image origin cannot be proven from dimensions.
- TIFF parsing uses Pillow. These caps reduce resource abuse but do not establish parser-level exploit resistance. No adversarial penetration test, production isolation, authentication, antivirus or supply-chain vulnerability audit is claimed.

## Runtime
Pinned top-level requirements were exercised in Python 3.12. A tested dependency snapshot is recorded separately. No credentials, telemetry keys, paid API integration, GitHub Actions or deployment token is required. Optional data restoration has a static public TRAIN allowlist and strict 30 MiB cap and is never invoked on uploads.
