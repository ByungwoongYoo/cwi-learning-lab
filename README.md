# CWI Learning Lab

Two **AI-assisted, local CPU** software projects: fixed-model inference with honest false-alarm review, and a tiny PyTorch training/checkpoint-restart experiment. This is not a successful detector, deployed service, or certification of personal competence.

**PUBLIC_SOURCE_RELEASE · NOT_DEPLOYED · UI_NOT_RUN · USER_COMPETENCY_NOT_ASSESSED**

| Project | What was demonstrated | Main limitation |
|---|---|---|
| [Stage 1: inference](stage1_inference/README.md) | Frozen JSON model inference; local HTTP upload/inference; 41 tests and 65 separate, potentially overlapping AI checks | Both baseline models alarmed on all 10/10 normal development-validation images; no browser UI QA |
| [Stage 2: training](stage2_training/README.md) | Real CPU PyTorch, 145 parameters; exact checkpoint restart: 128 updates = 37 + 91 in a new process | Only 1 positive among 8,192 training pixels; final predictions all negative |

## Run locally

Follow each stage's README in a separate Python 3.12 environment. Stage 1: `python -m pytest -q`, then `python app.py`. Stage 2: `python -m unittest discover -s tests -v`, then `python -m training_lab.verify --out artifacts/my_reproduction`. Requirements and CPU install commands are in those READMEs. No GitHub Actions workflow is included.

The shared public dataset subset is **two image pairs, Original004/006, from one physical Chicken1 fillet**, not two independent specimens. Each standalone stage contains the same eight original TIFFs. Dataset #2, held-out payloads, private data, dependency wheels, environments and caches are not included.

## Evidence, attribution and publishing boundary

- [Evidence and non-claims](EVIDENCE.md)
- [License index](LICENSE_INDEX.md): original code/docs MIT; CWI data CC BY 4.0; data-derived models/results retain attribution.
- [Public-release manifest](PUBLIC_RELEASE_MANIFEST.json): frozen source hashes, sanitized-file ledger and proposed target.
- [Preparation checks](PUBLIC_PREPARATION_REPORT.json): local snapshot checks, not browser or deployment evidence.

This is the **public source release** at [ByungwoongYoo/cwi-learning-lab](https://github.com/ByungwoongYoo/cwi-learning-lab). The owner approved public sharing of the code, models, public-data examples and verification results. Source publication does **not** establish a running web demo, Hugging Face deployment or CI execution.

The original release ZIPs were not modified and are not duplicated inside this snapshot. Original code, model coefficients, checkpoint arrays and image bytes are preserved; selected documentation/logs were sanitized, with hashes recorded. No CUDA, WebGL, multi-node execution, independent user mastery, food-safety, industrial or medical-use claim is made.
