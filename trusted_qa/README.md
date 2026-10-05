# Independent Mool Katha QA runner

This is a prototype for the separate private `ashivam-dot/mool-katha-control`
repository. It reads one immutable `ashivam-dot/mool-katha` Git commit and one
private Modal draft archive. It never imports or executes code from that source
checkout. An approved run writes an **unsigned** `agent_episode_qa_v1` review;
it has no publisher, release, or signing code. A failed or uncertain run writes
`private/agent-qa-hold.json` and exits with status 2.

## Entry points

Run these from the trusted control checkout with Python 3.12 and the locked
environment (`uv sync --frozen --project trusted_qa --python 3.12`):

```sh
python -m trusted_qa discover --source-repo source --source-commit <40-char Git commit>
python -m trusted_qa fetch-archive --episode ep004 --output private/draft.tar
python -m trusted_qa run --source-repo source --source-commit <40-char Git commit> \
  --episode ep004 --archive private/draft.tar --output-dir private/qa-ep004
```

`discover` emits a JSON list of pending episode IDs, final-video SHA-256 values,
and the exact source commit. It skips malformed records and legacy drafts without
valid cloud producer provenance, without clearing them. `fetch-archive` invokes only Modal `mool-katha/draft_archive`; the draft
workflow puts that call in a job with no model key. `run` requires actual GitHub
Actions run and workflow provenance. Without `--archive`, `run` performs the
same Modal read itself. The Python API is `runner.discover_pending`,
`archive.fetch_modal_archive`, and `runner.run_one`.

The review appears only after committed candidate fields, every archived file,
assets, source and rights pages, full MP4 decode, sampled frames, two complete
audio recognizers, a separate full-audio quality observation, and the final
semantic decisions validate. HTTP raw bodies live at
`agent-qa-responses/<sha256>.bin`; readable UTF-8 text/OCR snapshots live at
`agent-qa-snapshots/<sha256>.txt`. Both are hashed again before assembly. Gemini
and Whisper raw ASR results are inside their SHA-256-bound ASR JSON files.
Provider requests/responses, the full WAV, audit receipts, and the audio-quality
model observation are retained in the private output. A copy of the normalized
quality observation is also saved as `agent-audio-quality-observation.json` in
the episode; `audio_review.quality_observation` binds its exact file and SHA-256.
The observation records the input video/audio hashes, provider request/response
hashes, model call, and uncertainty. It is a model judgment, not human listening.
Any voice-quality concern or material uncertainty holds the run.

The visual stage also decodes every frame to a measured 120×214 RGB image and
saves `agent-video-frame-audit.json` with consecutive indices, presentation
times, pixel hashes, luminance statistics, and previous-frame differences.
Flat frames and byte-identical spans longer than two seconds hold. Every frame
appears once in a content-addressed JPEG sheet of at most 35 indexed tiles.
The explicitly configured Flash quality model judges every sheet and must
return a clear, low-uncertainty verdict naming every index. Raw batch requests
and responses remain private; their hashes, actual model calls, and exact sheet
references appear in `qa_run.frame_batch_review`. The later full-size contact
sheet and crops still carry the separate caption/source visual judgment.

## Required configuration

| Scope | Variables | Value or purpose |
| --- | --- | --- |
| Trusted Actions run | `GITHUB_REPOSITORY`, `GITHUB_WORKFLOW_REF`, `GITHUB_WORKFLOW_SHA`, `GITHUB_RUN_ID`, `GITHUB_RUN_ATTEMPT` | Actual GitHub context. Repository must be `ashivam-dot/mool-katha-control`; workflow ref may be a branch for shadow QA or `refs/tags/qa-v20`. Export `github.workflow_sha` as `GITHUB_WORKFLOW_SHA`. |
| Model review job secret | `QA_GEMINI_API_KEY` | QA-only Gemini API credential; never present in the Modal fetch job. |
| Explicit Gemini models | `QA_GEMINI_ASR_MODEL`, `QA_GEMINI_QUALITY_MODEL`, `QA_GEMINI_REVIEW_MODEL` | Draft: `gemini-2.5-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`. The quality model also reviews indexed frame sheets; the final model reviews sources, ASR, contact sheet and crops. Provider-returned model versions and request IDs are retained. Confirm access with a shadow run. |
| Pinned local ASR | `QA_WHISPER_MODEL_REPO`, `QA_WHISPER_MODEL_REVISION`, `QA_WHISPER_MODEL_SHA256`, `QA_WHISPER_MODEL_DIR` | `Systran/faster-whisper-large-v3`, commit `edaa852ec7e145841d8ffdb056a99866b5f0a478`, `model.bin` SHA-256 `69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1`, and its downloaded local directory. |
| Modal archive fetch job secret | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Separate credential for the existing archive-read function. The function path is read-only in code; the token's provider-side scope still needs verification. |
| Source checkout secret | `SOURCE_READONLY_DEPLOY_KEY` | Existing read-only GitHub deploy key for `ashivam-dot/mool-katha`. Both checkouts use `ssh-key` and `persist-credentials: false`. |

The local Whisper directory must contain regular `model.bin`, `config.json`,
`tokenizer.json`, and `vocabulary.json` files. The runner checks the pinned
`model.bin` hash before loading CTranslate2 and records the installed
`faster-whisper` version in the ASR result. The dependency lock pins PyAV
15.1.0, whose WAV decoder is compatible with faster-whisper 1.2.1. Whisper
uses 15-second internal windows over the complete WAV with VAD off; the
shorter windows recovered ep022 beats omitted by the observed 30-second
decode. System tools: `ffmpeg`, `ffprobe`,
`pdftotext`, and `tesseract` with Hindi and English data. Direct Python
dependencies and transitive hashes are in `pyproject.toml` and `uv.lock`.

## Workflow and trust boundary

[`workflow-draft.yml`](workflow-draft.yml) remains an inert reference. The
control repository's `.github/workflows/shadow-qa.yml` runs only on manual
dispatch. It discovers at most one candidate, fetches its archive in a
Modal-only job, and reviews it in a separate job with the model key and an
immutable source commit. It uploads private discovery, archive, and unsigned
QA/hold artifacts with short retention. It has no release or signing step.
When discovery finds no eligible candidate, it saves a discovery hold and
does not fetch media or call a model. Branch dispatches are for shadow QA
while the release gate is disabled.

The signed gate must accept and recheck the
`audio_review.quality_observation` and `qa_run.frame_batch_review` references
before a live release. Raw model
request/response bodies stay private; their SHA-256 values are inside the
observation, so the gate can verify the signed normalized file but cannot
independently rehash private raw bodies. The runner rechecks those private
bytes itself. The unsigned review also records the quality model call and a
labeled summary.
Model assessments of accent or pronunciation do not replace native Hindi
listening. Full video decoding verifies stream integrity; visual judgment uses
the saved contact sheet and crops, so uncertain visuals hold.

## Verification

```sh
trusted_qa/.venv/bin/python -m unittest discover -s trusted_qa/tests -v
python3.12 -m compileall -q trusted_qa
```

The tests use synthetic media and mocked provider output except one real
two-second FFmpeg decode. Those fixtures check code and gate schema; they are
never production QA evidence.
