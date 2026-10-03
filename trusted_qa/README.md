# Independent Mool Katha QA runner

This is a prototype for the separate private `ashivam-dot/mool-katha-control`
repository. It reads one immutable `ashivam-dot/mool-katha` Git commit and one
private Modal draft archive. It never imports or executes code from that source
checkout. An approved run writes an unsigned `agent_episode_qa_v1` review by
default. An explicitly configured run signs the just-assembled exact review
with a QA-only Ed25519 key. It has no publisher or release credentials. A failed or uncertain run writes
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

For a separately reviewed tagged QA workflow, add `--qa-key-file <private
32-byte-seed-file> --qa-key-id <reviewed-key-id>` to `run`. The runner reads
the key only after its fixed QA checks pass and writes a detached signature
beside the review. Keep the key file in the trusted runner only, outside the
producer checkout and uploaded artifacts. The existing draft workflow does
not supply the key and remains unsigned.

`discover` emits a JSON list of pending episode IDs, final-video SHA-256 values,
and the exact source commit. It skips malformed records and legacy drafts without
valid cloud producer provenance, without clearing them. `fetch-archive` invokes only Modal `mool-katha/draft_archive`; the draft
workflow puts that call in a job with no model key. `run` requires actual GitHub
Actions run and workflow provenance. Without `--archive`, `run` performs the
same Modal read itself. The Python API is `runner.discover_pending`,
`archive.fetch_modal_archive`, and `runner.run_one`.

The review appears only after committed candidate fields, every archived file,
assets, source and rights pages, full MP4 decode, every-frame pixel and model
review, sampled readable frames, two complete audio recognizers, a separate
full-audio quality observation, and the final semantic decisions validate.
For every external asset, the runner fetches both the official origin and rights
pages. They must identify the same `source_object_id`; the rights page must
name the specific licence and connect to the origin or exact file. The origin
must visibly print the exact used SHA-256 or link an `official_asset_url` whose
downloaded bytes match the used file. A generic licence page cannot clear a
specific asset. Each signed `asset_findings` item carries `origin_proof` with
the ledger object ID and rights basis, a content-addressed origin response and
snapshot, and either the visible file SHA-256 or an exact-byte official
download reference. Hidden HTML text and links are excluded. External
stylesheets and CSS hiding selectors the parser cannot check hold; ambiguous
inline styles are excluded. Internal visual, voice, font,
music, and sound-effect assets hold until a separate control-owned provenance
verifier exists. Generated animation holds pending dedicated provenance and
frame validation.

HTTP raw bodies live at
`agent-qa-responses/<sha256>.bin`; readable UTF-8 text/OCR snapshots live at
`agent-qa-snapshots/<sha256>.txt`. Both are hashed again before assembly. Gemini
and Whisper raw ASR results are inside their SHA-256-bound ASR JSON files.
Each beat citation must include every overlapping recognizer segment in full;
both recognizers must preserve every ordered Hindi word in the frozen beat,
including grammatical words and repetitions.
Provider requests/responses, the full WAV, audit receipts, and the audio-quality
model observation are retained in the private output. A copy of the normalized
quality observation is also saved as `agent-audio-quality-observation.json` in
the episode; `audio_review.quality_observation` binds its exact file and SHA-256.
The observation records the input video/audio hashes, provider request/response
hashes, model call, and uncertainty. It is a model judgment, not human listening.
Any voice-quality concern or material uncertainty holds the run.

The pixel audit decodes every final MP4 frame at 120×214, checks frame count,
timestamps, blank or uniform frames, and isolated temporal anomalies. It saves
`agent-video-frame-audit.json` and numbered, content-addressed sheets of at
most 36 frames in `agent-video-frames/`. A separate Gemini request judges each
sheet. Every response must declare all indices in its sheet checked, with a
clear decision, low uncertainty, and no defects. The signed
`qa_run.frame_batch_review` binds the video hash, audit file/hash, frame count,
sheet hashes and ranges, decisions, model calls, and private request/response
hashes. The contact sheet and full-size crops separately support readable
caption and source inspection. Runs exceeding 72 review sheets hold.

## Required configuration

| Scope | Variables | Value or purpose |
| --- | --- | --- |
| Trusted Actions run | `GITHUB_REPOSITORY`, `GITHUB_WORKFLOW_REF`, `GITHUB_WORKFLOW_SHA`, `GITHUB_RUN_ID`, `GITHUB_RUN_ATTEMPT` | Actual GitHub context. Repository must be `ashivam-dot/mool-katha-control`; workflow ref may be a branch for shadow QA or `refs/tags/qa-v1`. Export `github.workflow_sha` as `GITHUB_WORKFLOW_SHA`. |
| Model review job secret | `QA_GEMINI_API_KEY` | QA-only Gemini API credential; never present in the Modal fetch job. |
| Explicit Gemini models | `QA_GEMINI_ASR_MODEL`, `QA_GEMINI_QUALITY_MODEL`, `QA_GEMINI_REVIEW_MODEL` | Draft: `gemini-2.5-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`. Provider-returned model versions and request IDs are retained. Confirm access with a shadow run. |
| Pinned local ASR | `QA_WHISPER_MODEL_REPO`, `QA_WHISPER_MODEL_REVISION`, `QA_WHISPER_MODEL_SHA256`, `QA_WHISPER_MODEL_DIR` | `Systran/faster-whisper-large-v3`, commit `edaa852ec7e145841d8ffdb056a99866b5f0a478`, `model.bin` SHA-256 `69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1`, and its downloaded local directory. |
| Modal archive fetch job secret | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Separate credential for the existing archive-read function. The function path is read-only in code; the token's provider-side scope still needs verification. |
| Source checkout secret | `SOURCE_READONLY_DEPLOY_KEY` | Existing read-only GitHub deploy key for `ashivam-dot/mool-katha`. Both checkouts use `ssh-key` and `persist-credentials: false`. |

The local Whisper directory must contain regular `model.bin`, `config.json`,
`tokenizer.json`, and `vocabulary.json` files. The runner checks the pinned
`model.bin` hash before loading CTranslate2 and records the installed
`faster-whisper` version in the ASR result. System tools: `ffmpeg`, `ffprobe`,
`pdftotext`, and `tesseract` with Hindi and English data. Direct Python
dependencies and transitive hashes are in `pyproject.toml` and `uv.lock`.

## Workflow and trust boundary

[`workflow-draft.yml`](workflow-draft.yml) is inert in this directory. In a
future control repository, it can poll one pending candidate per manual dispatch,
fetch each archive in a Modal-only job, and review it in a separate job with
the model key and an immutable source commit. It uploads private archive and
unsigned QA/hold artifacts with short retention. It has no release or signing
step. Dispatching a workflow on the immutable `qa-v1` tag records that tag's
`github.workflow_ref` and exact `github.workflow_sha`; branch dispatches are
for shadow QA while the release gate is disabled.

The signed gate must accept and recheck the new
`audio_review.quality_observation` reference and `qa_run.frame_batch_review`
with its audit and sheet files before a live release. Raw model
request/response bodies stay private; their SHA-256 values are inside the
observations, so the gate can verify the signed normalized files but cannot
independently rehash private raw bodies. The runner rechecks those private
bytes itself. The unsigned review records each model call and a labeled
summary.
Model assessments of accent or pronunciation do not replace native Hindi
listening. Model inspection of numbered frames does not replace human viewing;
uncertain visuals hold.

## Verification

```sh
trusted_qa/.venv/bin/python -m unittest discover -s trusted_qa/tests -v
python3.12 -m compileall -q trusted_qa
```

The tests use synthetic media and mocked provider output plus short real
FFmpeg decodes. Those fixtures check code and gate schema; they are
never production QA evidence.
