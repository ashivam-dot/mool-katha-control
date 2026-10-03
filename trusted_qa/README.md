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
For every ordinary external media asset, the runner fetches both the official origin and rights
pages. They must identify the same `source_object_id`; the rights page must
name the specific licence and connect to the origin or exact file. The origin
must visibly print the exact used SHA-256 or link an `official_asset_url` whose
downloaded bytes match the used file. A generic licence page cannot clear a
specific asset. The ordinary HTTP `asset_findings` item carries `origin_proof` with
the ledger object ID and rights basis, a content-addressed origin response and
snapshot, and either the visible file SHA-256 or an exact-byte official
download reference. Hidden HTML text and links are excluded. External
stylesheets and CSS hiding selectors the parser cannot check hold; ambiguous
inline styles are excluded. New internal designed cards can pass the control
verifier only with an archived exact-output sidecar, matching frozen spec and
beat-selection bytes, reviewed generator source, exact font records, and a
control-rendered typography and background pixel check. The producer's card
grain is random, so this checks every pixel against the source design's bounded
grain instead of claiming byte-for-byte regeneration. New tanpura beds require
the matching sidecar and reviewed source, then all 2,649,600 PCM samples are
compared with independent control synthesis. The fixed SHA-256 allowlist for
both generators must be updated by review for any later code version.

Google Fonts may pass only when the new `font_sources` record binds the exact
rendered TTF and bundled OFL bytes to frozen Git and independent downloads
from the same pinned `google/fonts` revision match both files. The control
verifier also checks OFL 1.1's use and embedding clauses. A direct Google Fonts
URL without those new records, an internal sound effect, a different internal
visual or music source, and generated animation still hold. An ordinary external asset can
still use the existing object-specific HTTP proof.

The legacy Gemini `voice_take` receipt is inspected for its exact archived
take, frozen script, request, generator source, and internally consistent
metadata. It omits the provider's full response and returned audio. Its hashes
cannot independently prove that Google returned the selected take, so this
legacy voice path still holds before an approved review is assembled.

`voice_exchange.issue_voice_exchange` implements a separate control-owned
path. An owner-reviewed `VoiceExchangePolicy` pins the control repository's
`voice-v1` workflow tag, exact workflow SHA, and Ed25519 public key. The issuer
reads `short.yaml` and `script.json` from one frozen Git commit, sends their
canonical TTS request, retains `request.json`, the complete `response.json`,
the returned `provider-audio.wav`, and a derived mono 24 kHz PCM16
`narration.wav`. It signs the raw `exchange.json` record and saves a detached
`signature.json`. The private signing seed and provider key must stay in the
separate control job. No issuing workflow is installed; the offline validation
below uses a mocked provider response.

The studio handoff must place those exact `narration.wav` bytes at
`content/episodes/epNNN/work/narration.wav`, with a manifest
`control_voice_exchange` link and a `control:gemini:<exchange SHA-256>` asset
origin. That link has `schema: control_gemini_tts_exchange_v1`,
`exchange_sha256`, `narration_sha256`, and `narration_asset_id`. The voice asset
also needs the official terms URL as `rights_url` and
`Gemini API Additional Terms` as `license`. The studio's TTS and render path
must accept the control WAV by exact copy, skip its own TTS request and WAV
trimming, and avoid a tempo change. Its manifest must omit `voice_take` and
`narration_transform`; the archive must retain the exact narration WAV. The Python
`runner.run_one` API accepts a bundle path outside both producer inputs and a
separately pinned policy. QA copies the six files to `agent-control-voice/`,
replays the signature, saved response, WAV derivation, frozen request, and
asset hashes, then checks for the narration waveform in the complete mixed
final-video audio. Any future approval would also require both
full-final-audio recognizers; the current rights hold occurs before ASR.

A signed exchange issued by the separate control workflow could resolve the
missing provider-origin evidence; commercial-use rights remain unresolved.
After a valid proof, QA
writes `private/control-voice-origin.json` and an explicit hold; it does not
assemble an approved review. The Gemini terms URL and producer's
commercial-use claim are not an independently verified, explicit
commercial-use grant. The control gate path is coded to replay the bundle and
hold this rights gap if it reaches
a signed review. The current CLI and inert draft workflow do not supply the
voice bundle or policy.

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
| Separate control voice issuer | Owner-reviewed `VoiceExchangePolicy`, private Ed25519 seed, injected Gemini API key | Pins `refs/tags/voice-v1` and its exact workflow SHA; the key and workflow must be distinct from QA and gate signing. No issuer workflow or live key is configured here. |
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

The dormant control gate replays the new card, tanpura, and font checks from
the signed `origin_proof` and frozen candidate. The studio's current signed
review validator still uses the older object-page HTTP schema for any HTTPS
font origin. It must separately recognize `control_upstream_font_v1` before it
can consume a new font proof. The draft QA workflow fetches full source history
to verify claimed receipt commits as ancestors, and builds Pillow 12.3.0 with
libraqm for Devanagari card comparison. A Linux shadow run still needs to show
that producer and control font rasterization stays inside the pixel bounds;
Pillow's version alone does not pin the system text-shaping and font libraries.
The workflow remains inert and has not reviewed a production candidate.
The voice issuer has no workflow in this tree. Lifting its commercial-rights
hold requires independently verifiable grant evidence and a separate review of
the QA and gate rules before any workflow activation or release.

## Verification

```sh
trusted_qa/.venv/bin/python -m unittest discover -s trusted_qa/tests -v
trusted_qa/.venv/bin/python -m unittest \
  trusted_qa.tests.test_voice_exchange \
  trusted_qa.tests.test_control_voice_observations \
  tests.test_control_voice_gate -v
python3.12 -m compileall -q trusted_qa
```

The tests use synthetic media and mocked provider output plus short real
FFmpeg decodes. Those fixtures check code and gate schema; they are
never production QA evidence.

The focused voice tests are an offline synthetic dry run: they create a temporary
Git source and PCM WAV, replaces `_post_provider`, and checks signing, tamper
rejection, workflow pins, and final-audio matching. It makes no paid provider
call and supplies no commercial-rights evidence.
