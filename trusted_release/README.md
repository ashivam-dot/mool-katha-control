# Dormant trusted release prototype

This package lives in the private control repository. It has no workflow,
entry point, keys, destination IDs, enabled schedule, or deployment. Nothing
in this branch can publish without a separately reviewed caller supplying a
control-owned `ReleasePolicy`, `ReleasePlan`, private artifacts, exact Git
checkout, cross-run durable store, Buffer and Cloudinary clients, and
`execute=True`. `DynamoDBReleaseStore` is implemented, but no cloud table or
IAM role has been provisioned or tested against AWS.
`ep003` is explicitly excluded.

## Input contract

- `artifact_dir` is the trusted QA snapshot's episode directory. It contains
  `agent-release-review.json`, `agent-release-signature.json`,
  `agent-audio-quality-observation.json`, and exact copies of the five frozen
  Git files. A separate trusted gate must write
  `release-gate-attestation.json` and `release-gate-signature.json` only after
  it validates complete QA evidence. Neither an unsigned QA review nor an
  approved boolean in that review is release authority.
- `source_repo` is a read-only checkout whose HEAD must equal the plan's exact
  commit. The executor reads only Git blobs for `script.json`, `short.yaml`,
  `qc.json`, `evidence.json`, and `work/manifest.json`. Each blob must equal
  the artifact copy and its hash in the signed review. It never executes
  producer code.
- `archive` is the private producer archive pinned by `ReleasePlan.archive_sha256`.
  It contains `content/episodes/epNNN/epNNN.mp4`. The executor reads only that
  file, hashes its exact bytes, and never imports or executes producer code.
- `ReleasePolicy` pins distinct QA and gate public keys, their immutable
  workflow tags/SHAs, both Buffer destination IDs and exact usernames, and
  the Cloudinary cloud. Remove a revoked key from this reviewed policy.
- `ReleasePlan` is a reviewed control-repository decision for **one** source
  commit, archive, signed review, MP4, IST slot, YouTube description, Instagram
  caption, and YouTube posting metadata. Its digest also binds the policy's
  destination identities and the Instagram Reel/AI metadata. It must not be
  constructed from producer files or model output.

`release_pair` first verifies the detached QA Ed25519 signature over
`b"mool-katha-agent-release-v1\0" + exact_review_bytes`, independent QA run
provenance, the signed voice-quality artifact reference, and each frozen Git
blob. It separately verifies the trusted gate signature over
`b"mool-katha-control-release-gate-v1\0" + exact_attestation_bytes`; that
attestation binds the source commit, archive, review and signature, quality
artifact, MP4, all five frozen file hashes, and gate workflow identity.
The voice-quality artifact records a **model observation**, not human
listening. The executor then verifies the private MP4 hash, exact Buffer
usernames and explicit false readiness flags, hosts the hash-named video
without overwriting, and reconciles all posts on each bound channel. A
returned Buffer post must be seen again with matching channel, status, time,
text, media, and AI/destination metadata.

The gate attestation is a JSON object with exact fields `kind`, `decision`,
`episode_id`, `source_commit`, `archive_sha256`, `review_sha256`,
`review_signature_sha256`, `video_sha256`, `frozen_sha256`,
`quality_observation_sha256`, and `gate_run`. Its `kind` is
`control_release_gate_attestation_v1`; `decision` must be `approved`.
`frozen_sha256` maps the five paths above to raw SHA-256 digests. `gate_run`
names `system`, `repository`, `workflow_ref`, `workflow_sha`, `run_id`, and
`run_attempt`. The detached `release-gate-signature.json` has exact fields
`kind: control_release_gate_signature_v1`, `algorithm: Ed25519`, `key_id`,
`attestation_sha256`, and canonical base64 `signature`. The signer key and
workflow are distinct from QA's. `trusted_release.gate.sign_gate_attestation`
produces the attestation after rechecking the signed QA review, exact source
checkout and archive, candidate boundary, source and rights evidence, ASR
references, visual artifacts, QC dispositions, and voice-quality observation.
It requires object-specific HTTP origin and rights snapshots for ordinary
external media, checks exact Google Fonts TTF/OFL downloads at one pinned
upstream revision, and replays control pixel or PCM checks for newly receipted
designed cards and tanpura beds. Gemini voice receipts still hold because the
provider audio and full response are absent. It matches
each saved numbered frame sheet to the signed all-frame pixel audit. Every
decoded frame must have a clear, low-uncertainty model batch decision. The
batch request and response hashes are signed by QA, but their private provider
traces are not present in this gate artifact; a live shadow run must confirm
the QA archive preserves those traces before the gate is enabled.
No workflow invokes it yet.

The injected `DurableReleaseStore` must hold a lock across independent runner
machines and synchronously fsync or equivalently confirm each journal write.
The executor saves `create_started` with the exact payload hash **before**
calling Buffer. A confirmed response is saved with its post ID before the
next create. Any failed or interrupted request leaves `create_started` or
`unknown_outcome`; a later run refuses to retry even when Buffer's listing is
stale. An unreceipted post, duplicate, missing accepted post, changed media
or metadata, or incomplete listing also raises `ReleaseHold`. Reconciliation
of an unknown outcome requires a separate trusted operator action. No
rollback or automatic retry can silently make a second post.

## DynamoDB durable store

`DynamoDBReleaseStore.from_aws(table_name, namespace)` uses the trusted release
job's AWS identity. The table needs one string partition key named `pk`, on
demand capacity, a single AWS region, and TTL disabled. Restrict the release
role to that table and `GetItem`, `PutItem`, `DeleteItem`, and
`TransactWriteItems`; provision and audit the role separately. The table and
role must be owned by the control account, inaccessible to producer jobs.
The namespace must be a reviewed, stable production identifier. Changing it
changes lock and journal keys after a release attempt.

The store uses a strongly consistent journal read and a DynamoDB transaction
that checks ownership of the episode lock while conditionally writing the next
journal revision. Each lock has a random owner and no TTL or lease expiry. A
runner crash may leave the lock in place. An operator must inspect the exact
journal and Buffer channels before removing that exact lock outside this
module. Automatic stale-lock takeover is absent. A write timeout or
conditional conflict stops release; it never confirms `create_started` on
an uncertain write. `release_pair` reads its write back before a Buffer create.

The QA runner can sign only its own just-assembled review when explicitly
given a 32-byte Ed25519 seed and key ID. Its default remains unsigned.
The gate key must be distinct from the QA key and available only to the gate
job. Both public keys and tagged workflow identities must be pinned in a
reviewed `ReleasePolicy`. The gate compares its supplied run context with the
actual `GITHUB_REPOSITORY`, `GITHUB_WORKFLOW_REF`, `GITHUB_WORKFLOW_SHA`,
`GITHUB_RUN_ID`, and `GITHUB_RUN_ATTEMPT` environment values before signing.

## Before any live release

Provision the DynamoDB table and restricted role, then test the store against
the real table from independent runner machines. Shadow-test the QA signer and
gate as separate tagged control jobs against real private QA output; code
paths are implemented but no workflow, keys, or cloud resources are installed.
The gate job must use the locked NumPy and Pillow versions, with Pillow built
with libraqm for Hindi card reproduction, and a full read-only source history
to verify the receipt's claimed ancestor commits. Shadow-check real cards on
the producer and gate runners before enabling this comparison, since different
system text-shaping or font library builds can move glyph pixels.
Confirm Buffer's `name` field is the canonical username and validate its
YouTube and Instagram post-detail metadata schema against live read-only
responses. Then shadow-test Cloudinary's canonical unversioned URL and Buffer
reconciliation on nonpublishing test data. Use a release credential boundary
inaccessible to the producer repository; the producer's current Modal token
shares a workspace with its deployed app.

Run the focused mocked tests with
`uv run --frozen --python 3.12 python -m unittest discover -s tests -v`.
