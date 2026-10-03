# Dormant trusted release prototype

This package lives in the private control repository. It has no workflow,
entry point, keys, destination IDs, enabled schedule, or deployment. Nothing
in this branch can publish without a separately reviewed caller supplying a
control-owned `ReleasePolicy`, `ReleasePlan`, private artifacts, exact Git
checkout, cross-run durable store, Buffer and Cloudinary clients, and
`execute=True`. This branch provides **no production durable store**.
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
workflow are distinct from QA's. This module verifies the attestation; it
does not produce one.

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

## Before any live release

The QA runner currently emits an unsigned review. Implement and shadow-test
the independent QA signer and complete trusted gate, including the new
audio-quality artifact binding. Implement a real `DurableReleaseStore` with
cross-run locking and durable writes; execution refuses a missing store.
Confirm Buffer's `name` field is the canonical username and validate its
YouTube and Instagram post-detail metadata schema against live read-only
responses. Then shadow-test Cloudinary's canonical unversioned URL and Buffer
reconciliation on nonpublishing test data. Use a release credential boundary
inaccessible to the producer repository; the producer's current Modal token
shares a workspace with its deployed app.

Run the focused mocked tests with
`uv run --frozen --python 3.12 python -m unittest discover -s tests -v`.
