# Dormant trusted release prototype

This package lives in the private control repository. It has no workflow,
entry point, keys, destination IDs, enabled schedule, or deployment. Nothing
in this branch can publish without a separately reviewed caller supplying a
control-owned `ReleasePolicy`, `ReleasePlan`, private artifacts, Buffer and
Cloudinary clients, and `execute=True`. `ep003` is explicitly excluded.

## Input contract

- `artifact_dir` is the trusted QA snapshot's episode directory. It contains
  `agent-release-review.json`, `agent-release-signature.json`, and
  `agent-audio-quality-observation.json`. The QA runner writes the first and
  last files; a **separate trusted signer** must write the detached signature
  only after the complete release gate passes.
- `archive` is the private producer archive pinned by `ReleasePlan.archive_sha256`.
  It contains `content/episodes/epNNN/epNNN.mp4`. The executor reads only that
  file, hashes its exact bytes, and never imports or executes producer code.
- `ReleasePolicy` pins the trusted QA public key and immutable workflow tag/SHA,
  both Buffer destination IDs and handles, and the Cloudinary cloud. Remove a
  revoked key from this reviewed policy.
- `ReleasePlan` is a reviewed control-repository decision for **one** source
  commit, archive, signed review, MP4, IST slot, YouTube description, Instagram
  caption, and YouTube posting metadata. Its digest also binds the policy's
  destination identities and the Instagram Reel/AI metadata. It must not be
  constructed from producer files or model output.

`release_pair` first verifies the detached Ed25519 signature over
`b"mool-katha-agent-release-v1\0" + exact_review_bytes`, independent QA run
provenance, the signed voice-quality artifact reference, and the private MP4
hash. The voice-quality artifact records a **model observation**, not human
listening. It then verifies channel identity and readiness, hosts the exact
hash-named video without overwriting, and reconciles all posts on each bound
channel before creating the pair. A returned Buffer post must be seen again
with matching channel, status, time, text, media, and AI/destination metadata.

The receipt records the full posting-intent digest and each confirmed post ID
before the next create. Repeated calls with that durable receipt adopt exact
posts and create only a missing companion. An unreceipted post, duplicate,
missing accepted post, changed media or metadata, lost create response, or
incomplete API listing raises `ReleaseHold`; the caller must reconcile it
manually. No rollback or automatic retry can silently make a second post.

## Before any live release

The QA runner currently emits an unsigned review. The signer and its complete
evidence gate, including the new audio-quality artifact binding, must be
implemented and shadow-tested in trusted control code. The future control
workflow must supply a protected durable receipt store and cross-run lock;
the local file and `flock` here only protect one shared filesystem. Confirm
Buffer's YouTube and Instagram post-detail metadata schema against live
read-only responses, then shadow-test Cloudinary's canonical unversioned URL
and Buffer reconciliation on nonpublishing test data. Use a release credential
boundary inaccessible to the producer repository; the producer's current
Modal token shares a workspace with its deployed app, so a control Modal
secret in that workspace is insufficient.

Run the focused mocked tests with `python3 -m unittest discover -s tests -v`.
