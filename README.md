# Mool Katha control

Private cloud control for independent episode QA and release. The producer lives
in [`ashivam-dot/mool-katha`](https://github.com/ashivam-dot/mool-katha) and can
write drafts, so this repository holds the decision code and later release
credentials. Producer jobs have no write access here.

## Current state

- The production repository has one read-only deploy key for this repository's
  cloud checkout. The private half is stored only as this repository's
  `SOURCE_READONLY_DEPLOY_KEY` Actions secret.
- `source-check.yml` verifies the separate checkout and exact channel identity
  without executing source-repository code. It does not approve or publish a
  video.
- `trusted_qa/` contains an unsigned, fail-closed QA runner prototype and an
  inert workflow draft. Its synthetic tests pass, but independent review found
  source/rights and full-frame coverage gaps that are being repaired before a
  real cloud shadow run. No QA workflow is active.
- QA signing, release credentials, and automatic publishing are **not enabled**.
  A draft, checkout, or unsigned QA result cannot release a post.

The first reviewed pilot, `ep003`, is scheduled independently of this control
repository for 2026-10-03 19:00 IST. Its publication verifier and cloud
watchdog remain in the production repository until the pilot finishes.

## Planned trust boundary

1. Producer: research and render drafts; save an immutable candidate and media
   archive. No Buffer, Cloudinary, or signing key.
2. QA: read one candidate snapshot and the exact archive, independently check
   scripture, asset rights, full video and Hindi speech; sign a review of exact
   bytes only after fixed code validates it. No publisher credentials.
3. Release: verify the QA signature and snapshot, then host and schedule once
   on the bound YouTube and Instagram destinations. No production model key or
   QA private key.

Each cloud job must fail closed on missing evidence, source/rights ambiguity,
speech disagreement, destination mismatch, changed bytes, or uncertain prior
post state. Release enablement needs a separate tested change here; adding a
file to the producer repository cannot enable it.
