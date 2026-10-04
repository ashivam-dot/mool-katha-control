# Mool Katha control

The independent signed-release migration is staged in
[`RELEASE-MIGRATION.md`](RELEASE-MIGRATION.md). Keep its gate disabled until the
publisher credentials and source write key are configured in this repository
and removed from producer-writable jobs.

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
- `control-monitor.yml` runs read-only analytics and privileged destination
  checks from this repository every three hours. It keeps a secret-free report
  artifact and watches this repository's release schedule. See
  [`trusted_monitor/README.md`](trusted_monitor/README.md).
- `trusted_qa/` contains the fail-closed QA runner. The manual `shadow-qa.yml`
  produces unsigned reviews. The separate `release-qa.yml` runs at the pinned
  `qa-v1` tag and can sign approved reviews; the latest observed run found no
  eligible pending candidate and did not enter its fetch, review, or sign jobs.
- `ep001` and `ep002` are legacy pending drafts without a cloud
  `production_agent_id`; strict discovery cannot treat them as eligible QA
  candidates. A new producer-authenticated snapshot is needed for full QA.
- QA signing is configured here. Four publisher credential names have been
  staged as control-repository secrets, while copies still exist in the
  producer repository. Its signed-release gate is disabled. The proposed
  control-side release has a source write key secret name and the reviewed
  Buffer organization variable, but its credentials and release path remain
  untested and its gate is off.

The producer repository's cloud watchdog still handles its unprivileged studio
checks. Control-owned monitoring covers Buffer, Cloudinary, owned YouTube
analytics, and this repository's signed-release runs.

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
post state. The current producer repository still has publisher-capable secrets;
complete the migration before enabling unattended production.
