# Independent signed release migration

The producer repository currently contains a signed-release workflow and other
jobs that receive publisher-capable secrets. `YTC_ENABLE_SIGNED_RELEASE=0` stops
the current release job, but does not isolate those secrets from producer code.
Keep both `YTC_ENABLE_SIGNED_RELEASE` in the producer repository and
`YTC_ENABLE_CONTROL_RELEASE` in this repository disabled during migration.

The proposed [control release workflow](.github/workflows/signed-release.yml)
checks out the mutable producer `main` only as episode data and a receipt
destination. It installs and executes release code from the exact producer
commit `760a740687249b2b6b79fbb073ab2c00b1ce00bf`. Updating that pin
requires a control-repository review. The code at this pin verifies QA signed by
the control repository's `qa-v1` workflow commit
`c71f803b5965df2046ba52c4a39856f4adca64cf` and its embedded public key.
The source checkout is not placed on `PYTHONPATH`, installed, or executed.

## Configuration still required

1. `BUFFER_API_KEY`, `BUFFER_YOUTUBE_CHANNEL_ID`,
   `BUFFER_INSTAGRAM_CHANNEL_ID`, and `CLOUDINARY_URL` were staged as Actions
   secrets in this control repository on 2026-10-04 without displaying values.
   This repository's nonsecret `BUFFER_ORG_ID` is set to the reviewed Mool Katha
   Buffer organization ID `6ac066851cde9b9edca25c7b`.
2. `SOURCE_WRITE_DEPLOY_KEY` now appears among this repository's Actions secret
   names. Verify its public half is installed as a write-enabled deploy key on
   the producer repository, then test checkout and a harmless push before live
   release. The existing `SOURCE_READONLY_DEPLOY_KEY` remains read-only for QA
   and the code checkout.
3. Manually dispatch `signed-release.yml` on `main` with `mode=verify-only`.
   This runs the pinned code checkout, a source write-key dry-run push, the
   locked dependency install, and live exact Buffer destination checks. It
   cannot call release code or push a receipt, even if the release gate is on.
   Inspect that run before setting `YTC_ENABLE_CONTROL_RELEASE=1`. A later
   enabled run with no candidate tests storage access, and an approved signed
   candidate is required for end-to-end release proof.
4. After control release credentials work, remove the producer repository's
   `BUFFER_API_KEY`, `CLOUDINARY_URL`, and Google OAuth publishing secrets. Its
   `watchdog`, `daily-analytics`, and `verify-ep003-publication` workflows also
   receive those secrets today; move or replace their privileged reads before
   removal if those functions must continue. Remove the old signed-release
   workflow or keep it permanently unable to receive publisher credentials.
   Confirm the producer repository has no publisher-capable Actions secrets or
   Modal secret mounts before enabling unattended production. The local
   producer `pipeline/.env` (mode `0600`) also holds publisher key names; move
   those values to control-owned local configuration or remove them before
   executing untrusted producer code on the Mac.
5. Add control-repository scheduled-release run monitoring and inspect the
   first scheduled QA dispatch and release runs. GitHub Actions schedules can
   start late or omit a slot. Existing producer watchdog checks the old
   repository's signed-release workflow, not the proposed control workflow.

Release receipts are pushed to producer `main`. A concurrent producer commit
can make that push fail after Buffer accepts a post. The pinned release code
reconciles exact existing Buffer posts on retry, but a failed push must be
observed and investigated. Avoid concurrent producer and release runs while
validating the migration.
