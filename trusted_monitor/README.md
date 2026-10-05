# Control-owned analytics and watchdog

[`control-monitor.yml`](../.github/workflows/control-monitor.yml) runs every three
hours on GitHub Actions and can be dispatched manually. It executes only code
from this private control repository. A sparse, read-only producer checkout
supplies signed episode receipts as data; no local laptop is needed. The read
job's token has read-only repository permissions. The job reads:

- The exact Buffer organization and pinned YouTube and Instagram channels, their
  readiness, recent post failures, and 30-day sent-post metric coverage. Instagram
  totals are reported only when every sent post supplies that metric.
- Cloudinary usage from the pinned cloud account.
- The owned YouTube channel's public counters and 28 daily Analytics rows through
  a refreshed Google read token held only in memory. It also attempts the public
  channel Atom feed. If one YouTube source is down, the other remains in the report.
- The control repository's own signed-release workflow. Once
  `YTC_ENABLE_CONTROL_RELEASE=1`, a missing scheduled run for five hours, a
  failed latest scheduled run, or an inactive workflow alerts. While the release
  gate is off, skipped runs are expected.
- The six-hour QA dispatcher and its exact `qa-v17` run. An inactive workflow,
  a missing run for ten hours, or a failed latest completed run alerts. The
  dispatcher is checked separately so a manual QA run cannot hide its failure.
  A newly changed dispatcher waits for its first following cron slot and the
  normal four-hour scheduling slack before a missing slot alerts.
- Every signed episode with a published due time that has passed. It checks the
  saved control readback receipt and independently reads both exact Buffer post
  IDs, destinations, hosted video URLs, sent times, and canonical public links.
  Each missing or mismatched YouTube or Instagram receipt is reported by
  episode. Source files are parsed as bounded JSON and never imported.
- For each new sent pair, it also finds the exact YouTube ID in the pinned
  channel's public Atom feed and fetches the Instagram Reel page to check its
  Open Graph URL names `moolkatha.hindi` and the exact shortcode. A generic
  HTTP 200 sign-in page does not pass. A pair with no public proof after 72
  hours keeps alerting instead of aging out of the feed window.

The run summary and a 14-day JSON artifact hold only destination IDs, public
links, aggregate metrics, status, and bounded operational messages. They contain
no OAuth, Buffer, Cloudinary, or GitHub tokens. Operational alerts fail the job;
Google or public-feed outages are warnings unless both YouTube sources fail.
The read job makes no mutation requests and publishes nothing.

A separate persistence job downloads only that run's report and writes an
append-only snapshot to the private `analytics-data` branch under
`analytics/snapshots/<IST date>/<run ID>-<attempt>.json`. Its allowlist retains
YouTube and Instagram Buffer metric coverage and per-post metrics, owned YouTube
daily Analytics rows, and the public video feed. It drops operational messages,
raw provider data, and credentials. This job has control-repository write access
but no Buffer, Google, Cloudinary, Modal, or producer key. The producer's
learning automation remains disabled; the private snapshots are available for
a reviewed feedback path later.

The same job saves each successful exact public pair once under
`analytics/public-delivery/<episode>.json`, binding both Buffer post IDs, public
links, and due time. Later monitor runs read those private proofs without
requesting old Instagram pages again. A changed receipt or missing proof
alerts and cannot silently replace an archived proof.

[`control-alerts.yml`](../.github/workflows/control-alerts.yml) watches completed
control monitor, pinned QA, dispatcher, and signed-release runs. Its hourly
check also catches missed control, QA, and enabled-release schedules. It sends
one owner phone notification per active incident through the control
`YTC_NTFY_TOPIC` secret and keeps one private control issue as a durable
fallback. A failed phone send leaves the issue pending and is retried; a
recovered workflow closes its issue. It has no producer or publisher key.

[`qa-failure-feedback.yml`](../.github/workflows/qa-failure-feedback.yml)
inspects failed `qa-v17` run artifacts in a separate job. It recognizes only
specific content and evidence holds from the pinned runner. A second job, with
the producer write key but no QA artifact or signing key, checks that the
entire tracked episode tree still matches QA's source commit before committing
`editorial-lock.json`. A changed, released, or already locked draft is left
untouched. Provider, fetch, setup, and unknown failures remain retryable and
show up as failed QA runs in this monitor. The producer classifies editorially
locked drafts outside its active review capacity; both repository changes are
needed for a hold to free a producer slot.

Required control Actions secrets: `SOURCE_READONLY_DEPLOY_KEY`, `BUFFER_API_KEY`,
`BUFFER_YOUTUBE_CHANNEL_ID`, `BUFFER_INSTAGRAM_CHANNEL_ID`, `CLOUDINARY_URL`,
`YTC_GOOGLE_CLIENT`, `YTC_GOOGLE_TOKEN`, and `YTC_NTFY_TOPIC`. The nonsecret `BUFFER_ORG_ID` variable
is pinned in code. Google credentials are optional for a degraded public-feed
read, but both missing Google and an unavailable public feed alert. No Modal
credential is mounted: the release workflow and its Modal reads need separate
validation, while this job monitors its GitHub run outcome.

Run the offline tests with:

```sh
python3 -m unittest discover -s trusted_monitor/tests -v
```
