# Control-owned analytics and watchdog

[`control-monitor.yml`](../.github/workflows/control-monitor.yml) runs every three
hours on GitHub Actions and can be dispatched manually. It uses only code from
this private control repository. No producer checkout or local laptop is needed.
Its token has read-only repository permissions. The job reads:

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

The run summary and a 14-day JSON artifact hold only destination IDs, public
links, aggregate metrics, status, and bounded operational messages. They contain
no OAuth, Buffer, Cloudinary, or GitHub tokens. Operational alerts fail the job;
Google or public-feed outages are warnings unless both YouTube sources fail.
The workflow makes no mutation requests and publishes nothing.

Required control Actions secrets: `BUFFER_API_KEY`,
`BUFFER_YOUTUBE_CHANNEL_ID`, `BUFFER_INSTAGRAM_CHANNEL_ID`, `CLOUDINARY_URL`,
`YTC_GOOGLE_CLIENT`, and `YTC_GOOGLE_TOKEN`. The nonsecret `BUFFER_ORG_ID` variable
is pinned in code. Google credentials are optional for a degraded public-feed
read, but both missing Google and an unavailable public feed alert. No Modal
credential is mounted: the release workflow and its Modal reads need separate
validation, while this job monitors its GitHub run outcome.

Run the offline tests with:

```sh
python3 -m unittest discover -s trusted_monitor/tests -v
```
