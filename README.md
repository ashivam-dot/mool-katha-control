# Instagram Story receipts

Written only by `.github/workflows/instagram-stories.yml` on `main`. One file per episode in `stories/`:
`scheduled`, `sent`, `error`, `held` (hosted media did not match the signed hash) or `skipped` (video longer
than a Story allows). A file's presence means that episode never gets a second Story.
