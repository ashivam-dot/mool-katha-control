# Lite lane direction (since 2026-10-13)

`lite-state/LEARNINGS.md` is rewritten by `learn.py` every run; this file holds the editorial direction and the
reasons behind it.

## Why it changed

After two weeks at two Shorts a day the YouTube channel had 2 subscribers and 1-50 views per Short, while the same
videos drew 368-36,523 views as Instagram Reels. Watching the Shorts frame by frame showed museum paintings that
were small, flat and often not of the line being spoken; every script ran the same template (a 'kyun' question,
a literal retelling, then "यही कारण है कि पूछा जाता है..."); titles led with the scripture's name; and the Shorts ran
48-51 seconds. On 2026-10-13 the top Hindi mythology channels measured with yt-dlp won with 60-136 second emotional
stories (devotion, curses, death, "what happened next", "why X has Y"), close-up painterly faces, and emotional
titles with the mystery first.

## What the lane does now

- **Script** (`writer.py`): 130-185 words in 8-12 beats (about 60-80 s). A hook of at most 12 words ('क्यों' or a
  stakes line), an emotional arc, and an explicit payoff from the passage. The ending rotates between loop,
  comment question and lesson (`writer.ending_for`). Titles put the mystery first and the text name last. The old
  closing formula and like/subscribe requests are refused. The accuracy gates are unchanged: exact evidence quotes
  from the fetched page, the source review, repairs, red lines, and the citation on screen and in the description.
- **Pictures** (`look.py`, `imagegen.py`, `aiart.py`): every beat's English `scene` becomes an image prompt with one
  fixed style and one fixed look per character. Pictures are painted by Z-Image-Turbo (Apache-2.0) through
  stable-diffusion.cpp on the GitHub runners' CPUs, in parallel `paint` jobs. A vision model refuses nudity,
  sexual content, gore, disrespect, malformed figures or garbled text, and pictures that don't fit the line. Beats
  whose picture failed repeat the nearest earlier one. Depth-parallax motion (`depth_motion`) moves each picture.
- **Fallback:** when the plan or paint jobs fail, or too few AI pictures pass (less than 60% of beats, or the
  hook's), the same run illustrates the Short with museum paintings as before. Publishing never waits on the image
  model. Set the repository variable `LITE_AI_ART=off`, or dispatch with `art: paintings`, to turn AI art off.
  (Not `0`: Actions compares an unset variable as the number 0, so `!= '0'` would turn AI art off everywhere.)
- **Disclosure:** Shorts with AI pictures set YouTube's `isAiGenerated`, and the description and caption say the
  pictures and voice are AI-made (`publisher.DISCLOSURE_AI_ART`). The ledger entry records `ai_art`.
- **Topics:** 36 Ramayana sargas chosen for emotion and mystery (Jatayu, Sampati, dying Vali, Sita refusing
  Hanuman's ride, the Brahmastra binding, ...) were added ahead of the unused curated topics, each checked against
  the live valmikiramayan.net page.

## Measured cost (2026-10-13)

- One 576x1024 picture at 4 steps takes 11-20 minutes of a free 4-vCPU runner (median about 19; run 38378496851).
  A native `GGML_NATIVE=ON` build was no faster (run 38369839962). FLUX.2-klein-4B was as slow and looked like a
  cartoon, so Z-Image-Turbo stays.
- One Short end to end: plan 2 min, 10 parallel paint jobs 22 min, narration + checks + depth render 12 min, so
  about 37 minutes. The 6.7 GB of weights are cached under `lite-art-zimage-q4k-sd959-v1`.
- Zero spend: public-repo Actions minutes, the free Gemini tier for text, voice and the picture check.

## What the first AI render taught

- The vision check works: it refused a Ravana whose extra heads merged together (beat 4 of the Jatayu replay).
- Animal characters must be described as real animals ("a real giant old vulture bird, not a man"), or the model
  paints a bird-headed man in clothes.
- Expressions must be written into every scene: with only "calm radiant face" in her look, Sita smiled next to
  her abductor. The writer now asks for every person's expression, and Sita's look forbids smiling near Ravana.
- Replays (`.github/replay/*.json`, dispatch input `replay`) re-paint a reviewed script without Gemini text calls,
  for testing the picture path when Gemini is overloaded. They run as dry runs only and never touch the ledger.
- A run takes about 40 minutes and the concurrency group keeps one waiting run: a cron firing while a test run
  waits cancels the test (scheduled runs often start 10-15 minutes late). Dispatch tests at least an hour away
  from 01:30, 07:40, 10:50 and 13:20 UTC.

## Two pictures a beat, three Shorts a day (since 2026-10-27)

The channel was paused, and another agent filled the schedule with ten flat slideshow Shorts. The owner reviewed
them and asked for a remake to the channel's standard. The Shorts were deleted. One picture held for 6-8 seconds was
the main reason a beat felt static, so:

- **Shots** (`shots.py`): the writer gives every beat after the hook a second English `close` scene (a
  close-up of the face, hands or object that carries the line). Beats of 12 or more words are cut at the spoken
  seam nearest the middle (a comma, danda or joiner such as और, लेकिन, तभी). The first half shows the wide `scene`,
  the second half shows the close-up, and the voice runs straight across the cut with no pause. The longest beats
  split first, up to 20 shots. Old scripts without `close` keep one picture a beat.
- The picture check runs on the stronger Flash models first, and refuses non-spouses holding each other (Hanuman
  carrying Sita, Ravana clasping her) and pictures whose main figure isn't one of the people named in the line.
- **Cadence:** three Shorts a day at 12:30, 17:30 and 21:45 IST, booked for the same day only, so a bad batch never
  fills the queue for days. Replacements after the evening slot go to 10:30 the next morning.
- **Topics:** the queue now leads with the highest-curiosity sourced stories (Savitri and Yama, Yudhishthira's dog,
  Parashurama challenging Rama, Kumbhakarna waking, Lanka burning, Brahma's arrow, Hanuman's herbs, ...).
- **Cost:** about 18 pictures a Short means two per paint job, so painting takes about 40 minutes and a run about
  55. That still fits between the crons.

## Test and pre-committed pivot (from 14 Oct 2026)

Judge after 7 days or 10 new-format Shorts, whichever comes first, on the median YouTube views at 48 hours. At 30
or more, keep going and optimise toward 150+ (with Instagram not falling below its Ramayana baseline of about
3,000). Under 30, pivot without asking again to temple mysteries and "why" iconography stories from sourced
Puranas; the voice, look and pipeline stay the same.
