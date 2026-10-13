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

## Two-week test (14-27 Oct 2026)

Keep the AI look if the median YouTube views per Short reach 150 or more at 48 hours, with Instagram not falling
below its Ramayana baseline of about 3,000. If after 10 AI-art Shorts the YouTube median is still under 30,
rethink the angle before the visuals.
