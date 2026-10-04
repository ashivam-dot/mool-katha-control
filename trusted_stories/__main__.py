import json
import os
import sys
from pathlib import Path

import requests

from . import MAX_MEDIA_BYTES, MEDIA_HOST, StoryHold, run_once


def _fetch(url: str) -> bytes:
    if not url.startswith(f"https://{MEDIA_HOST}/"):
        raise StoryHold("Story media is not on the approved host")
    out = bytearray()
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("video/"):
            raise StoryHold("hosted Story media is not a video")
        for chunk in response.iter_content(1 << 20):
            out.extend(chunk)
            if len(out) > MAX_MEDIA_BYTES:
                raise StoryHold("hosted Story media is too large")
    return bytes(out)


def main() -> None:
    from ytc import publish

    trusted = Path(os.environ["TRUSTED_PIPELINE"]).resolve()
    if Path(publish.__file__).resolve() != trusted / "src/ytc/publish.py":
        raise RuntimeError("publisher module is outside the pinned checkout")
    result = run_once(Path(os.environ["SOURCE_CHECKOUT"]).resolve(), Path(os.environ["STORY_STATE"]).resolve(),
                      publish, os.environ["BUFFER_INSTAGRAM_CHANNEL_ID"].strip(), fetch=_fetch)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["decision"] == "held":
        sys.exit(2)


if __name__ == "__main__":
    main()
