"""Read-only Modal draft archive adapter for a future isolated QA job."""

from __future__ import annotations

import os
from pathlib import Path

from .candidate import MAX_ARCHIVE_BYTES
from .common import QaHold, EPISODE, require, write_bytes_new


MODAL_APP = "mool-katha"
MODAL_READ_FUNCTION = "draft_archive"


def fetch_modal_archive(episode_id: str, output: Path) -> Path:
    """Invoke only the existing archive-read function; never import producer code.

    Modal credential scope still needs a separate owner-controlled read identity
    before this can be used as an independent production trust boundary.
    """
    require(EPISODE.fullmatch(episode_id) is not None, "Modal archive needs an epNNN ID")
    require(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"),
            "read-only Modal archive credentials are unavailable")
    require(not output.exists(), "Modal archive output must be new")
    try:
        import modal
    except ImportError as exc:
        raise QaHold("Modal client is unavailable to fetch the private draft archive") from exc
    try:
        function = modal.Function.from_name(MODAL_APP, MODAL_READ_FUNCTION)
        data = function.remote(episode_id)
    except Exception as exc:
        raise QaHold("private Modal draft archive read failed") from exc
    require(isinstance(data, bytes) and 0 < len(data) <= MAX_ARCHIVE_BYTES,
            "Modal draft archive is missing or oversized")
    write_bytes_new(output, data)
    return output
