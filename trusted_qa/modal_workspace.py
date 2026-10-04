"""Pin every independent QA Modal read to the isolated producer workspace."""

from __future__ import annotations

from .common import QaHold

WORKSPACE = "mool-katha-producer"


def require_modal_workspace(modal_api: object) -> None:
    try:
        name = modal_api.Workspace.from_context().hydrate().name
    except Exception as exc:
        raise QaHold("Modal workspace could not be verified") from exc
    if name != WORKSPACE:
        raise QaHold("Modal token does not belong to mool-katha-producer")
