"""Wrong workspace credentials must never reach the private archive function."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from trusted_qa.archive import fetch_modal_archive
from trusted_qa.common import QaHold


class ArchiveWorkspaceTests(unittest.TestCase):
    def test_old_workspace_rejected_before_function_lookup(self) -> None:
        function = Mock()
        modal = SimpleNamespace(
            Workspace=SimpleNamespace(from_context=lambda: SimpleNamespace(
                hydrate=lambda: SimpleNamespace(name="akshshivam5"))),
            Function=SimpleNamespace(from_name=function),
        )
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"MODAL_TOKEN_ID": "old-id", "MODAL_TOKEN_SECRET": "old-secret"}), \
             patch.dict(sys.modules, {"modal": modal}):
            output = Path(directory) / "archive.tar"
            with self.assertRaisesRegex(QaHold, "does not belong to mool-katha-producer"):
                fetch_modal_archive("ep001", output)
            function.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
