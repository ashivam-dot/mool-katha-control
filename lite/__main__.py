"""python -m lite run --mode dry-run|publish --out DIR --producer-root PATH | status | verify"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="lite")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=("dry-run", "publish"), default="dry-run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--producer-root", type=Path, required=True)
    sub.add_parser("status")
    sub.add_parser("verify")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.command == "status":
        from .run import load_ledger

        ledger = load_ledger()
        print(json.dumps({"published": len(ledger["episodes"]), "pending": bool(ledger.get("pending")),
                          "ready": bool(ledger.get("ready")), "skipped": ledger.get("skipped")},
                         ensure_ascii=False, indent=1))
        return 0
    if args.command == "verify":
        from . import publisher

        youtube, instagram = publisher.destinations()
        print("Both Mool Katha Buffer destinations are ready")
        return 0

    from .run import run as run_once

    summary = run_once(args.mode, args.out.resolve(), args.producer_root.resolve())
    text = json.dumps(summary, ensure_ascii=False, indent=1, default=str)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(text + "\n", encoding="utf-8")
    if step := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step, "a", encoding="utf-8") as fh:
            fh.write(f"## Mool Katha lite: {summary.get('status')}\n\n```json\n{text}\n```\n")
    return 0 if summary.get("status") in ("rendered", "scheduled", "no_free_slot", "waiting_for_quota",
                                          "ready_waiting_for_slot") else 1


if __name__ == "__main__":
    sys.exit(main())
