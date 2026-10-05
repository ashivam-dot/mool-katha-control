"""Ask Gemini whether it will serve each reviewer schema shape, without sending any episode data."""

from __future__ import annotations

import argparse
import copy
import json
import os
import urllib.error
import urllib.request

from .common import GEMINI_ENDPOINT
from .reviewer import _response_schema


def _inline(schema: dict) -> dict:
    definitions = schema.get("$defs", {})

    def walk(value):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                return walk(copy.deepcopy(definitions[value["$ref"].rsplit("/", 1)[1]]))
            return {k: walk(v) for k, v in value.items() if k != "$defs"}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return walk(schema)


def _arrays(schema: dict) -> dict:
    inline = _inline(schema)
    audio = inline["properties"]["audio_review"]["properties"]
    for parent, name in ((audio, "speech_difference_dispositions"),
                         (inline["properties"], "qc_warning_dispositions")):
        keyed = parent[name]
        if keyed.get("type") == "object" and keyed["properties"]:
            parent[name] = {"type": "array", "items": next(iter(keyed["properties"].values()))}
    return inline


def probe(schema: dict, model: str, key: str) -> tuple[int, str]:
    body = {"contents": [{"role": "user", "parts": [{"text": "Return the smallest JSON value you can."}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "maxOutputTokens": 16, "responseJsonSchema": schema}}
    request = urllib.request.Request(f"{GEMINI_ENDPOINT}/{model}:generateContent",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, ""
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read().decode("utf-8", "replace"))["error"]["message"]
        except Exception:  # noqa: BLE001
            message = "unreadable error body"
        return exc.code, message[:600]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--asr-files", type=int, default=2)
    parser.add_argument("--beats", type=int, default=8)
    parser.add_argument("--differences", type=int, default=16)
    parser.add_argument("--warnings", type=int, default=16)
    args = parser.parse_args()
    packet = {"full_final_audio_asr": [{"file": f"asr-{n}.json"} for n in range(args.asr_files)],
              "script_beats": [{}] * args.beats,
              "qc": {"speech_differences": ["d"] * args.differences, "warnings": ["w"] * args.warnings}}
    current = _response_schema(packet)
    key = os.environ["QA_GEMINI_API_KEY"]
    def counted(exact: bool) -> dict:
        schema = _arrays(current)
        audio = schema["properties"]["audio_review"]["properties"]
        for parent, name, count in ((audio, "speech_difference_dispositions", args.differences),
                                    (schema["properties"], "qc_warning_dispositions", args.warnings)):
            parent[name]["minItems"] = count
            if exact:
                parent[name]["maxItems"] = count
        return schema

    for name, schema in (("current", current), ("arrays_min", counted(False)),
                         ("arrays_exact", counted(True))):
        status, message = probe(schema, args.model, key)
        print(json.dumps({"variant": name, "bytes": len(json.dumps(schema)), "status": status,
                          "message": message}))


if __name__ == "__main__":
    main()
