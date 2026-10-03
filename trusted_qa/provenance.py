"""Verify new producer receipts against frozen Git, archived bytes, and control code.

A receipt by itself is a producer statement. Only the independently checked
card pixels, reproduced tanpura samples, or upstream font bytes can clear a
supported asset. Gemini metadata cannot prove the omitted provider audio.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .candidate import Candidate
from .common import (QaHold, digest_bytes, digest_file, expect_sha, json_object,
                     path_under, require)
from .procedural import CARD_FONTS, CARD_HEIGHT, CARD_WIDTH, check_card, check_tanpura


# Reviewed generator versions. A changed source needs a new control implementation
# and a separate review; matching a producer-supplied hash does not authorize it.
PLATES_SHA256 = "1f4eac86a5286efb1161766c46659d8f0725baf8f6e0a6d6355d11690aefb51c"
DRONE_SHA256 = "32035c41b10fe18a9b248ff48a33845fcbcbfd44710764d3c078a9d3af3fbeed"
TTS_SHA256 = "aa4c26380c7ad1116846dfdabbf90acc1376adc6889b9630189b4493c2456985"
PILLOW_VERSION = "12.3.0"
NUMPY_VERSION = "2.5.3"
SOUNDFILE_VERSION = "0.14.0"
MAX_RECEIPT_BYTES = 2_000_000


def _source(candidate: Candidate, value: Any, name: str, *,
            required_commit: bool = True) -> bytes:
    require(isinstance(value, dict) and set(value) == {"file", "sha256", "commit"} and
            value["file"] == name, f"source {name}: receipt names another file")
    expected = expect_sha(value["sha256"], f"source {name} SHA-256")
    claimed = value["commit"]
    require(not required_commit or isinstance(claimed, str),
            f"source {name}: committed generator provenance is missing")
    require(claimed is None or isinstance(claimed, str),
            f"source {name}: Git commit is malformed")
    raw = candidate.committed_blob(name, claimed)
    require(digest_bytes(raw) == expected,
            f"source {name}: producer source hash differs from frozen Git bytes")
    return raw


def _archived_source(candidate: Candidate, value: Any, name: str) -> bytes:
    require(isinstance(value, dict) and set(value) == {"file", "sha256", "commit"} and
            value["file"] == name, f"source {name}: archived input is misnamed")
    expected = expect_sha(value["sha256"], f"source {name} SHA-256")
    require(name in candidate.archive_members,
            f"source {name}: input was not retained in the private archive")
    path = path_under(candidate.root, name, f"source {name}")
    require(digest_file(path) == expected,
            f"source {name}: archived input changed")
    claimed = value["commit"]
    require(claimed is None or isinstance(claimed, str),
            f"source {name}: input commit is malformed")
    if claimed is not None:
        require(candidate.committed_blob(name, claimed) == path.read_bytes(),
                f"source {name}: archived input differs from its claimed commit")
    return path.read_bytes()


def _asset_receipt(candidate: Candidate, asset: dict) -> tuple[dict, str, str]:
    rows = candidate.manifest.get("generated_asset_receipts")
    require(isinstance(rows, list),
            f"asset {asset['id']}: new generated-asset receipts are missing")
    require(all(isinstance(item, dict) and set(item) == {"asset_id", "file", "sha256"}
                and isinstance(item["asset_id"], str) and isinstance(item["file"], str)
                for item in rows), "manifest generated-asset receipt index is malformed")
    ids = [item["asset_id"] for item in rows]
    require(len(ids) == len(set(ids)) and set(ids).issubset(candidate.assets),
            "manifest generated-asset receipt index has duplicate or unknown asset IDs")
    matches = [item for item in rows if item["asset_id"] == asset["id"]]
    require(len(matches) == 1,
            f"asset {asset['id']}: exact new production receipt is absent")
    item = matches[0]
    path = candidate.asset_paths[asset["id"]].with_suffix(".provenance.json")
    relative = path.relative_to(candidate.root).as_posix()
    require(item["file"] == relative and relative in candidate.archive_members,
            f"asset {asset['id']}: receipt is not archived beside the exact asset")
    sha = expect_sha(item["sha256"], f"asset {asset['id']} receipt SHA-256")
    require(path.stat().st_size <= MAX_RECEIPT_BYTES and digest_file(path) == sha,
            f"asset {asset['id']}: archived production receipt differs from manifest")
    record = json_object(path.read_bytes(), f"asset {asset['id']} production receipt")
    # The output basename is the media file, not the sidecar basename.
    require(record.get("schema") == "ytc.generated-asset/v1" and
            record.get("output") == {"file": candidate.asset_paths[asset["id"]].name,
                                      "sha256": asset["sha256"]},
            f"asset {asset['id']}: receipt does not bind exact output bytes")
    return record, relative, sha


def _font_sources(candidate: Candidate) -> dict[str, dict]:
    sources = candidate.manifest.get("font_sources")
    font_ids = candidate.manifest.get("font_asset_ids")
    require(isinstance(sources, list) and bool(sources),
            "manifest exact font and licence sources are missing")
    require(isinstance(font_ids, list) and bool(font_ids) and
            all(isinstance(identity, str) and identity in candidate.assets and
                candidate.assets[identity].get("role") == "font" for identity in font_ids) and
            len(font_ids) == len(set(font_ids)),
            "manifest rendered font IDs are malformed")
    found: dict[str, dict] = {}
    for item in sources:
        require(isinstance(item, dict) and set(item) ==
                {"font", "license", "license_name", "copyright"} and
                isinstance(item["font"], dict),
                "manifest font source record is malformed")
        name = item["font"].get("file")
        require(isinstance(name, str) and name.startswith("pipeline/assets/fonts/") and
                name not in found, "manifest font source path is invalid or repeated")
        _source(candidate, item["font"], name)
        license_name = f"pipeline/assets/fonts/{Path(name).name.split('-')[0]}-OFL.txt"
        raw = _source(candidate, item["license"], license_name)
        try:
            copyright_line = raw.decode("utf-8").splitlines()[0]
        except (UnicodeError, IndexError) as exc:
            raise QaHold(f"font {name}: frozen licence text cannot be read") from exc
        require(item["license_name"] == "SIL Open Font License 1.1" and
                isinstance(item["copyright"], str) and
                item["copyright"] == copyright_line,
                f"font {name}: licence details differ from frozen source")
        found[name] = item
    expected = {candidate.assets[identity]["file"] for identity in font_ids}
    require(set(found) == expected,
            "manifest font source records differ from every rendered font")
    return found


def font_source(candidate: Candidate, asset: dict) -> dict:
    """Return a verified Git-bound font source; upstream rights are checked later."""
    font_ids = candidate.manifest.get("font_asset_ids")
    require(asset.get("role") == "font" and isinstance(font_ids, list) and
            asset["id"] in font_ids,
            f"asset {asset['id']}: font was not recorded as used in the render")
    item = _font_sources(candidate).get(asset["file"])
    require(item is not None and item["font"]["sha256"] == asset["sha256"],
            f"asset {asset['id']}: exact font source differs from used TTF")
    try:
        from PIL import ImageFont
        ImageFont.truetype(str(candidate.asset_paths[asset["id"]]), 16)
    except (ImportError, OSError, ValueError) as exc:
        raise QaHold(f"asset {asset['id']}: exact rendered TTF cannot be decoded") from exc
    return item


def _card_spec(candidate: Candidate, asset: dict) -> tuple[int, bool, dict, str]:
    prefix = f"content/episodes/{candidate.episode_id}/work/assets/beat"
    match = re.fullmatch(re.escape(prefix) + r"([0-9]{2})(_then)?\.png", asset["file"])
    require(match is not None, f"asset {asset['id']}: designed card file is not a beat plate")
    index, second = int(match[1]), bool(match[2])
    require(index < len(candidate.spec["beats"]),
            f"asset {asset['id']}: designed card beat is outside the frozen spec")
    visual = candidate.spec["beats"][index].get("visual")
    rendered = candidate.manifest["beats"][index]
    key = "then_card" if second else "card"
    manifest_id = rendered.get("then_card_asset_id") if second else (
        rendered.get("asset") or {}).get("asset_id")
    require(isinstance(visual, dict) and visual.get("source") == "card" and
            visual.get("reuse") is None and not visual.get("fallbacks") and
            manifest_id == asset["id"],
            f"asset {asset['id']}: card role differs from frozen beat and render")
    raw = visual.get(key)
    require(isinstance(raw, dict) and set(raw).issubset({"kind", "big", "small"}),
            f"asset {asset['id']}: card specification is missing")
    card = {"kind": raw.get("kind", "fact"), "big": raw.get("big"),
            "small": raw.get("small", "")}
    require(all(isinstance(card[name], str) for name in ("big", "small")),
            f"asset {asset['id']}: card text is invalid")
    color = visual.get("color", "#101820")
    require(isinstance(color, str), f"asset {asset['id']}: card color is invalid")
    return index, second, card, color


def _verify_card(candidate: Candidate, asset: dict) -> dict:
    index, second, card, color = _card_spec(candidate, asset)
    expected_origin = f"internal:content/episodes/{candidate.episode_id}/short.yaml"
    require(asset.get("origin") == expected_origin and asset.get("rights_url") == expected_origin and
            asset.get("license") == "Original",
            f"asset {asset['id']}: designed card origin or licence is unsupported")
    record, file, sha = _asset_receipt(candidate, asset)
    require(set(record) == {"schema", "kind", "generator", "inputs", "parameters", "fonts", "output"} and
            record["kind"] == "designed_card",
            f"asset {asset['id']}: receipt is not an exact designed-card record")
    generator = _source(candidate, record["generator"], "pipeline/src/ytc/plates.py")
    require(digest_bytes(generator) == PLATES_SHA256,
            f"asset {asset['id']}: card generator has no reviewed control verifier")
    inputs = record["inputs"]
    require(isinstance(inputs, dict) and set(inputs) == {"short_spec", "beat_selection"},
            f"asset {asset['id']}: card inputs are incomplete")
    spec_file = f"content/episodes/{candidate.episode_id}/short.yaml"
    _source(candidate, inputs["short_spec"], spec_file, required_commit=False)
    require(inputs["short_spec"]["sha256"] == candidate.hashes["spec"],
            f"asset {asset['id']}: card input differs from frozen spec")
    selection_file = f"content/episodes/{candidate.episode_id}/work/assets/beat{index:02d}.json"
    selection = json_object(_archived_source(candidate, inputs["beat_selection"], selection_file),
                            f"asset {asset['id']} beat selection")
    selected = selection.get("visual")
    key = "then_card" if second else "card"
    require(isinstance(selected, dict) and selected.get("source") == "card" and
            selected.get(key) == card and selected.get("color") == color,
            f"asset {asset['id']}: archived beat selection differs from frozen card")
    seed = f"{index}:{'then:' if second else ''}{card['big']}"
    parameters = record["parameters"]
    require(parameters == {"card": card, "seed": seed, "color": color,
                           "plate_size": [CARD_WIDTH, CARD_HEIGHT],
                           "pillow_version": PILLOW_VERSION},
            f"asset {asset['id']}: card parameters differ from frozen inputs")
    rows = record["fonts"]
    require(isinstance(rows, list) and len(rows) == 2,
            f"asset {asset['id']}: card font sources are incomplete")
    manifest_fonts = _font_sources(candidate)
    fonts: dict[str, bytes] = {}
    for entry, name in zip(rows, CARD_FONTS):
        path = f"pipeline/assets/fonts/{name}"
        require(entry == manifest_fonts.get(path),
                f"asset {asset['id']}: card font differs from exact rendered font source")
        fonts[name] = _source(candidate, entry["font"], path)
    metrics = check_card(candidate.asset_paths[asset["id"]], card, seed, color, fonts)
    return {"kind": "control_generated_card_v1", "receipt_file": file,
            "receipt_sha256": sha, "generator_sha256": PLATES_SHA256,
            "method": "control_typography_and_pixel_bounds", "metrics": metrics}


TANPURA_PARAMETERS = {
    "seconds": 55.0, "seed": 3, "sample_rate": 48000, "sa_hz": 138.59,
    "strings": [[0.75, 0.8], [1.0, 1.0], [1.0, 0.9], [0.5, 1.1]],
    "pluck_at": [0.0, 0.75, 1.5, 2.25], "cycle": 3.45, "ring": 7.0,
    "variants": 4, "reverb_seconds": 2.2,
}


def _verify_tanpura(candidate: Candidate, asset: dict) -> dict:
    music = candidate.spec.get("music")
    require(asset.get("file") == f"content/episodes/{candidate.episode_id}/work/sfx/tanpura.wav" and
            isinstance(music, dict) and music.get("path") == "tanpura" and
            candidate.manifest.get("music_asset_id") == asset["id"] and
            asset.get("origin") == "internal:pipeline/src/ytc/drone.py" and
            asset.get("rights_url") == "internal:pipeline/src/ytc/drone.py" and
            asset.get("license") == "Original",
            f"asset {asset['id']}: tanpura source or licence is unsupported")
    record, file, sha = _asset_receipt(candidate, asset)
    require(set(record) == {"schema", "kind", "generator", "parameters", "format",
                            "numpy_version", "soundfile_version", "output"} and
            record["kind"] == "synthesized_tanpura" and
            record["parameters"] == TANPURA_PARAMETERS and
            record["format"] == "WAV PCM_16" and
            record["numpy_version"] == NUMPY_VERSION and
            record["soundfile_version"] == SOUNDFILE_VERSION,
            f"asset {asset['id']}: tanpura receipt uses unreviewed parameters or libraries")
    generator = _source(candidate, record["generator"], "pipeline/src/ytc/drone.py")
    require(digest_bytes(generator) == DRONE_SHA256,
            f"asset {asset['id']}: tanpura generator has no reviewed control verifier")
    metrics = check_tanpura(candidate.asset_paths[asset["id"]])
    return {"kind": "control_generated_tanpura_v1", "receipt_file": file,
            "receipt_sha256": sha, "generator_sha256": DRONE_SHA256,
            "method": "control_full_pcm_reproduction", "metrics": metrics}


def verify_generated_asset(candidate: Candidate, asset: dict) -> dict:
    """Return a gate-replayable proof only after an independent media check."""
    candidate.recheck()
    if asset.get("role") == "visual":
        return _verify_card(candidate, asset)
    if asset.get("role") == "music":
        return _verify_tanpura(candidate, asset)
    raise QaHold(f"asset {asset['id']}: no control verifier exists for this generated role")


_OVERRIDE = re.compile(r"\[([^\]]+)\]\(/[^)]*/\)")
DEFAULT_GEMINI_DIRECTION = (
    "Read the transcript below aloud as a warm, natural storyteller. Speak only the transcript.")


def inspect_voice_take(candidate: Candidate, asset: dict) -> dict:
    """Check the new voice record without elevating its producer metadata to proof."""
    voice = candidate.spec.get("voice")
    require(asset.get("role") == "voice" and
            candidate.manifest.get("narration_asset_id") == asset["id"] and
            isinstance(voice, dict) and voice.get("engine") == "gemini",
            f"asset {asset['id']}: only a recorded Gemini voice take is supported")
    prefix = f"content/episodes/{candidate.episode_id}/work/"
    link = candidate.manifest.get("voice_take")
    require(isinstance(link, dict) and set(link) ==
            {"schema", "record_file", "record_sha256", "take_file", "take_sha256",
             "source_narration_sha256", "final_narration_asset_id",
             "final_narration_sha256"} and link["schema"] == "ytc.voice-take/v1" and
            link["record_file"] == prefix + "take.json" and
            link["take_file"] == prefix + "take.wav" and
            link["final_narration_asset_id"] == asset["id"] and
            link["final_narration_sha256"] == asset["sha256"] and
            asset.get("origin") == f"internal:{prefix}take.json" and
            asset.get("rights_url") == "https://ai.google.dev/gemini-api/terms",
            f"asset {asset['id']}: Gemini voice receipt or origin is incomplete")
    for file_key, hash_key in (("record_file", "record_sha256"),
                               ("take_file", "take_sha256")):
        name = link[file_key]
        expected = expect_sha(link[hash_key], f"Gemini {file_key} SHA-256")
        require(name in candidate.archive_members and
                digest_file(path_under(candidate.root, name, f"Gemini {file_key}")) == expected,
                f"asset {asset['id']}: exact Gemini take record or audio is not archived")
    saved = json_object((candidate.root / link["record_file"]).read_bytes(), "Gemini take record")
    require(set(saved) == {"request", "provenance", "take_sha256", "narration_sha256"} and
            saved["take_sha256"] == link["take_sha256"] and
            saved["narration_sha256"] == link["source_narration_sha256"],
            f"asset {asset['id']}: Gemini take record differs from linked files")
    transform = candidate.manifest.get("narration_transform")
    require((transform is None and saved["narration_sha256"] == asset["sha256"]) or
            (isinstance(transform, dict) and
             transform.get("source_narration_sha256") == saved["narration_sha256"] and
             transform.get("output_narration_sha256") == asset["sha256"]),
            f"asset {asset['id']}: Gemini source and final narration hashes differ")
    transcript = "\n".join(_OVERRIDE.sub(r"\1", beat["text"])
                           for beat in candidate.spec["beats"])
    require(isinstance(voice.get("direction", ""), str) and
            isinstance(voice.get("voice", "af_heart"), str),
            f"asset {asset['id']}: Gemini voice or direction is malformed")
    expected_request = {"script": transcript, "engine": "gemini",
                        "voice": voice.get("voice", "af_heart"),
                        "speed": voice.get("speed", 1.0),
                        "lang_code": voice.get("lang_code", "a"),
                        "model": voice.get("model", ""),
                        "direction": voice.get("direction", "")}
    require(saved["request"] == expected_request,
            f"asset {asset['id']}: Gemini cached request differs from frozen script and voice")
    record = saved["provenance"]
    require(isinstance(record, dict) and record.get("response_capture") ==
            "metadata_only_audio_omitted",
            f"asset {asset['id']}: Gemini provider response capture is not the new schema")
    _source(candidate, record.get("generator"), "pipeline/src/ytc/tts.py")
    require(record["generator"]["sha256"] == TTS_SHA256,
            f"asset {asset['id']}: Gemini generator has no reviewed control verifier")
    model = record.get("requested_model")
    require(isinstance(model, str) and re.fullmatch(r"gemini-[A-Za-z0-9._-]+", model) and
            record.get("voice") == voice.get("voice", "af_heart") and
            record.get("transcript_sha256") == digest_bytes(transcript.encode("utf-8")) and
            record.get("direction_sha256") == digest_bytes(
                (voice.get("direction") or DEFAULT_GEMINI_DIRECTION).encode("utf-8")),
            f"asset {asset['id']}: Gemini model, voice, transcript, or direction changed")
    direction = voice.get("direction") or DEFAULT_GEMINI_DIRECTION
    if model.startswith("gemini-3.8-"):
        expected_body = {"model": model,
                         "input": [{"type": "user_input", "content": [{"type": "text", "text": transcript,
                                   "annotations": [{"type": "speech_metadata", "style": direction}]}]}],
                         "response_format": {"type": "audio", "mime_type": "audio/wav"},
                         "generation_config": {"speech_config": [{"voice": record["voice"]}]},
                         "store": False}
        expected_endpoint = "https://generativelanguage.googleapis.com/v1beta/interactions"
        api = "interactions"
    else:
        expected_body = {"contents": [{"parts": [{"text":
                          f"{direction}\n\nTRANSCRIPT:\n{transcript}"}]}],
                         "generationConfig": {"responseModalities": ["AUDIO"],
                                              "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig":
                                                             {"voiceName": record["voice"]}}}}}
        expected_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        api = "generateContent"
    canonical_body = json.dumps(expected_body, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":")).encode("utf-8")
    require(record.get("api") == api and record.get("endpoint") == expected_endpoint and
            record.get("request_body") == expected_body and
            record.get("request_body_sha256") == digest_bytes(canonical_body) and
            isinstance(record.get("response_metadata"), dict) and
            expect_sha(record.get("response_sha256"), "Gemini response hash") and
            expect_sha(record.get("returned_audio_sha256"), "Gemini returned audio hash") and
            record.get("terms") == {"url": "https://ai.google.dev/gemini-api/terms",
                                    "last_updated": "2026-04-28",
                                    "service": "Gemini Developer API (AI Studio key)"},
            f"asset {asset['id']}: Gemini request body or response metadata is inconsistent")
    metadata = record["response_metadata"]
    if api == "interactions":
        require(set(metadata).issubset({"status", "model", "id", "created", "usage"}) and
                metadata.get("status") == record.get("response_status") == "completed" and
                metadata.get("model") == record.get("served_model") and
                metadata.get("id") == record.get("response_id") and
                metadata.get("usage") == record.get("usage") and
                isinstance(metadata.get("usage"), dict) and bool(metadata["usage"]),
                f"asset {asset['id']}: Gemini interaction metadata contradicts itself")
    else:
        require(set(metadata) == {"modelVersion", "responseId", "usageMetadata", "finish_reason"} and
                metadata.get("modelVersion") == record.get("served_model") and
                metadata.get("responseId") == record.get("response_id") and
                metadata.get("usageMetadata") == record.get("usage") and
                metadata.get("finish_reason") == record.get("response_status") == "STOP" and
                isinstance(metadata.get("usageMetadata"), dict) and bool(metadata["usageMetadata"]),
                f"asset {asset['id']}: Gemini generateContent metadata contradicts itself")
    return {"kind": "producer_gemini_receipt_checked_but_unproved_v1",
            "record_sha256": link["record_sha256"], "take_sha256": link["take_sha256"],
            "request_body_sha256": record["request_body_sha256"],
            "response_sha256_claimed": record["response_sha256"],
            "returned_audio_sha256_claimed": record["returned_audio_sha256"]}
