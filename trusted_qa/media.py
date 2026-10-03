"""Independently decode the whole final MP4 and save hash-bound visual samples."""

from __future__ import annotations

import io
import json
import math
import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import QaHold, digest_bytes, digest_file, json_object, require, write_bytes_new


def _command(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QaHold(f"media tool {args[0]} is unavailable or timed out") from exc


def _tool_version(name: str) -> str:
    result = _command([name, "-version"], timeout=10)
    require(result.returncode == 0, f"{name} cannot report its version")
    return result.stdout.decode("utf-8", errors="replace").splitlines()[0][:180]


def probe_final_mp4(video: Path, qc_duration: float) -> dict[str, Any]:
    result = _command(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                       "-of", "json", str(video)], timeout=30)
    require(result.returncode == 0, "ffprobe failed on final MP4")
    info = json_object(result.stdout, "ffprobe result")
    streams = info.get("streams")
    require(isinstance(streams, list) and len(streams) == 2, "final MP4 must have one video and one audio stream")
    video_stream = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_stream = [stream for stream in streams if stream.get("codec_type") == "audio"]
    require(len(video_stream) == len(audio_stream) == 1, "final MP4 stream types differ from QC")
    picture = video_stream[0]
    sound = audio_stream[0]
    require(picture.get("codec_name") == "h264" and picture.get("width") == 1080 and
            picture.get("height") == 1920 and picture.get("pix_fmt") == "yuv420p" and
            sound.get("codec_name") == "aac", "final MP4 stream codecs or dimensions differ from QC")
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QaHold("final MP4 has no measured duration") from exc
    require(math.isfinite(duration) and abs(duration - qc_duration) <= 0.5,
            "final MP4 duration differs from QC")
    return {"duration_seconds": duration, "streams": streams,
            "ffprobe_version": _tool_version("ffprobe")}


def extract_full_final_audio(video: Path, target: Path, qc_duration: float) -> float:
    """Decode the entire mixed MP4 audio once, with no trimming or VAD."""
    require(not target.exists(), "full-audio extraction target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = _command(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                       "-i", str(video), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                       "-c:a", "pcm_s16le", "-f", "wav", str(target)], timeout=180)
    require(result.returncode == 0 and target.is_file(), "full final MP4 audio extraction failed")
    try:
        with wave.open(str(target), "rb") as source:
            require(source.getnchannels() == 1 and source.getframerate() == 16000 and
                    source.getsampwidth() == 2, "extracted final audio format is unexpected")
            duration = source.getnframes() / source.getframerate()
    except (OSError, wave.Error) as exc:
        raise QaHold("extracted final audio is unreadable") from exc
    require(duration > 0 and abs(duration - qc_duration) <= 0.5,
            "extracted whole final audio differs from QC duration")
    return duration


def decode_complete_mp4(video: Path, qc_duration: float) -> dict[str, Any]:
    """Force FFmpeg to decode every encoded video frame and the whole audio stream."""
    info = probe_final_mp4(video, qc_duration)
    result = _command(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                       "-progress", "pipe:1", "-nostats", "-xerror", "-err_detect", "explode",
                       "-i", str(video), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                      timeout=240)
    require(result.returncode == 0, "full MP4 decode reported an error")
    counts = [int(match) for match in re.findall(rb"(?:^|\n)frame=(\d+)", result.stdout)]
    require(bool(counts) and counts[-1] > 0, "full MP4 decode did not report positive frame count")
    return {"decode_tool": _tool_version("ffmpeg"), "decode_exit_code": 0,
            "decoded_frame_count": counts[-1],
            "decoded_duration_seconds": round(info["duration_seconds"], 3),
            "decoded_full_video": True}


def sample_times(manifest_beats: list[dict], duration: float) -> tuple[list[float], dict[int, float]]:
    require(isinstance(manifest_beats, list) and bool(manifest_beats), "render has no beats for visual sampling")
    times = {round(0.1, 3), round(duration - 0.1, 3)}
    beat_times: dict[int, float] = {}
    for index, beat in enumerate(manifest_beats, 1):
        start, end = beat.get("start"), beat.get("end")
        require(isinstance(start, (int, float)) and isinstance(end, (int, float)) and
                not isinstance(start, bool) and not isinstance(end, bool) and
                math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.5,
                f"render beat {index} has an invalid time span")
        midpoint = round((start + end) / 2, 3)
        beat_times[index] = midpoint
        times.add(midpoint)
    cursor = 8.0
    while cursor < duration:
        times.add(round(cursor, 3))
        cursor += 8.0
    ordered = sorted(times)
    require(len(ordered) >= len(manifest_beats) + 2 and ordered[0] <= 1 and
            ordered[-1] >= duration - 1 and
            all(0 <= first < second <= duration + 0.5 and second - first <= 10
                for first, second in zip(ordered, ordered[1:])),
            "systematic frame samples do not cover opening, every beat, and ending")
    return ordered, beat_times


def _frame_at(video: Path, seconds: float):
    try:
        from PIL import Image
    except ImportError as exc:
        raise QaHold("trusted QA needs Pillow for contact-sheet inspection") from exc
    result = _command(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                       "-i", str(video), "-ss", f"{seconds:.3f}", "-frames:v", "1",
                       "-f", "image2pipe", "-vcodec", "png", "-"], timeout=30)
    require(result.returncode == 0 and result.stdout, f"frame sample at {seconds:.3f}s failed")
    try:
        frame = Image.open(io.BytesIO(result.stdout))
        frame.load()
    except (OSError, ValueError) as exc:
        raise QaHold(f"frame sample at {seconds:.3f}s is unreadable") from exc
    require(frame.size == (1080, 1920), "sampled frame dimensions differ from final MP4 probe")
    return frame.convert("RGB")


@dataclass(frozen=True)
class VisualEvidence:
    contact_sheet_file: str
    contact_sheet_sha256: str
    sample_times_seconds: list[float]
    beat_sample_times: dict[int, float]
    readable_crops: list[dict[str, Any]]
    decoder: dict[str, Any]


def make_visual_evidence(video: Path, episode_dir: Path, beats: list[dict],
                         qc_duration: float, video_sha256: str) -> VisualEvidence:
    """Create a contact sheet plus full-size source/caption crops from actual frames."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise QaHold("trusted QA needs Pillow for contact-sheet inspection") from exc
    require(digest_file(video) == video_sha256, "video changed before visual inspection")
    decoder = decode_complete_mp4(video, qc_duration)
    duration = decoder["decoded_duration_seconds"]
    times, beat_times = sample_times(beats, duration)
    frames = {time: _frame_at(video, time) for time in times}
    tile_width, tile_height, label_height, columns = 360, 640, 30, 4
    rows = (len(times) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, (tile_height + label_height) * rows), "#101820")
    draw = ImageDraw.Draw(sheet)
    for index, seconds in enumerate(times):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        tile = frames[seconds].resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y + label_height))
        beat_labels = ",".join(str(number) for number, moment in beat_times.items() if moment == seconds)
        label = f"{seconds:.3f}s" + (f" beat {beat_labels}" if beat_labels else "")
        draw.text((x + 8, y + 8), label, fill="white")
    output = io.BytesIO()
    sheet.save(output, format="JPEG", quality=88, optimize=True)
    contact_bytes = output.getvalue()
    contact_name = "agent-video-contact.jpg"
    write_bytes_new(episode_dir / contact_name, contact_bytes)
    crops: list[dict[str, Any]] = []

    def save_crop(frame, seconds: float, region: str, box: tuple[int, int, int, int]) -> None:
        crop = frame.crop(box)
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
        digest = digest_bytes(data)
        name = f"agent-video-crops/{digest}.png"
        path = episode_dir / name
        if path.exists():
            require(path.read_bytes() == data, "visual crop SHA-256 collision")
        else:
            write_bytes_new(path, data)
        crops.append({"file": name, "sha256": digest,
                      "source_sample_seconds": seconds, "region": region})

    opening = frames[times[0]]
    save_crop(opening, times[0], "opening_source_top", (0, 0, 1080, 900))
    save_crop(opening, times[0], "opening_source_bottom", (0, 800, 1080, 1920))
    for number, seconds in beat_times.items():
        save_crop(frames[seconds], seconds, f"beat_{number}_caption_and_label", (0, 820, 1080, 1920))
    require(digest_file(video) == video_sha256, "video changed during visual inspection")
    return VisualEvidence(contact_name, digest_bytes(contact_bytes), times, beat_times,
                          crops, decoder)
