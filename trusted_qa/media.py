"""Independently decode and inspect every final MP4 frame and saved visual sample."""

from __future__ import annotations

import io
import json
import math
import re
import statistics
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (QaHold, digest_bytes, digest_file, json_object, require,
                     write_bytes_new, write_json_new)


FRAME_WIDTH = 120
FRAME_HEIGHT = 214
FRAME_BATCH_SIZE = 36
MAX_REVIEW_BATCHES = 72
MAX_REVIEW_FRAMES = FRAME_BATCH_SIZE * MAX_REVIEW_BATCHES


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


def _frame_timestamps(video: Path, count: int, duration: float) -> list[float]:
    result = _command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                       "-show_entries", "frame=best_effort_timestamp_time", "-of", "json",
                       str(video)], timeout=120)
    require(result.returncode == 0, "frame-indexed probe failed")
    frames = json_object(result.stdout, "frame-indexed probe").get("frames")
    require(isinstance(frames, list) and len(frames) == count,
            "frame-indexed probe does not cover every decoded frame")
    try:
        stamps = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    except (KeyError, TypeError, ValueError) as exc:
        raise QaHold("decoded video has a frame without a measurable timestamp") from exc
    require(all(math.isfinite(value) and 0 <= value <= duration + 0.5 for value in stamps) and
            all(first < second for first, second in zip(stamps, stamps[1:])),
            "decoded frame timestamps are missing or nonmonotonic")
    if len(stamps) > 1:
        intervals = [second - first for first, second in zip(stamps, stamps[1:])]
        median = statistics.median(intervals)
        require(median > 0 and all(gap <= max(0.12, 3 * median) for gap in intervals),
                "decoded frames have a material temporal gap")
    return stamps


def _write_frame_batch(episode_dir: Path, frames: list[tuple[int, float, Any]]) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    columns = 6
    label_height = 20
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * FRAME_WIDTH, rows * (FRAME_HEIGHT + label_height)), "#101820")
    draw = ImageDraw.Draw(sheet)
    for position, (index, seconds, frame) in enumerate(frames):
        x = (position % columns) * FRAME_WIDTH
        y = (position // columns) * (FRAME_HEIGHT + label_height)
        draw.text((x + 3, y + 3), f"f{index:04d} {seconds:.2f}s", fill="white")
        sheet.paste(frame, (x, y + label_height))
    output = io.BytesIO()
    sheet.save(output, format="JPEG", quality=88, optimize=True)
    data = output.getvalue()
    digest = digest_bytes(data)
    name = f"agent-video-frames/{digest}.jpg"
    path = episode_dir / name
    if path.exists():
        require(path.read_bytes() == data, "frame batch SHA-256 collision")
    else:
        write_bytes_new(path, data)
    return {"file": name, "sha256": digest, "start_index": frames[0][0],
            "end_index": frames[-1][0], "first_seconds": frames[0][1],
            "last_seconds": frames[-1][1]}


def audit_every_frame(video: Path, episode_dir: Path, private_audit_dir: Path,
                      decoder: dict[str, Any], video_sha256: str) -> tuple[list[dict[str, Any]], Path, str]:
    """Inspect every decoded pixel/timestamp and create numbered visual review batches."""
    try:
        from PIL import Image, ImageChops, ImageStat
    except ImportError as exc:
        raise QaHold("trusted QA needs Pillow for all-frame inspection") from exc
    private_audit_dir.mkdir(parents=True, exist_ok=True)
    frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
            "-err_detect", "explode", "-i", str(video), "-map", "0:v:0", "-an",
            "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=bicubic,format=rgb24",
            "-vsync", "0", "-f", "rawvideo", "-"]
    with tempfile.TemporaryFile(dir=private_audit_dir) as raw_frames:
        try:
            result = subprocess.run(args, stdout=raw_frames, stderr=subprocess.PIPE,
                                    timeout=240, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QaHold("all-frame pixel decode is unavailable or timed out") from exc
        require(result.returncode == 0, "all-frame pixel decode reported an error")
        total_bytes = raw_frames.tell()
        require(total_bytes > 0 and total_bytes % frame_size == 0,
                "all-frame pixel decode has a truncated frame")
        count = total_bytes // frame_size
        require(1 <= count <= MAX_REVIEW_FRAMES and count == decoder["decoded_frame_count"],
                "all-frame pixel decode count differs from the complete MP4 decode")
        stamps = _frame_timestamps(video, count, decoder["decoded_duration_seconds"])
        raw_frames.seek(0)
        records: list[dict[str, Any]] = []
        batches: list[dict[str, Any]] = []
        pending: list[tuple[int, float, Any]] = []
        previous = None
        before_previous = None
        previous_delta = 0.0
        for index, seconds in enumerate(stamps, 1):
            raw = raw_frames.read(frame_size)
            require(len(raw) == frame_size, "all-frame pixel decode ended early")
            frame = Image.frombytes("RGB", (FRAME_WIDTH, FRAME_HEIGHT), raw)
            gray = frame.convert("L")
            brightness, spread = ImageStat.Stat(gray).mean[0], ImageStat.Stat(gray).stddev[0]
            require(spread >= 3,
                    f"frame {index}: blank or uniform visual anomaly")
            delta = (ImageStat.Stat(ImageChops.difference(previous, gray)).mean[0]
                     if previous is not None else 0.0)
            if before_previous is not None:
                neighbor_delta = ImageStat.Stat(ImageChops.difference(before_previous, gray)).mean[0]
                require(not (previous_delta > 4 and delta > 4 and
                             neighbor_delta < min(previous_delta, delta) * 0.35),
                        f"frame {index - 1}: isolated temporal visual anomaly")
            records.append({"index": index, "seconds": round(seconds, 6),
                            "rgb_sha256": digest_bytes(raw),
                            "mean_luma": round(brightness, 3),
                            "stddev_luma": round(spread, 3),
                            "delta_previous": round(delta, 3)})
            pending.append((index, round(seconds, 6), frame))
            if len(pending) == FRAME_BATCH_SIZE:
                batches.append(_write_frame_batch(episode_dir, pending))
                pending = []
            before_previous, previous = previous, gray
            previous_delta = delta
        if pending:
            batches.append(_write_frame_batch(episode_dir, pending))
    require(digest_file(video) == video_sha256, "video changed during all-frame visual inspection")
    audit = {"kind": "all_frame_pixel_temporal_audit_v1", "input_video_sha256": video_sha256,
             "decoded_frame_count": count, "scaled_width": FRAME_WIDTH,
             "scaled_height": FRAME_HEIGHT, "frames": records, "frame_batches": batches,
             "anomalies": []}
    audit_path = episode_dir / "agent-video-frame-audit.json"
    write_json_new(audit_path, audit)
    write_json_new(private_audit_dir / "visual-frame-audit.json", audit)
    return batches, audit_path, digest_file(audit_path)


@dataclass(frozen=True)
class VisualEvidence:
    contact_sheet_file: str
    contact_sheet_sha256: str
    sample_times_seconds: list[float]
    beat_sample_times: dict[int, float]
    readable_crops: list[dict[str, Any]]
    decoder: dict[str, Any]
    frame_batches: list[dict[str, Any]]
    frame_audit_path: Path
    frame_audit_sha256: str


def make_visual_evidence(video: Path, episode_dir: Path, beats: list[dict],
                         qc_duration: float, video_sha256: str,
                         private_audit_dir: Path) -> VisualEvidence:
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
    batches, audit_path, audit_sha256 = audit_every_frame(
        video, episode_dir, private_audit_dir, decoder, video_sha256)
    require(digest_file(video) == video_sha256, "video changed during visual inspection")
    return VisualEvidence(contact_name, digest_bytes(contact_bytes), times, beat_times,
                          crops, decoder, batches, audit_path, audit_sha256)
