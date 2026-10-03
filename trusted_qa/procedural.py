"""Control-owned checks for the two studio generators with no external media inputs.

These routines do not import or execute producer code. The source hashes and
parameters are checked separately before a caller may use their results.
"""

from __future__ import annotations

import io
import random
import re
import textwrap
import wave
from pathlib import Path

from .common import QaHold, require


CARD_WIDTH, CARD_HEIGHT = 1210, 2150
CARD_FONTS = ("RozhaOne-Regular.ttf", "Mukta-ExtraBold.ttf")


def _balanced(value: str, count: int) -> list[str] | None:
    best = None
    for width in range(max(len(value), 1), 0, -1):
        lines = textwrap.wrap(value, width=width, break_long_words=False) or [value]
        if len(lines) > count:
            break
        if len(lines) == count:
            best = lines
    return best


def _reference_card(card: dict, seed: str, color: str, fonts: dict[str, bytes]):
    """Draw fixed card typography over a smooth, noise-free version of its ground."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    width, height = CARD_WIDTH, CARD_HEIGHT
    rgb = tuple(bytes.fromhex(color[1:]))
    ground = Image.new("RGB", (width, height), rgb)
    vignette = Image.new("L", (width, height), 0)
    vdraw = ImageDraw.Draw(vignette)
    rng = random.Random(seed)
    cx, cy = width / 2 + rng.uniform(-40, 40), height * 0.36
    for step in range(40):
        radius = max(width, height) * (1 - step / 40)
        vdraw.ellipse((cx - radius, cy - radius * 1.2,
                       cx + radius, cy + radius * 1.2), fill=int(150 * step / 40))
    vignette = vignette.filter(ImageFilter.GaussianBlur(60))
    ground = Image.composite(ground, Image.new("RGB", (width, height), (6, 9, 12)),
                             vignette.point(lambda value: min(255, value + 105)))
    draw = ImageDraw.Draw(ground)
    ink, accent, muted = (244, 239, 230), (255, 212, 0), (201, 194, 182)

    def font(name: str, size: int):
        return ImageFont.truetype(io.BytesIO(fonts[name]), size)

    def lines_height(lines: list[str], face, spacing: float = 1.08) -> float:
        ascent, descent = face.getmetrics()
        return (ascent + descent) * spacing * len(lines)

    def fit(value: str, name: str, max_width: int, max_size: int,
            max_lines: int, min_size: int = 60, max_height: float | None = None,
            spacing: float = 1.08):
        for size in range(max_size, min_size - 1, -6):
            face = font(name, size)
            for count in range(1, max_lines + 1):
                lines = _balanced(value, count)
                if (lines and all(face.getlength(line) <= max_width for line in lines) and
                        (max_height is None or lines_height(lines, face, spacing) <= max_height)):
                    return face, lines
        face = font(name, min_size)
        return face, (_balanced(value, max_lines) or
                      textwrap.wrap(value, width=max(8, len(value) // max_lines + 1)))[:max_lines]

    def draw_lines(lines: list[str], face, top: float, fill: tuple[int, int, int],
                   spacing: float = 1.08, tracking: int = 0) -> float:
        ascent, descent = face.getmetrics()
        step = (ascent + descent) * spacing
        for line in lines:
            if tracking and line.isascii():
                extent = sum(face.getlength(char) for char in line) + tracking * (len(line) - 1)
                x = (width - extent) / 2
                for char in line:
                    draw.text((x, top), char, font=face, fill=fill)
                    x += face.getlength(char) + tracking
            else:
                draw.text(((width - face.getlength(line)) / 2, top), line, font=face, fill=fill)
            top += step
        return top

    big, small = " ".join(card["big"].split()), " ".join(card["small"].split())

    def first_line(total: float) -> float:
        return min(height * 0.33 - total / 2, height * 0.52 - total)

    if card["kind"] == "dateline":
        match = re.fullmatch(r"(.*?)[\s,]*\b(\d{3,4}(?:\s*(?:BC|BCE|AD|CE))?)", big)
        lead, year = (match.group(1).strip(), match.group(2)) if match else ("", big)
        big_font, big_lines = fit(year, CARD_FONTS[0], 1000, 420, 2)
        lead_font, lead_lines = fit(lead.upper(), CARD_FONTS[1], 960, 78, 1, 44) if lead else (None, [])
        small_font, small_lines = fit(small.upper(), CARD_FONTS[1], 960, 66, 2, 40)
        gap = 70
        total = ((lines_height(lead_lines, lead_font) + 10 if lead else 0) +
                 lines_height(big_lines, big_font, 1.0) +
                 (gap + lines_height(small_lines, small_font) if small else 0))
        top = first_line(total)
        if lead:
            top = draw_lines(lead_lines, lead_font, top, ink, 1.0, tracking=10) + 10
        bottom = draw_lines(big_lines, big_font, top, ink, 1.0)
        if small:
            rule_y = bottom + gap / 2 - 4
            draw.rectangle((width / 2 - 130, rule_y, width / 2 + 130, rule_y + 7), fill=accent)
            draw_lines(small_lines, small_font, bottom + gap, accent, 1.1, tracking=6)
    elif card["kind"] == "quote":
        small_font, small_lines = fit(f"— {small}" if small else "", CARD_FONTS[1], 900, 52, 2, 36)
        mark, mark_height = font(CARD_FONTS[0], 340), 190
        room = height * 0.52 - 260 - mark_height - (90 + lines_height(small_lines, small_font) if small else 0)
        quote_font, quote_lines = fit(big.strip("\"“”'"), CARD_FONTS[0], 960, 132, 6, 56, room, 1.12)
        total = (mark_height + lines_height(quote_lines, quote_font, 1.12) +
                 (90 + lines_height(small_lines, small_font) if small else 0))
        top = first_line(total)
        draw.text(((width - mark.getlength("“")) / 2, top - 70), "“", font=mark, fill=accent)
        bottom = draw_lines(quote_lines, quote_font, top + mark_height, ink, 1.12)
        if small:
            draw_lines(small_lines, small_font, bottom + 90, accent, 1.1)
    else:
        small_font, small_lines = fit(small, CARD_FONTS[1], 940, 60, 3, 40)
        gap = 60
        room = height * 0.52 - 160 - (gap + lines_height(small_lines, small_font, 1.12) if small else 0)
        big_font, big_lines = fit(big.upper(), CARD_FONTS[0], 1000, 300, 3, 90, room, 1.02)
        total = (lines_height(big_lines, big_font, 1.02) +
                 (gap + lines_height(small_lines, small_font) if small else 0))
        top = first_line(total)
        bottom = draw_lines(big_lines, big_font, top, ink, 1.02)
        if small:
            draw.rectangle((width / 2 - 90, bottom + gap / 2 - 4,
                            width / 2 + 90, bottom + gap / 2 + 3), fill=accent)
            draw_lines(small_lines, small_font, bottom + gap, muted, 1.12)
    return ground


def check_card(path: Path, card: dict, seed: str, color: str,
               fonts: dict[str, bytes]) -> dict[str, int]:
    """Reject added imagery or missing typography in an exact designed-card PNG.

    Producer grain is random and cannot be reproduced byte for byte. Every
    background pixel nevertheless stays within a narrow bound around the
    independently rendered smooth ground. Text must occupy its expected shape.
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter, __version__ as pillow_version, features
    except ImportError as exc:
        raise QaHold("card verification needs locked Pillow and NumPy") from exc
    require(isinstance(color, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is not None and
            max(bytes.fromhex(color[1:])) <= 64,
            "designed card uses a color outside the independently checked dark palette")
    require(set(card) == {"kind", "big", "small"} and
            card["kind"] in {"dateline", "fact", "quote"} and
            all(isinstance(card[key], str) for key in ("big", "small")) and
            bool(card["big"].strip()), "designed card parameters are malformed")
    require(set(fonts) == set(CARD_FONTS) and all(fonts.values()),
            "designed card exact font bytes are unavailable")
    require(pillow_version == "12.3.0" and
            (not any("\u0900" <= char <= "\u097f" for char in card["big"] + card["small"])
             or features.check("raqm")),
            "designed card needs the reviewed Pillow build with Devanagari text shaping")
    Image.MAX_IMAGE_PIXELS = 8_000_000
    try:
        with Image.open(path) as opened:
            require(opened.format == "PNG" and opened.mode == "RGB" and
                    opened.size == (CARD_WIDTH, CARD_HEIGHT) and not getattr(opened, "is_animated", False),
                    "designed card is not a full RGB PNG plate")
            actual = np.asarray(opened.copy(), dtype=np.int16)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise QaHold("designed card PNG cannot be decoded") from exc
    try:
        expected = np.asarray(_reference_card(card, seed, color, fonts), dtype=np.int16)
    except (OSError, ValueError) as exc:
        raise QaHold("designed card exact font bytes cannot be rendered") from exc
    bright = (expected.max(axis=2) > 100).astype("uint8") * 255
    bright_image = Image.fromarray(bright, mode="L")
    allowed = np.asarray(bright_image.filter(ImageFilter.MaxFilter(5))) > 0
    core = np.asarray(bright_image.filter(ImageFilter.MinFilter(5))) > 0
    unexpected = int(np.count_nonzero((actual.max(axis=2) > 100) & ~allowed))
    missing = int(np.count_nonzero((actual.max(axis=2) < 100) & core))
    residual = np.max(np.abs(actual - expected), axis=2)
    background_outliers = int(np.count_nonzero((residual > 31) & ~allowed))
    core_outliers = int(np.count_nonzero((residual > 31) & core))
    require(int(core.sum()) >= 1_000 and unexpected <= 100 and missing <= 100 and
            background_outliers == 0 and core_outliers <= 100,
            "designed card pixels differ from independently checked typography and ground")
    return {"width": CARD_WIDTH, "height": CARD_HEIGHT,
            "unexpected_bright_pixels": unexpected, "missing_core_pixels": missing,
            "background_outliers": background_outliers, "core_outliers": core_outliers}


def _pluck(freq: float, rng, *, sample_rate: int, ring_seconds: float):
    import numpy as np

    t = np.arange(int(ring_seconds * sample_rate)) / sample_rate
    clock = np.cumsum(1.0 + 0.0006 * np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 2 * np.pi))) / sample_rate
    rise, top, width = rng.uniform(0.6, 1.3), rng.uniform(14.0, 22.0), rng.uniform(14.0, 24.0)
    wobble = 1.5 * np.sin(2 * np.pi * t / rng.uniform(1.5, 3.0) + rng.uniform(0, 2 * np.pi)) * (1 - np.exp(-t / 0.5))
    centre = 2.0 + top * (1.0 - np.exp(-t / rise)) + wobble
    decay_scale = 4.5 * rng.uniform(0.85, 1.15)
    out = np.zeros_like(t)
    for harmonic in range(1, int(min(48, 9000 / freq)) + 1):
        swell = 0.25 + np.exp(-((harmonic - centre) ** 2) / width)
        decay = np.exp(-t * (1 + 0.035 * harmonic) / decay_scale)
        out += (harmonic**-0.9 * swell * decay *
                np.sin(2 * np.pi * freq * harmonic * clock + rng.uniform(0, 2 * np.pi)))
    thump = rng.standard_normal(len(t)) * np.exp(-t / 0.004) * 0.05
    return (out + thump) * np.minimum(t / 0.006, 1.0)


def _tanpura_samples(seconds: float = 55.0, seed: int = 3,
                     sample_rate: int = 48_000):
    """Independent fixed-code synthesis for the receipt's default no-sample bed."""
    import numpy as np

    sa_hz, cycle, ring, variants, reverb_seconds = 138.59, 3.45, 7.0, 4, 2.2
    strings = ((0.75, 0.8), (1.0, 1.0), (1.0, 0.9), (0.5, 1.1))
    pluck_at = (0.0, 0.75, 1.5, 2.25)
    rng = np.random.default_rng(seed)
    rounds = max(1, round(seconds / cycle))
    total = int(rounds * cycle * sample_rate)
    out = np.zeros(total)
    plucks = [[_pluck(sa_hz * ratio, rng, sample_rate=sample_rate, ring_seconds=ring) * gain
               for _ in range(variants)] for ratio, gain in strings]
    for index in range(rounds):
        for string, at in enumerate(pluck_at):
            start = int((index * cycle + at + rng.normal(0.0, 0.03)) * sample_rate) % total
            pluck = plucks[string][rng.integers(variants)] * 10 ** (rng.normal(0.0, 0.8) / 20)
            head = min(len(pluck), total - start)
            out[start:start + head] += pluck[:head]
            out[:len(pluck) - head] += pluck[head:]
    n = int(reverb_seconds * sample_rate)
    impulse = rng.standard_normal(n) * np.exp(-np.arange(n) / sample_rate * 6.9 / reverb_seconds)
    wet = np.fft.irfft(np.fft.rfft(out) * np.fft.rfft(impulse, total), total)
    out = out + 0.35 * wet * np.abs(out).max() / (np.abs(wet).max() + 1e-9)
    return (out / (np.abs(out).max() + 1e-9) * 0.7).astype(np.float32)


def check_tanpura(path: Path) -> dict[str, float | int]:
    """Compare all PCM samples with control synthesis; unrelated recordings hold."""
    try:
        import numpy as np
    except ImportError as exc:
        raise QaHold("tanpura verification needs locked NumPy") from exc
    require(np.__version__ == "2.5.3", "tanpura verification needs the reviewed NumPy version")
    try:
        with wave.open(str(path), "rb") as stream:
            require(stream.getnchannels() == 1 and stream.getsampwidth() == 2 and
                    stream.getframerate() == 48_000 and stream.getcomptype() == "NONE",
                    "tanpura is not mono 48 kHz PCM16")
            frame_count = stream.getnframes()
            require(frame_count == 2_649_600, "tanpura duration differs from fixed synthesis")
            pcm = np.frombuffer(stream.readframes(frame_count), dtype="<i2").astype(np.float32) / 32768.0
    except (OSError, EOFError, wave.Error) as exc:
        raise QaHold("tanpura WAV cannot be decoded") from exc
    expected = _tanpura_samples()
    require(len(expected) == len(pcm), "tanpura decoded sample count is wrong")
    difference = np.abs(pcm - expected) * 32768.0
    maximum, rms = float(np.max(difference)), float(np.sqrt(np.mean(difference**2)))
    require(maximum <= 8.0 and rms <= 1.2,
            "tanpura samples differ from independent no-sample synthesis")
    return {"sample_rate": 48_000, "channels": 1, "frames": frame_count,
            "max_abs_lsb": round(maximum, 4), "rms_lsb": round(rms, 4)}
