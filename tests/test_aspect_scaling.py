"""End-to-end tests for screen-aspect-aware tag placement.

Each test runs add_video_sync_tags on a synthetic input of a specific
size against a fixed-screen calibration JSON, then verifies:
  - the output frame is sized to preserve the screen aspect ratio with
    minimal padding around the input,
  - the encoded bit pattern at each scaled tag position decodes to the
    expected frame number for every frame.

These exercise the real ffmpeg pipeline, so they are skipped when ffmpeg
isn't on PATH.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from calibration import gray
from conftest import (
    decode_frames,
    make_calibration_json,
    make_video,
    probe_video_size,
    requires_ffmpeg,
)

SCRIPT = Path(__file__).resolve().parent.parent / "Source" / "add_video_sync_tags.py"
SCREEN_W, SCREEN_H = 2400, 1600
PD_XS = (300, 700, 1100, 1500)
PD_Y = 100
BIT_R = 20


def _tag(vin: Path, vout: Path, cal: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(vout),
         "--calibration-file", str(cal)],
        capture_output=True, check=True,
    )


def _verify_bits(vout: Path, n_bits: int):
    """Decode every output frame, sample each scaled tag position, and
    verify the assembled bit value matches gray(frame_number % cycle).

    The script writes a 4-bit Gray-coded frame number across the 4 PDs;
    bit k is the k-th PD's circle being lit (white) or dark.
    """
    frames = decode_frames(vout)
    n, oh, ow, _ = frames.shape
    cycle = 1 << n_bits
    scale = ow / SCREEN_W
    sampled_xs = [int(round(x * scale)) for x in PD_XS]
    sampled_y = int(round(PD_Y * scale))
    for i in range(1, n + 1):
        observed = 0
        for k, ox in enumerate(sampled_xs):
            v = int(frames[i - 1, sampled_y, ox, 0])
            if v > 128:
                observed |= (1 << k)
        expected = int(gray.encode(np.int64(i % cycle)))
        assert observed == expected, (
            f"frame {i}: observed bits {observed:0{n_bits}b}, "
            f"expected {expected:0{n_bits}b}"
        )


@requires_ffmpeg
@pytest.mark.parametrize(
    ("in_w", "in_h", "out_w", "out_h"),
    [
        (2400, 1600, 2400, 1600),  # exact-match: no pad, scale=1
        (1200,  800, 1200,  800),  # already 3:2 but smaller: no pad, scale=0.5
        (1920, 1080, 1920, 1280),  # 16:9 input on 3:2 screen: pad height
        (1200,  900, 1350,  900),  # 4:3 input on 3:2 screen: pad width
        ( 600,  400,  600,  400),  # very small 3:2: scale=0.25, tiny tags
    ],
    ids=["exact-match", "half-3:2", "16:9-pad-height", "4:3-pad-width", "tiny-3:2"],
)
def test_output_size_and_bit_pattern(tmp_path, in_w, in_h, out_w, out_h):
    cal = make_calibration_json(
        tmp_path / "cal.json",
        screen_w=SCREEN_W, screen_h=SCREEN_H,
        pd_xs=PD_XS, pd_y=PD_Y, bit_radius=BIT_R,
    )
    # 20 frames at 1 fps so wrap (cycle=16) is exercised within a tiny clip.
    vin = make_video(tmp_path / "in.mp4", in_w, in_h, duration=20, fps=1)
    vout = tmp_path / "out.mp4"
    _tag(vin, vout, cal)
    assert probe_video_size(vout) == (out_w, out_h)
    _verify_bits(vout, n_bits=len(PD_XS))


@requires_ffmpeg
def test_input_larger_than_screen_errors(tmp_path):
    cal = make_calibration_json(
        tmp_path / "cal.json", screen_w=SCREEN_W, screen_h=SCREEN_H,
    )
    vin = make_video(tmp_path / "in.mp4", SCREEN_W + 100, SCREEN_H, duration=1)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(tmp_path / "out.mp4"),
         "--calibration-file", str(cal)],
        capture_output=True,
    )
    assert proc.returncode != 0
    assert b"larger than the calibrated screen" in proc.stderr


@requires_ffmpeg
def test_tiny_radius_warning(tmp_path):
    # 300x200 on 2400x1600 -> scale 0.125 -> radius 20*0.125 = 2.5 -> 2 px
    cal = make_calibration_json(
        tmp_path / "cal.json", screen_w=SCREEN_W, screen_h=SCREEN_H,
    )
    vin = make_video(tmp_path / "in.mp4", 300, 200, duration=1)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(tmp_path / "out.mp4"),
         "--calibration-file", str(cal)],
        capture_output=True,
    )
    assert proc.returncode == 0
    assert b"scaled bit radius" in proc.stderr
    assert b"may be too small" in proc.stderr


@requires_ffmpeg
def test_offscreen_tag_warning(tmp_path):
    # Place a tag at y=2000 on a 1600-tall screen -> falls outside the
    # output frame.
    import json
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({
        "monitor": {"width": SCREEN_W, "height": SCREEN_H},
        "photodiodes": [
            {"channel": "Dev1/ai0", "x_px": 300.0, "y_px": 100.0,
             "bit_radius_px": 20, "background_radius_px": 35},
            {"channel": "Dev1/ai1", "x_px": 700.0, "y_px": 2000.0,
             "bit_radius_px": 20, "background_radius_px": 35},
        ],
    }))
    vin = make_video(tmp_path / "in.mp4", 1500, 1000, duration=1)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(tmp_path / "out.mp4"),
         "--calibration-file", str(p)],
        capture_output=True,
    )
    assert proc.returncode == 0
    assert b"fall entirely outside" in proc.stderr


@requires_ffmpeg
def test_manual_mode_requires_screen_size(tmp_path):
    vin = make_video(tmp_path / "in.mp4", 800, 600, duration=1)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(tmp_path / "out.mp4"),
         "--bit-xs", "100,200,300,400", "--bit-ys", "50",
         "--bit-radius", "10", "--background-radius", "20"],
        capture_output=True,
    )
    assert proc.returncode != 0
    assert b"screen_size" in proc.stderr


@requires_ffmpeg
def test_manual_mode_with_screen_size_works(tmp_path):
    vin = make_video(tmp_path / "in.mp4", 1920, 1080, duration=1)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(tmp_path / "out.mp4"),
         "--bit-xs", "300,700,1100,1500", "--bit-ys", "100",
         "--bit-radius", "20", "--background-radius", "35",
         "--screen-size", "2400x1600"],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode()
