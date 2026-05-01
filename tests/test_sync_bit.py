"""End-to-end tests for sync-bit mode.

When --sync-bit is set (the script's default), the first PD is reserved
as an always-on "video active" indicator and the remaining n-1 PDs
Gray-encode the frame number. These tests verify:
  - the sync PD (index 0 in the calibration JSON) is lit on every video
    frame,
  - the remaining PDs encode gray(frame % 2**(n-1)) by their JSON-list
    order (PD k+1 carries frame bit k),
  - the cycle drops by 2x relative to --no-sync-bit,
  - --no-sync-bit reverts to the n-bit encoding,
  - the n_pds=1 + sync-bit case errors out (nothing left for frame bits).
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
    requires_ffmpeg,
)

SCRIPT = Path(__file__).resolve().parent.parent / "Source" / "add_video_sync_tags.py"
SCREEN_W, SCREEN_H = 2400, 1600
PD_XS = (300, 700, 1100, 1500)  # in JSON order
PD_Y = 100


def _decode_pd_states(vout: Path):
    """Return a list of [is_lit_pd0, is_lit_pd1, ...] per output frame."""
    frames = decode_frames(vout)
    n, oh, ow, _ = frames.shape
    scale = ow / SCREEN_W
    sample_xs = [int(round(x * scale)) for x in PD_XS]
    sample_y = int(round(PD_Y * scale))
    return [
        [int(frames[i, sample_y, x, 0]) > 128 for x in sample_xs]
        for i in range(n)
    ]


@requires_ffmpeg
def test_sync_bit_default_reserves_first_pd(tmp_path):
    cal = make_calibration_json(
        tmp_path / "cal.json", screen_w=SCREEN_W, screen_h=SCREEN_H,
        pd_xs=PD_XS, pd_y=PD_Y,
    )
    vin = make_video(tmp_path / "in.mp4", SCREEN_W, SCREEN_H, duration=20, fps=1)
    vout = tmp_path / "out.mp4"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(vout),
         "--calibration-file", str(cal),  # default: --sync-bit on
         "--leading-guard-frames", "0"],  # disable guards: tests verify
                                          # bit pattern starting at frame 1
        capture_output=True, check=True,
    )
    assert b"sync bit:" in proc.stderr
    assert b"frame bits:" in proc.stderr

    states = _decode_pd_states(vout)
    cycle = 1 << (len(PD_XS) - 1)  # 8 with 4 PDs

    # PD 0 is always lit on every video frame.
    assert all(s[0] for s in states), (
        f"sync PD (index 0) not lit on every frame: "
        f"{[i+1 for i, s in enumerate(states) if not s[0]][:5]}"
    )
    # PDs 1..n-1 encode gray(frame % cycle).
    for i, frame_states in enumerate(states, start=1):
        observed = 0
        for k in range(len(PD_XS) - 1):
            if frame_states[k + 1]:
                observed |= (1 << k)
        expected = int(gray.encode(np.int64(i % cycle)))
        assert observed == expected, (
            f"frame {i}: observed bits {observed:0{len(PD_XS) - 1}b}, "
            f"expected {expected:0{len(PD_XS) - 1}b}"
        )


@requires_ffmpeg
def test_sync_bit_frame_at_cycle_multiple_is_distinguishable_from_off(tmp_path):
    # The whole point of sync-bit: at frame=2**(n-1), Gray-encoded value
    # is 0 (all frame bits dark) but the sync bit must still be lit, so
    # the decoder can tell this apart from "video off" (sync bit dark).
    cal = make_calibration_json(
        tmp_path / "cal.json", screen_w=SCREEN_W, screen_h=SCREEN_H,
        pd_xs=PD_XS, pd_y=PD_Y,
    )
    vin = make_video(tmp_path / "in.mp4", SCREEN_W, SCREEN_H, duration=10, fps=1)
    vout = tmp_path / "out.mp4"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(vout),
         "--calibration-file", str(cal),
         "--leading-guard-frames", "0"],
        capture_output=True, check=True,
    )
    states = _decode_pd_states(vout)
    # cycle = 8 with 4 PDs in sync-bit mode; frame 8 -> gray(0) = all dark.
    frame8 = states[8 - 1]
    assert frame8[0] is True, "frame 8: sync bit must be lit"
    assert frame8[1:] == [False, False, False], (
        "frame 8: all 3 frame bits should be dark (gray(0)=0); "
        f"got {frame8[1:]}"
    )


@requires_ffmpeg
def test_no_sync_bit_uses_all_pds_for_frame(tmp_path):
    cal = make_calibration_json(
        tmp_path / "cal.json", screen_w=SCREEN_W, screen_h=SCREEN_H,
        pd_xs=PD_XS, pd_y=PD_Y,
    )
    vin = make_video(tmp_path / "in.mp4", SCREEN_W, SCREEN_H, duration=20, fps=1)
    vout = tmp_path / "out.mp4"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(vout),
         "--calibration-file", str(cal), "--no-sync-bit",
         "--leading-guard-frames", "0"],
        capture_output=True, check=True,
    )
    states = _decode_pd_states(vout)
    cycle = 1 << len(PD_XS)  # 16 with 4 PDs
    # No sync PD in this mode -- frame 16 should have ALL bits dark.
    frame16 = states[16 - 1]
    assert all(not lit for lit in frame16), (
        f"--no-sync-bit, frame 16 -> gray(0) = all dark, but got {frame16}"
    )
    # And bits encode gray(frame % 16) across all 4 PDs.
    for i, fs in enumerate(states, start=1):
        observed = sum((1 << k) for k, lit in enumerate(fs) if lit)
        expected = int(gray.encode(np.int64(i % cycle)))
        assert observed == expected, (
            f"frame {i}: observed {observed:04b}, expected {expected:04b}"
        )


@requires_ffmpeg
def test_sync_bit_with_one_pd_errors(tmp_path):
    # One PD + sync-bit leaves zero frame bits -- nonsense, must error.
    cal = make_calibration_json(
        tmp_path / "cal.json", screen_w=SCREEN_W, screen_h=SCREEN_H,
        pd_xs=(300,), pd_y=PD_Y,
    )
    vin = make_video(tmp_path / "in.mp4", SCREEN_W, SCREEN_H, duration=2, fps=1)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(tmp_path / "out.mp4"),
         "--calibration-file", str(cal)],  # default --sync-bit
        capture_output=True,
    )
    assert proc.returncode != 0
    assert b"at least 2 photodiodes" in proc.stderr


@requires_ffmpeg
def test_sync_bit_announcement_uses_channel_names(tmp_path):
    # The startup banner names the PD channels rather than generic
    # indices when the calibration JSON carries channel names. Lets
    # decoder authors confirm the assignment at a glance.
    cal = make_calibration_json(
        tmp_path / "cal.json", screen_w=SCREEN_W, screen_h=SCREEN_H,
        pd_xs=PD_XS, pd_y=PD_Y,
    )
    vin = make_video(tmp_path / "in.mp4", SCREEN_W, SCREEN_H, duration=2, fps=1)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(tmp_path / "out.mp4"),
         "--calibration-file", str(cal)],
        capture_output=True, check=True,
    )
    err = proc.stderr.decode()
    assert "sync bit: Dev1/ai0" in err
    assert "bit 0=Dev1/ai1" in err
    assert "bit 1=Dev1/ai2" in err
    assert "bit 2=Dev1/ai3" in err
