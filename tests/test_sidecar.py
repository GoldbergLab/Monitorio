"""Tests for the tagger's <video>.tags.json sidecar.

The sidecar records tagging-time decisions (sync_bit on/off, channel-to-
bit assignment, fps, frame count, screen size, calibration source) so
the decoder can recover them without the user having to re-pass them.
The decoder reads it; humans can inspect it directly. These tests check
the sidecar's contents in both sync and no-sync modes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_calibration_json, make_video, requires_ffmpeg

SCRIPT = Path(__file__).resolve().parent.parent / "Source" / "add_video_sync_tags.py"
SCREEN_W, SCREEN_H = 2400, 1600


def _tag(tmp_path: Path, *, sync_bit: bool) -> tuple[Path, Path]:
    cal = make_calibration_json(
        tmp_path / "cal.json",
        screen_w=SCREEN_W, screen_h=SCREEN_H,
        pd_xs=(300, 700, 1100, 1500), pd_y=100,
    )
    vin = make_video(tmp_path / "in.mp4", SCREEN_W, SCREEN_H, duration=1, fps=30)
    vout = tmp_path / "tagged.mp4"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(vin), str(vout),
         "--calibration-file", str(cal),
         "--sync-bit" if sync_bit else "--no-sync-bit"],
        check=True, capture_output=True,
    )
    return vout, cal


@requires_ffmpeg
def test_sidecar_written_alongside_output(tmp_path):
    vout, _ = _tag(tmp_path, sync_bit=True)
    sidecar = vout.with_suffix(vout.suffix + ".tags.json")
    assert sidecar.exists()


@requires_ffmpeg
def test_sidecar_records_sync_bit_assignment(tmp_path):
    vout, cal = _tag(tmp_path, sync_bit=True)
    sidecar = vout.with_suffix(vout.suffix + ".tags.json")
    payload = json.loads(sidecar.read_text())
    assert payload["sync_bit"] is True
    assert payload["n_pds"] == 4
    assert payload["n_frame_bits"] == 3
    assert payload["cycle"] == 8
    assert payload["channel_assignment"]["sync"] == "Dev1/ai0"
    assert payload["channel_assignment"]["frame_bits"] == [
        "Dev1/ai1", "Dev1/ai2", "Dev1/ai3",
    ]
    assert payload["fps"] == 30.0
    assert payload["screen_size"] == [SCREEN_W, SCREEN_H]
    assert payload["calibration_file"] == str(cal)


@requires_ffmpeg
def test_sidecar_records_no_sync_bit_assignment(tmp_path):
    vout, _ = _tag(tmp_path, sync_bit=False)
    sidecar = vout.with_suffix(vout.suffix + ".tags.json")
    payload = json.loads(sidecar.read_text())
    assert payload["sync_bit"] is False
    assert payload["n_pds"] == 4
    assert payload["n_frame_bits"] == 4
    assert payload["cycle"] == 16
    assert payload["channel_assignment"]["sync"] is None
    assert payload["channel_assignment"]["frame_bits"] == [
        "Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3",
    ]


@requires_ffmpeg
def test_sidecar_has_required_provenance_fields(tmp_path):
    vout, _ = _tag(tmp_path, sync_bit=True)
    sidecar = vout.with_suffix(vout.suffix + ".tags.json")
    payload = json.loads(sidecar.read_text())
    for field in (
        "schema_version", "tagged_at_utc", "input_video", "output_video",
        "calibration_file", "fps", "n_frames_written", "screen_size",
        "output_size", "sync_bit", "n_pds", "n_frame_bits", "cycle",
        "channel_assignment",
    ):
        assert field in payload, f"sidecar missing field: {field}"
