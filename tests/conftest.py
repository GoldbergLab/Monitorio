"""Shared pytest fixtures and helpers for the Monitorio test suite.

Adds Source/ to sys.path so tests can `from calibration import gray`,
`import add_video_sync_tags`, etc. Defines the `requires_ffmpeg` skip
marker and small builders for synthetic videos and calibration JSON
payloads.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "Source"
sys.path.insert(0, str(SOURCE_DIR))


def _ffmpeg_present() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_present(),
    reason="ffmpeg/ffprobe not on PATH",
)


def make_calibration_json(
    path: Path,
    *,
    screen_w: int = 2400,
    screen_h: int = 1600,
    pd_xs=(300, 700, 1100, 1500),
    pd_y: int = 100,
    bit_radius: int = 20,
    background_radius: int = 35,
) -> Path:
    """Write a minimally-valid calibration JSON to `path` and return it.

    The JSON includes only the fields add_video_sync_tags reads from --
    `monitor.{width,height}` and the per-PD tag positions/radii. Other
    fields (DAQ/crosstalk/etc.) are omitted; the consumer ignores them.
    """
    cal = {
        "version": 1,
        "monitor": {"index": 0, "width": int(screen_w), "height": int(screen_h)},
        "photodiodes": [
            {
                "channel": f"Dev1/ai{i}",
                "x_px": float(x), "y_px": float(pd_y),
                "bit_radius_px": int(bit_radius),
                "background_radius_px": int(background_radius),
            }
            for i, x in enumerate(pd_xs)
        ],
    }
    path.write_text(json.dumps(cal))
    return path


def make_video(path: Path, w: int, h: int, *, duration: float = 1.0, fps: int = 30) -> Path:
    """Write an h.264 testsrc video at `path` and return it.

    Uses ffmpeg's lavfi testsrc generator -- no input file needed. The
    output is libx264 at a fast preset so this stays cheap to run inside
    tests.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration}:size={w}x{h}:rate={fps}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            str(path),
        ],
        check=True,
    )
    return path


def probe_video_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream in `path`."""
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path),
        ]
    )
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def decode_frames(path: Path):
    """Decode all video frames from `path` into a (n, h, w, 3) uint8 array."""
    import numpy as np

    w, h = probe_video_size(path)
    raw = subprocess.check_output(
        [
            "ffmpeg", "-loglevel", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
