"""Regression tests against real Monitorio-rig recordings.

These tests run end-to-end against actual photodiode data captured on
the lab's Intan Recording Controller, with the matching tagged video,
sidecar, and calibration JSON. They catch regressions that a synthetic
data set might miss -- analog noise floor and clipping shape, monitor
refresh aliasing patterns, the rising-edge skew between PDs, real
ffprobe output, etc.

Fixture data lives in tests/fixtures/<name>/. See each fixture's
README.md for capture conditions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from decode_sync_tags import decode_sync_tags
from conftest import requires_ffmpeg

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@requires_ffmpeg
def test_test2_segment1_decodes_cleanly():
    fixture = FIXTURES / "test2_segment1"
    if not (fixture / "samples.npz").exists():
        pytest.skip(f"fixture not present: {fixture}")

    payload = np.load(fixture / "samples.npz", allow_pickle=False)
    samples = payload["samples"].astype(np.float64)
    sample_rate = float(payload["sample_rate"])

    result = decode_sync_tags(
        samples,
        sample_rate=sample_rate,
        video_path=fixture / "tagged_video.mp4",
        calibration_path=fixture / "calibration.json",
    )

    # The fixture is one complete playback of a 451-source-frame video
    # tagged with 5 leading guard frames. Guards have sync OFF so the
    # decoder ignores them; a clean recording produces exactly 451
    # decoded frames with zero warnings.
    assert result.frame_table.shape == (451, 2)
    assert result.warnings_ == [], (
        f"unexpected warnings on real-world fixture: {result.warnings_}"
    )

    # Frame numbers run 1..451 contiguously.
    np.testing.assert_array_equal(
        result.frame_table[:, 0],
        np.arange(1, 452, dtype=np.int64),
    )

    # Sample indices are monotonically increasing.
    deltas = np.diff(result.frame_table[:, 1])
    assert (deltas > 0).all()

    # Inter-frame timing matches the video's 45 fps within ~1% (the
    # measured-fps cross-check inside the decoder uses the same logic
    # but is a softer warn-only check; here we want a hard assert).
    n_intervals = result.frame_table.shape[0] - 1
    span_s = (
        int(result.frame_table[-1, 1]) - int(result.frame_table[0, 1])
    ) / sample_rate
    measured_fps = n_intervals / span_s
    assert abs(measured_fps - 45.0) / 45.0 < 0.01, (
        f"measured fps {measured_fps:.4f} differs from declared 45.0 by more than 1%"
    )

    # Sync-bit + cycle parameters propagated correctly out of the
    # sidecar.
    assert result.sync_bit is True
    assert result.cycle == 4
    assert result.n_pds == 3
