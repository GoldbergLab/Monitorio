"""End-to-end tests for the photodiode-recording decoder.

These build a synthetic DAQ-style multi-channel signal that matches what
the tagger would produce, run the decoder, and verify the recovered
frame->sample mapping is exact (or within a documented tolerance).

The decoder operates in strict mode: it assumes the recording contains
exactly one complete video playback bracketed by "video off" padding on
both sides, and errors out otherwise.

Coverage:
  - sync-bit mode round-trip with cycle wraps
  - --no-sync-bit mode round-trip
  - multi-segment recording is rejected (raises)
  - Intan-style raw ADC units via scale='intan_aux'
  - debounce filters short glitch dips
  - clear error when no video segments are present
  - sidecar is read for sync_bit setting
  - sync_bit_override beats the sidecar
  - CSV output contains the documented header fields and user metadata
  - voltage histogram differing from calibration produces a warning
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from calibration import gray
from conftest import make_calibration_json, make_video, requires_ffmpeg
from decode import SCALE_PRESETS, decode_sync_tags

SCRIPT = Path(__file__).resolve().parent.parent / "Source" / "add_video_sync_tags.py"


# Shared rig parameters.
SCREEN_W, SCREEN_H = 2400, 1600
PD_XS = (300, 700, 1100, 1500)
PD_Y = 100
DARK_V = 0.5
BRIGHT_V = 0.9
FPS = 30.0
SAMPLE_RATE = 50_000.0
N_FRAMES = 30  # per video segment
N_OFF_SAMPLES = 25_000  # 0.5 s of "video off" padding


def _make_cal(tmp_path: Path) -> Path:
    cal = make_calibration_json(
        tmp_path / "cal.json",
        screen_w=SCREEN_W, screen_h=SCREEN_H,
        pd_xs=PD_XS, pd_y=PD_Y,
    )
    # Add baseline_dark/bright_v fields the decoder uses for the sanity
    # check; make_calibration_json doesn't write them.
    payload = json.loads(cal.read_text())
    for p in payload["photodiodes"]:
        p["baseline_dark_v"] = DARK_V
        p["baseline_bright_v"] = BRIGHT_V
    cal.write_text(json.dumps(payload))
    return cal


def _tag_video(
    tmp_path: Path, cal: Path, *,
    sync_bit: bool = True, n_frames: int = N_FRAMES,
    pad_for_unambiguous_end: bool = True,
) -> Path:
    duration = n_frames / FPS
    vin = make_video(
        tmp_path / "in.mp4", SCREEN_W, SCREEN_H, duration=duration, fps=int(FPS),
    )
    vout = tmp_path / ("out_sync.mp4" if sync_bit else "out_no_sync.mp4")
    cmd = [sys.executable, str(SCRIPT), str(vin), str(vout),
           "--calibration-file", str(cal),
           "--sync-bit" if sync_bit else "--no-sync-bit"]
    if not pad_for_unambiguous_end:
        cmd.append("--no-pad-for-unambiguous-end")
    subprocess.run(cmd, capture_output=True, check=True)
    return vout


def _build_recording(
    *, n_plays: int = 1, sync_bit: bool = True, glitches: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """Build a synthetic 4-channel recording: pre-pad off, video plays, off pads.

    Returns (samples, frame_starts) where frame_starts[k] is the absolute
    sample index where the (k % N_FRAMES + 1)-th frame of the (k //
    N_FRAMES)-th play begins. Useful for asserting decoded sample indices
    exactly.
    """
    rng = np.random.default_rng(seed)
    samples_per_frame = SAMPLE_RATE / FPS
    n_video = int(round(N_FRAMES * samples_per_frame))
    n_total = N_OFF_SAMPLES + n_plays * (n_video + N_OFF_SAMPLES)
    ch = np.full((4, n_total), DARK_V, dtype=np.float64)
    ch += rng.normal(0, 0.005, ch.shape)

    n_frame_bits = 3 if sync_bit else 4
    cycle = 1 << n_frame_bits
    frame_bit_idx = list(range(1, 4)) if sync_bit else list(range(4))
    sync_idx = 0 if sync_bit else None

    frame_starts: list[int] = []
    cursor = N_OFF_SAMPLES
    for _ in range(n_plays):
        for f in range(1, N_FRAMES + 1):
            s0 = cursor + int(round((f - 1) * samples_per_frame))
            s1 = cursor + int(round(f * samples_per_frame))
            frame_starts.append(s0)
            if sync_idx is not None:
                ch[sync_idx, s0:s1] = BRIGHT_V + rng.normal(0, 0.005, s1 - s0)
            g = int(gray.encode(np.int64(f % cycle)))
            for k, idx in enumerate(frame_bit_idx):
                if (g >> k) & 1:
                    ch[idx, s0:s1] = BRIGHT_V + rng.normal(0, 0.005, s1 - s0)
        cursor += n_video + N_OFF_SAMPLES

    if glitches:
        # Inject ~1 ms (50-sample) glitch dips on channel 1, scattered.
        rng2 = np.random.default_rng(seed + 1)
        positions = rng2.integers(N_OFF_SAMPLES + 200, N_OFF_SAMPLES + n_video - 200, size=20)
        for p in positions:
            ch[1, p:p + 50] = DARK_V + rng2.normal(0, 0.005, 50)

    return ch, frame_starts


@requires_ffmpeg
def test_sync_bit_roundtrip_zero_error(tmp_path):
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)
    samples, expected_starts = _build_recording(sync_bit=True)
    result = decode_sync_tags(samples, SAMPLE_RATE, vout, cal)

    assert result.cycle == 8
    assert result.sync_bit is True
    assert result.frame_table.shape == (N_FRAMES, 2)
    for j, row in enumerate(result.frame_table):
        frame, sample = int(row[0]), int(row[1])
        assert frame == j + 1
        assert sample == expected_starts[j], (
            f"frame {frame}: decoded sample {sample}, expected {expected_starts[j]}"
        )
    assert result.warnings_ == []


@requires_ffmpeg
def test_no_sync_bit_roundtrip(tmp_path):
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=False)
    samples, expected_starts = _build_recording(sync_bit=False)
    result = decode_sync_tags(samples, SAMPLE_RATE, vout, cal)

    assert result.cycle == 16
    assert result.sync_bit is False
    assert result.frame_table.shape == (N_FRAMES, 2)
    for j, row in enumerate(result.frame_table):
        assert int(row[0]) == j + 1
        assert int(row[1]) == expected_starts[j]


@requires_ffmpeg
def test_multi_segment_recording_is_rejected(tmp_path):
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)
    # Two consecutive video plays in one recording -- decoder requires
    # exactly one segment and must raise.
    samples, _ = _build_recording(n_plays=2, sync_bit=True)
    with pytest.raises(RuntimeError, match="2 video segments detected"):
        decode_sync_tags(samples, SAMPLE_RATE, vout, cal)


@requires_ffmpeg
def test_intan_scale_factor(tmp_path):
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)
    samples, expected_starts = _build_recording(sync_bit=True)
    # Pretend the data is in Intan ADC steps: divide by the preset to
    # invert the scaling, then ask the decoder to multiply it back.
    intan_scale = SCALE_PRESETS["intan_aux"]
    intan_data = (samples / intan_scale).astype(np.float64)
    result = decode_sync_tags(intan_data, SAMPLE_RATE, vout, cal, scale="intan_aux")
    assert result.frame_table.shape == (N_FRAMES, 2)
    for j, row in enumerate(result.frame_table):
        assert int(row[1]) == expected_starts[j]


@requires_ffmpeg
def test_debounce_handles_short_glitches(tmp_path):
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)
    samples, expected_starts = _build_recording(sync_bit=True, glitches=True)
    result = decode_sync_tags(
        samples, SAMPLE_RATE, vout, cal, debounce_fraction=0.25,
    )
    assert result.frame_table.shape == (N_FRAMES, 2)
    # The snap-to-first-occurrence debounce should reconstruct exact
    # alignment even when a glitch lands close to a real transition.
    for j, row in enumerate(result.frame_table):
        assert int(row[1]) == expected_starts[j], (
            f"glitches broke frame {j+1} alignment "
            f"(decoded {int(row[1])}, expected {expected_starts[j]})"
        )


@requires_ffmpeg
def test_no_video_segments_errors(tmp_path):
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)
    rng = np.random.default_rng(0)
    blank = np.full((4, 100_000), DARK_V) + rng.normal(0, 0.005, (4, 100_000))
    with pytest.raises(RuntimeError, match="No bimodal signal"):
        decode_sync_tags(blank, SAMPLE_RATE, vout, cal)


@requires_ffmpeg
def test_sidecar_drives_sync_bit(tmp_path):
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=False)
    # Build an --no-sync-bit recording but ALSO build a sync-bit recording
    # of the same shape: the sidecar (sync_bit=False) should win and the
    # decoder should treat the data as 4-bit-frame.
    samples, _ = _build_recording(sync_bit=False)
    result = decode_sync_tags(samples, SAMPLE_RATE, vout, cal)
    assert result.sync_bit is False
    assert result.cycle == 16


@requires_ffmpeg
def test_sync_bit_override_beats_sidecar(tmp_path):
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)  # sidecar says True
    samples, _ = _build_recording(sync_bit=False)    # data has no sync bit
    # Force the override: now decoder treats it as no-sync-bit and decodes correctly.
    result = decode_sync_tags(
        samples, SAMPLE_RATE, vout, cal, sync_bit_override=False,
    )
    assert result.sync_bit is False
    assert result.cycle == 16


@requires_ffmpeg
def test_csv_header_includes_provenance_and_metadata(tmp_path):
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)
    samples, _ = _build_recording(sync_bit=True)
    csv_path = tmp_path / "decoded.csv"
    decode_sync_tags(
        samples, SAMPLE_RATE, vout, cal,
        output_path=csv_path,
        metadata="rig: dev\ntrial 7",
    )
    text = csv_path.read_text()
    # Provenance fields
    assert "# Monitorio sync-tag decoder" in text
    assert "# source_video:" in text
    assert "# calibration:" in text
    assert "# sample_rate_hz: 50000" in text
    assert "# fps: 30" in text
    assert "# sync_bit: true" in text
    assert "# cycle: 8" in text
    assert "# segment_samples:" in text
    assert "# thresholds_v:" in text
    # Multi-line metadata flattened to one line per `# user_metadata:` row.
    assert "# user_metadata: rig: dev" in text
    assert "# user_metadata: trial 7" in text
    # Data rows present, header present.
    assert "frame_number,sample_index" in text
    # First data row is for frame 1.
    data_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("#") and not line.startswith("frame_number,")
    ]
    assert data_lines[0].split(",")[0] == "1"


@requires_ffmpeg
def test_measured_fps_mismatch_warning(tmp_path):
    # Tag a 30 fps video, then claim the DAQ ran at 25 kHz when really
    # the synthetic samples were laid out at 50 kHz (ratio of 2x). The
    # decoder will see frame intervals half as long as expected and the
    # measured fps will be ~60 Hz instead of 30. The strict "decoded
    # count == expected count" check still passes (the bit pattern is
    # right), but the timing cross-check should warn.
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)
    samples, _ = _build_recording(sync_bit=True)
    # Pass half the true sample_rate -> measured fps doubles.
    result = decode_sync_tags(samples, SAMPLE_RATE / 2, vout, cal)
    assert any("measured fps" in w for w in result.warnings_), (
        f"expected an fps-mismatch warning, got: {result.warnings_}"
    )


@requires_ffmpeg
def test_measured_fps_matches_no_warning(tmp_path):
    # Sanity: when sample_rate is correct, no fps warning is emitted.
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)
    samples, _ = _build_recording(sync_bit=True)
    result = decode_sync_tags(samples, SAMPLE_RATE, vout, cal)
    assert not any("measured fps" in w for w in result.warnings_)


@requires_ffmpeg
def test_no_sync_bit_last_frame_all_zeros_synthesized(tmp_path):
    # In --no-sync-bit mode with cycle=16, a video whose frame count is
    # a multiple of 16 has its last frame Gray-encode to all-zeros --
    # indistinguishable from the post-video "off" pad. The decoder
    # should detect this case, synthesize the last frame at the nominal
    # interval, and emit a clear warning naming --sync-bit as the fix.
    cal = _make_cal(tmp_path)
    n_frames = 16  # exactly one cycle
    # Disable the tagger's padding so we exercise the decoder's
    # safety-net synthesis path (otherwise the tagger removes the
    # ambiguity by appending a frame before the decoder ever sees it).
    vout = _tag_video(
        tmp_path, cal, sync_bit=False, n_frames=n_frames,
        pad_for_unambiguous_end=False,
    )
    samples_per_frame = SAMPLE_RATE / FPS
    rng = np.random.default_rng(7)
    n_total = N_OFF_SAMPLES + int(round(n_frames * samples_per_frame)) + N_OFF_SAMPLES
    ch = np.full((4, n_total), DARK_V, dtype=np.float64)
    ch += rng.normal(0, 0.005, ch.shape)
    expected_starts = []
    for f in range(1, n_frames + 1):
        s0 = N_OFF_SAMPLES + int(round((f - 1) * samples_per_frame))
        s1 = N_OFF_SAMPLES + int(round(f * samples_per_frame))
        expected_starts.append(s0)
        g = int(gray.encode(np.int64(f % 16)))
        for k in range(4):
            if (g >> k) & 1:
                ch[k, s0:s1] = BRIGHT_V + rng.normal(0, 0.005, s1 - s0)

    result = decode_sync_tags(ch, SAMPLE_RATE, vout, cal)
    # Synthesized last frame brings the total back up to n_frames.
    assert result.frame_table.shape == (n_frames, 2)
    assert int(result.frame_table[-1, 0]) == n_frames
    # And the warning explains what happened.
    assert any(
        "Gray-encodes to all zeros" in w and "--sync-bit" in w
        for w in result.warnings_
    ), f"expected ambiguity warning, got: {result.warnings_}"
    # Synthesized sample index is at last_real + nominal_interval --
    # within ±1 sample of the true frame start.
    last_real_sample = int(result.frame_table[-2, 1])
    synth_sample = int(result.frame_table[-1, 1])
    assert synth_sample == last_real_sample + int(round(SAMPLE_RATE / FPS))
    # Sanity: prior frames still align exactly.
    for j in range(n_frames - 1):
        assert int(result.frame_table[j, 1]) == expected_starts[j]


@requires_ffmpeg
def test_no_sync_bit_padding_eliminates_ambiguity(tmp_path):
    # Same input video as the safety-net test (16 frames, exact cycle),
    # but let the tagger pad as it does by default. The output now has
    # 17 frames (frame 17 = gray(1) = bit 0 lit, distinct from off), so
    # there's no ambiguity and the decoder produces a clean answer with
    # no synthesis-warning.
    cal = _make_cal(tmp_path)
    n_frames = 16  # exact cycle in source
    vout = _tag_video(
        tmp_path, cal, sync_bit=False, n_frames=n_frames,
        # default: pad_for_unambiguous_end=True
    )
    # Sidecar should reflect the padding.
    sidecar = json.loads((vout.with_suffix(vout.suffix + ".tags.json")).read_text())
    assert sidecar["padded_frames"] == 1
    assert sidecar["n_frames_written"] == n_frames + 1

    # Build a recording matching the PADDED 17-frame output.
    samples_per_frame = SAMPLE_RATE / FPS
    rng = np.random.default_rng(11)
    n_total = N_OFF_SAMPLES + int(round((n_frames + 1) * samples_per_frame)) + N_OFF_SAMPLES
    ch = np.full((4, n_total), DARK_V, dtype=np.float64)
    ch += rng.normal(0, 0.005, ch.shape)
    for f in range(1, n_frames + 1 + 1):  # frames 1..17
        s0 = N_OFF_SAMPLES + int(round((f - 1) * samples_per_frame))
        s1 = N_OFF_SAMPLES + int(round(f * samples_per_frame))
        g = int(gray.encode(np.int64(f % 16)))
        for k in range(4):
            if (g >> k) & 1:
                ch[k, s0:s1] = BRIGHT_V + rng.normal(0, 0.005, s1 - s0)
    result = decode_sync_tags(ch, SAMPLE_RATE, vout, cal)
    assert result.frame_table.shape == (n_frames + 1, 2)
    # No ambiguity warning since the tagger pre-empted the all-zeros
    # last-frame case by padding.
    assert not any("Gray-encodes to all zeros" in w for w in result.warnings_)


@requires_ffmpeg
def test_sync_bit_last_frame_all_zeros_no_ambiguity(tmp_path):
    # Sanity: with --sync-bit on, the same exact-cycle situation has no
    # ambiguity because the sync bit stays lit on the all-zeros frame.
    cal = _make_cal(tmp_path)
    n_frames = 8  # exactly one cycle in sync-bit mode
    vout = _tag_video(tmp_path, cal, sync_bit=True, n_frames=n_frames)
    samples_per_frame = SAMPLE_RATE / FPS
    rng = np.random.default_rng(8)
    n_total = N_OFF_SAMPLES + int(round(n_frames * samples_per_frame)) + N_OFF_SAMPLES
    ch = np.full((4, n_total), DARK_V, dtype=np.float64)
    ch += rng.normal(0, 0.005, ch.shape)
    for f in range(1, n_frames + 1):
        s0 = N_OFF_SAMPLES + int(round((f - 1) * samples_per_frame))
        s1 = N_OFF_SAMPLES + int(round(f * samples_per_frame))
        ch[0, s0:s1] = BRIGHT_V + rng.normal(0, 0.005, s1 - s0)
        g = int(gray.encode(np.int64(f % 8)))
        for k in range(3):
            if (g >> k) & 1:
                ch[k + 1, s0:s1] = BRIGHT_V + rng.normal(0, 0.005, s1 - s0)
    result = decode_sync_tags(ch, SAMPLE_RATE, vout, cal)
    assert result.frame_table.shape == (n_frames, 2)
    # No "Gray-encodes to all zeros" warning -- sync bit kept it visible.
    assert not any("all zeros" in w for w in result.warnings_)


@requires_ffmpeg
def test_threshold_drift_warning(tmp_path):
    # Build a recording whose voltages are wildly different from the
    # calibration baselines (e.g. someone forgot the scale factor).
    cal = _make_cal(tmp_path)
    vout = _tag_video(tmp_path, cal, sync_bit=True)
    samples, _ = _build_recording(sync_bit=True)
    # Multiply all signals by 100 -- threshold detection still finds a
    # bimodal split, but the absolute level is way outside calibration's
    # 0.5..0.9 V range, so the sanity check should warn.
    inflated = samples * 100.0
    result = decode_sync_tags(inflated, SAMPLE_RATE, vout, cal)
    assert any("differ" in w or "far from" in w for w in result.warnings_), (
        f"expected a threshold-mismatch warning, got: {result.warnings_}"
    )
