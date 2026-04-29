"""Decode photodiode recordings into a video-frame -> DAQ-sample table.

Given recorded analog signals from C photodiodes over N samples (e.g.
captured on an NI DAQ or Intan controller), the matching tagged video
file, and the calibration JSON, this module:

  1. Picks per-channel thresholds from the recording's own bimodal
     histogram (Otsu), with the calibration JSON as a sanity check.
  2. Binarizes each channel and debounces short transients (mid-frame
     blips, scan-line artifacts) by dropping runs shorter than a
     fraction of one frame interval.
  3. Identifies "video on" segments via the sync bit (or, with
     --no-sync-bit, via channel activity).
  4. Within each segment, Gray-decodes the cyclic frame number and
     unwraps to absolute frame numbers using the video's known fps and
     the DAQ sample rate.
  5. Returns (and optionally writes to CSV) a table mapping each
     decoded video frame number to the DAQ sample at which that frame
     started on screen.

API:

    result = decode_sync_tags(
        samples,                # (n_channels, n_samples)
        sample_rate,            # Hz
        video_path,             # the tagged video; ffprobe is called on it
        calibration_path,       # the Monitorio calibration JSON
        scale=1.0,              # multiply samples by this to get volts
        debounce_fraction=0.25, # of one frame interval
        sync_bit_override=None, # else read from the video's .tags.json sidecar
        output_path=None,       # CSV out
        metadata=None,          # extra string baked into the CSV header
    )

The decoder is unit-agnostic given the right `scale`. NI DAQ records in
volts (`scale=1.0`); the Intan auxiliary input channels record raw
ADC steps that convert to volts via `scale=0.0000374`. See
SCALE_PRESETS for shorthand names.

Channel ordering: `samples[i]` must correspond to the i-th photodiode
in the calibration JSON's `photodiodes` list (same order as the
calibration tool's AI-physical-pin enumeration).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration import gray  # noqa: E402


# Convenience presets for the sample -> volts conversion. Pass these
# (or their numeric values) as the `scale` argument.
SCALE_PRESETS = {
    "volts": 1.0,                          # NI-DAQmx default unit
    "intan_aux": 0.0000374,                # RHD2000 auxiliary input ADC steps
    "intan_supply": 0.0000748,             # RHD2000 supply voltage ADC steps
    "intan_adc_board_mode_0": 0.000050354, # RHD USB interface board, board_mode=0
    # Intan board ADCs in board_mode 1 / 13 also need an offset subtraction
    # before scaling (raw - 32768) * step. They're not single-multiplier
    # presets and aren't included here -- subtract the offset upstream.
}


@dataclass
class DecodeResult:
    """Result of `decode_sync_tags`.

    frame_table: int64 (n_frames, 2) -- columns are (frame_number,
                 sample_index). frame_number is 1-indexed (mirrors the
                 tagger's 1-indexing). sample_index is into the original
                 `samples` array passed in.

    The decoder assumes the recording contains exactly one complete
    video playback bracketed by "video off" padding on both sides; it
    errors out if that assumption is violated (zero or multiple
    segments detected, segment doesn't start at frame 1, total decoded
    frame count doesn't match the source video). `warnings_` collects
    softer diagnostics (e.g. thresholds drifted vs calibration, frames
    dropped and recovered via timing).
    """
    frame_table: np.ndarray
    fps: float
    sample_rate: float
    sync_bit: bool
    cycle: int
    n_pds: int
    thresholds_v: list[float]
    segment_start_sample: int
    segment_end_sample: int
    warnings_: list[str] = field(default_factory=list)


def decode_sync_tags(
    samples,
    sample_rate: float,
    video_path,
    calibration_path,
    *,
    scale: float | str = 1.0,
    debounce_fraction: float = 0.25,
    sync_bit_override: bool | None = None,
    output_path=None,
    metadata: str | None = None,
) -> DecodeResult:
    """High-level decoder: load context from disk, then call the core."""
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2D (n_channels, n_samples); got shape {samples.shape}"
        )
    if samples.shape[1] < 2:
        raise ValueError(
            f"samples has only {samples.shape[1]} time points; need at least 2"
        )

    # Resolve scale: accept a preset name or a number.
    if isinstance(scale, str):
        if scale not in SCALE_PRESETS:
            raise ValueError(
                f"Unknown scale preset {scale!r}; "
                f"choose from {list(SCALE_PRESETS)} or pass a numeric scale."
            )
        scale_value = SCALE_PRESETS[scale]
    else:
        scale_value = float(scale)

    samples_v = samples.astype(np.float64) * scale_value

    video_path = Path(video_path)
    calibration_path = Path(calibration_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not calibration_path.exists():
        raise FileNotFoundError(calibration_path)

    # Pull fps + frame count from the tagged video.
    info = _probe_video(video_path)
    fps = float(info["fps"])
    expected_n_frames = info["n_frames"]

    # Calibration: per-PD baselines (for sanity check) and channel order.
    with open(calibration_path) as f:
        cal = json.load(f)
    pds = cal.get("photodiodes") or []
    n_pds = len(pds)
    if n_pds == 0:
        raise ValueError(f"{calibration_path} has no photodiodes")
    if samples_v.shape[0] != n_pds:
        raise ValueError(
            f"samples has {samples_v.shape[0]} channels but calibration has "
            f"{n_pds} photodiodes; channel order must match"
        )
    cal_dark_v = [
        float(p.get("baseline_dark_v", float("nan"))) for p in pds
    ]
    cal_bright_v = [
        float(p.get("baseline_bright_v", float("nan"))) for p in pds
    ]
    cal_channels = [p.get("channel", f"PD#{i}") for i, p in enumerate(pds)]

    # Sidecar: read tagging-time decisions (sync_bit + bit assignment).
    sidecar_path = video_path.with_suffix(video_path.suffix + ".tags.json")
    sidecar = None
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"  warning: could not read sidecar {sidecar_path}: {e}; "
                f"falling back to defaults",
                file=sys.stderr,
            )

    if sync_bit_override is not None:
        sync_bit = bool(sync_bit_override)
    elif sidecar is not None:
        sync_bit = bool(sidecar.get("sync_bit", True))
    else:
        sync_bit = True
        print(
            f"  note: no sidecar at {sidecar_path}; assuming sync_bit=True. "
            f"Pass sync_bit_override to force the value if this is wrong.",
            file=sys.stderr,
        )

    result = _decode_core(
        samples_v=samples_v,
        sample_rate=float(sample_rate),
        fps=fps,
        sync_bit=sync_bit,
        cal_dark_v=cal_dark_v,
        cal_bright_v=cal_bright_v,
        debounce_fraction=float(debounce_fraction),
        expected_n_frames=expected_n_frames,
    )

    # Optional CSV output. Header carries provenance + per-channel
    # thresholds + the user-supplied metadata string, so a reader can
    # always reconstruct what produced the table.
    if output_path is not None:
        _write_csv(
            output_path=Path(output_path),
            result=result,
            video_path=str(video_path),
            calibration_path=str(calibration_path),
            sidecar_path=str(sidecar_path) if sidecar is not None else None,
            scale=scale_value,
            cal_channels=cal_channels,
            metadata=metadata,
        )

    return result


def _decode_core(
    *,
    samples_v: np.ndarray,
    sample_rate: float,
    fps: float,
    sync_bit: bool,
    cal_dark_v: list[float],
    cal_bright_v: list[float],
    debounce_fraction: float,
    expected_n_frames: int | None,
) -> DecodeResult:
    """Pure-numpy decoder. Takes voltages + all params explicitly."""
    n_channels, n_samples = samples_v.shape
    if sync_bit and n_channels < 2:
        raise ValueError(
            "sync_bit=True requires at least 2 channels (one sync + at "
            f"least one frame bit); got {n_channels}"
        )

    n_frame_bits = n_channels - 1 if sync_bit else n_channels
    cycle = 1 << n_frame_bits
    sync_idx = 0 if sync_bit else None
    frame_bit_idx = (
        list(range(1, n_channels)) if sync_bit else list(range(n_channels))
    )

    warnings_: list[str] = []

    # 1. Per-channel thresholds via Otsu, sanity-checked vs calibration,
    #    plus an SNR check to catch "no signal at all" before we try to
    #    decode pure noise.
    thresholds: list[float] = []
    snr_per_channel: list[float] = []
    for i in range(n_channels):
        thr = _otsu_threshold(samples_v[i])
        thresholds.append(thr)
        # SNR := (mean above threshold - mean below) / max(within-cluster std).
        # On a true bimodal signal this is huge (~100); on flat noise it is ~1.
        above = samples_v[i] > thr
        if above.any() and (~above).any():
            mu_hi = float(samples_v[i, above].mean())
            mu_lo = float(samples_v[i, ~above].mean())
            std_hi = float(samples_v[i, above].std())
            std_lo = float(samples_v[i, ~above].std())
            sep = mu_hi - mu_lo
            within = max(std_hi, std_lo, 1e-12)
            snr = sep / within
        else:
            snr = 0.0
        snr_per_channel.append(snr)

        d, b = cal_dark_v[i], cal_bright_v[i]
        if np.isfinite(d) and np.isfinite(b) and b > d:
            cal_mid = (d + b) / 2.0
            cal_range = b - d
            # If the auto-detected threshold is outside the dynamic range
            # the calibration measured by more than one full range, the
            # data and the calibration disagree -- worth flagging.
            if not (d - cal_range < thr < b + cal_range):
                warnings_.append(
                    f"channel {i}: detected threshold {thr:.4g} "
                    f"(post-scale units = volts) is far from the calibration "
                    f"midpoint {cal_mid:.4g} V (range {cal_range:.4g} V). "
                    f"Either the rig has drifted, the scale factor is wrong, "
                    f"or this channel saw no real signal."
                )

    # If the channel responsible for "video on" detection has no real
    # signal (SNR ~ noise), the rest of the pipeline will hallucinate
    # segments out of pure noise. Bail out clearly instead.
    SNR_FLOOR = 5.0
    sync_idx_for_check = 0 if sync_bit else None
    relevant_channels = (
        [sync_idx_for_check] if sync_bit else list(range(n_channels))
    )
    if all(snr_per_channel[i] < SNR_FLOOR for i in relevant_channels):
        which = (
            f"sync-bit channel (index 0)" if sync_bit
            else f"any of {n_channels} channels"
        )
        raise RuntimeError(
            f"No bimodal signal detected on {which}: per-channel SNR = "
            f"{[round(s, 2) for s in snr_per_channel]} (need >= {SNR_FLOOR}). "
            f"Recording probably contains no video playback, or the channel "
            f"order is wrong, or the scale factor produced near-constant "
            f"voltages."
        )

    # 2. Binarize.
    binary = np.zeros(samples_v.shape, dtype=bool)
    for i in range(n_channels):
        binary[i] = samples_v[i] > thresholds[i]

    # 3. Debounce: drop runs shorter than a fraction of one frame.
    samples_per_frame = sample_rate / fps
    if samples_per_frame < 2:
        raise ValueError(
            f"sample_rate ({sample_rate} Hz) must be at least 2 samples/frame "
            f"at fps={fps}; got {samples_per_frame:.2f}. The decoder can't "
            f"resolve frame-by-frame transitions below Nyquist."
        )
    debounce_n = max(1, int(round(debounce_fraction * samples_per_frame)))
    for i in range(n_channels):
        binary[i] = _debounce_runs(binary[i], debounce_n)

    # 4. Identify "video on" segments.
    if sync_bit:
        video_on = binary[sync_idx]
    else:
        # No sync bit: assume video is on whenever any frame bit is high.
        # The decoder will still see brief all-dark frames at multiples of
        # cycle as "off" -- which is exactly the limitation the sync-bit
        # mode was added to remove. Without sync bit, treat such gaps as
        # part of a continuous segment if they're shorter than a few
        # frames; rely on debounce + segment merging.
        video_on = binary.any(axis=0)
        # Bridge short False gaps within a segment (multiple of cycle ->
        # all-dark for one frame): close gaps shorter than 2 frames.
        bridge_n = max(1, int(round(2 * samples_per_frame)))
        video_on = ~_debounce_runs(~video_on, bridge_n)

    segments = _runs_of_true(video_on)
    if not segments:
        raise RuntimeError(
            "No video segments detected (no sustained 'video on' state). "
            "Check that the recording contains actual video playback, that "
            "channel order matches the calibration JSON, and that the "
            "scale factor produces voltages comparable to the calibration."
        )
    if len(segments) > 1:
        seg_lengths = [e - s for s, e in segments]
        raise RuntimeError(
            f"{len(segments)} video segments detected (lengths in samples: "
            f"{seg_lengths}). The decoder assumes exactly one complete "
            f"video playback bracketed by 'video off' periods on both sides. "
            f"If the recording really does contain only one play, this "
            f"likely means a debounce / threshold issue split the segment; "
            f"otherwise chunk the recording before passing it in."
        )

    start, end = segments[0]
    seg_len = end - start

    # Per-sample cyclic Gray code over the segment.
    cyclic = np.zeros(seg_len, dtype=np.int64)
    for k, idx in enumerate(frame_bit_idx):
        cyclic |= (binary[idx, start:end].astype(np.int64) << k)
    decoded = gray.decode(cyclic, n_bits=max(n_frame_bits, 1)).astype(np.int64)

    # Where the decoded value changes, that's the start of a new frame.
    change = np.diff(decoded)
    transition_offsets = np.where(change != 0)[0] + 1
    # First sample of segment is the start of frame 1.
    frame_start_offsets = np.concatenate(([0], transition_offsets))
    cyclic_at_starts = decoded[frame_start_offsets]

    # First frame must encode to gray(1 % cycle); anything else means the
    # segment doesn't actually start at frame 1 (recording is partial,
    # or thresholding/debouncing missed the real first transition).
    expected_first = 1 % cycle
    if cyclic_at_starts[0] != expected_first:
        raise RuntimeError(
            f"first decoded cyclic frame is {int(cyclic_at_starts[0])}, "
            f"expected {expected_first}. The segment doesn't start at "
            f"frame 1 -- either the recording is missing the start of "
            f"the video, or the threshold / debounce is misidentifying "
            f"the first transition."
        )

    # Unwrap to absolute frame numbers, using sample timing to resolve
    # cycle wraps (and any dropped frames).
    absolute = np.zeros(len(frame_start_offsets), dtype=np.int64)
    absolute[0] = 1
    for j in range(1, len(frame_start_offsets)):
        delta_samples = frame_start_offsets[j] - frame_start_offsets[j - 1]
        timing_advance = int(round(delta_samples * fps / sample_rate))
        cyclic_advance = int(
            (cyclic_at_starts[j] - cyclic_at_starts[j - 1]) % cycle
        )
        if cyclic_advance == 0:
            cyclic_advance = cycle
        # Pick k such that cyclic_advance + k*cycle is closest to
        # timing_advance. Negative k would mean going backwards in time
        # -- impossible, so flag it.
        k = round((timing_advance - cyclic_advance) / cycle)
        if k < 0:
            raise RuntimeError(
                f"transition #{j}: timing-derived advance ({timing_advance}) "
                f"is less than the cyclic advance ({cyclic_advance}). The "
                f"DAQ sample rate or video fps doesn't match what the "
                f"sidecar reports, or the threshold split the bits "
                f"inconsistently."
            )
        if k > 0:
            warnings_.append(
                f"transition #{j} (frame ~{absolute[j - 1] + cyclic_advance}): "
                f"detected {k} dropped cycle(s) ({k * cycle} frames) -- "
                f"recovered via timing."
            )
        advance = cyclic_advance + k * cycle
        absolute[j] = absolute[j - 1] + advance

    rows = [
        (int(absolute[j]), int(start + frame_start_offsets[j]))
        for j in range(len(frame_start_offsets))
    ]

    # Recording-level check: decoded frame count vs. the source video's
    # frame count.
    if expected_n_frames is not None:
        decoded_total = int(absolute[-1])
        diff = expected_n_frames - decoded_total
        samples_per_frame_nominal = sample_rate / fps
        if not sync_bit and diff == 1 and expected_n_frames % cycle == 0:
            # Specific ambiguity unique to --no-sync-bit mode: the last
            # frame Gray-encodes to all zeros (frame number is a multiple
            # of the cycle), so it's indistinguishable from the
            # post-video "off" period in the recording. The user has
            # guaranteed the recording brackets a complete video, so we
            # synthesize the missing last frame at the nominal interval
            # past the last detected frame and warn explicitly.
            synth_sample = int(rows[-1][1] + round(samples_per_frame_nominal))
            rows.append((expected_n_frames, synth_sample))
            warnings_.append(
                f"in --no-sync-bit mode, frame {expected_n_frames} "
                f"Gray-encodes to all zeros and is indistinguishable from "
                f"the post-video 'off' period; synthesized its sample "
                f"index ({synth_sample}) as last detected sample + "
                f"sample_rate/fps. Re-tag with --sync-bit to remove the "
                f"ambiguity."
            )
        elif abs(diff) > 1:
            raise RuntimeError(
                f"decoded {decoded_total} frames but the source video has "
                f"{expected_n_frames}. The recording is either truncated, "
                f"missing the trailing video-off pad, or the unwrap is "
                f"miscounting (dropped frames near the end can be "
                f"unrecoverable when there's no timing slack)."
            )
        elif diff != 0:
            warnings_.append(
                f"decoded {decoded_total} frames but the source video has "
                f"{expected_n_frames} (off by {diff:+d}). Likely a single "
                f"dropped frame near the end of the recording, or a "
                f"boundary-rounding artifact in the segment detection."
            )

    # Measured frame rate cross-check: time per frame as observed in the
    # recording must agree with the ffprobed nominal fps. A mismatch
    # points at a wrong DAQ sample_rate, a wrong fps in the video's
    # header, or a different display refresh rate than the encoded fps.
    if len(rows) >= 2:
        first_sample = rows[0][1]
        last_sample = rows[-1][1]
        n_intervals = len(rows) - 1
        duration_s = (last_sample - first_sample) / sample_rate
        if duration_s > 0:
            measured_fps = n_intervals / duration_s
            rel_err = abs(measured_fps - fps) / fps
            if rel_err > 0.01:  # >1% discrepancy
                warnings_.append(
                    f"measured fps from frame timing ({measured_fps:.4f} Hz) "
                    f"disagrees with the video's declared fps ({fps:.4f} Hz) "
                    f"by {rel_err:.1%}. Check that sample_rate ({sample_rate} "
                    f"Hz) matches the DAQ used for the recording, that the "
                    f"video was played at its native fps, and that the "
                    f"display refresh rate matches the video fps."
                )

    frame_table = (
        np.array(rows, dtype=np.int64) if rows
        else np.empty((0, 2), dtype=np.int64)
    )

    return DecodeResult(
        frame_table=frame_table,
        fps=fps,
        sample_rate=sample_rate,
        sync_bit=sync_bit,
        cycle=cycle,
        n_pds=n_channels,
        thresholds_v=thresholds,
        segment_start_sample=int(start),
        segment_end_sample=int(end),
        warnings_=warnings_,
    )


# ----- helpers --------------------------------------------------------

def _otsu_threshold(x: np.ndarray) -> float:
    """Otsu's threshold on a 1D array. Returns the threshold value (in x's units).

    If x is degenerate (essentially constant), returns its mean.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if not np.isfinite(x).all() or x.std() < 1e-12:
        return float(x.mean())
    n_bins = 256
    hist, edges = np.histogram(x, bins=n_bins)
    total = hist.sum()
    if total == 0:
        return float(x.mean())
    centers = (edges[:-1] + edges[1:]) / 2.0
    cum = np.cumsum(hist).astype(np.float64)
    cum_x = np.cumsum(hist * centers)
    total_x = cum_x[-1]
    w0 = cum / total
    w1 = 1.0 - w0
    # Avoid 0-division at the extremes.
    valid = (cum > 0) & (total - cum > 0)
    mu0 = np.where(valid, cum_x / np.where(cum > 0, cum, 1), 0.0)
    mu1 = np.where(valid, (total_x - cum_x) / np.where(total - cum > 0, total - cum, 1), 0.0)
    sigma_b2 = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, 0.0)
    best = int(np.argmax(sigma_b2))
    return float(centers[best])


def _debounce_runs(binary: np.ndarray, min_run: int) -> np.ndarray:
    """Identify sustained runs (length >= min_run) as the only "real"
    states; for the gaps between them, snap transitions to the first
    sample matching the next sustained state's value.

    This handles the case where a real transition is closely followed
    by a short glitch back to the old value: a naive "drop short runs"
    filter would treat the new state as transient and erase it. The
    snap-to-first-occurrence rule keeps the real transition at its
    correct sample even if the new state has a short dip in its first
    few samples.
    """
    binary = np.asarray(binary, dtype=bool)
    n = binary.size
    if min_run <= 1 or n < 2:
        return binary.copy()
    sig = binary.astype(np.int8)
    diff = np.diff(sig)
    breaks = np.where(diff != 0)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))
    lengths = ends - starts
    values = sig[starts]
    keep = lengths >= min_run

    if not keep.any():
        # Every run is shorter than min_run -- the signal is essentially
        # noise. Return unchanged; flipping it to a constant would invent
        # structure that isn't there. The SNR check earlier should already
        # have rejected this case.
        return binary.copy()

    out = binary.copy()
    kept_idx = np.where(keep)[0]

    # Prefix (before the first kept run): assume the rig was already in
    # the kept run's state.
    first = kept_idx[0]
    out[:starts[first]] = bool(values[first])
    # Suffix: similarly, after the last kept run, hold its state.
    last = kept_idx[-1]
    out[ends[last]:] = bool(values[last])

    # Gaps between consecutive kept runs.
    #
    # Inter-frame brightness dips on monitors are one-directional
    # (BRIGHT pixels briefly dim during refresh), so the contaminated
    # state in any gap is the BRIGHT/True one. The clean state has a
    # crisp boundary; the contaminated state has glitchy dips at the
    # edge. To recover the real transition we use the boundary of the
    # CLEAN state:
    #   OFF -> ON: the OFF state (before transition) is clean; the new
    #              ON state has dips at its start. Real transition is at
    #              the first True sample in the gap.
    #   ON -> OFF: the OFF state (after transition) is clean; the dying
    #              ON state has dips at its end. Real transition is at
    #              the sample after the last True in the gap.
    for ki in range(len(kept_idx) - 1):
        a = kept_idx[ki]
        b = kept_idx[ki + 1]
        gap_start = int(ends[a])
        gap_end = int(starts[b])
        prev_val = bool(values[a])
        next_val = bool(values[b])
        if prev_val == next_val:
            out[gap_start:gap_end] = prev_val
            continue
        gap = binary[gap_start:gap_end]
        if next_val:  # OFF -> ON: snap to first True in gap.
            ones = np.where(gap)[0]
            trans_idx = gap_start + int(ones[0]) if ones.size else gap_end
        else:  # ON -> OFF: snap to (last True in gap) + 1.
            ones = np.where(gap)[0]
            trans_idx = (
                gap_start + int(ones[-1]) + 1 if ones.size else gap_start
            )
        out[gap_start:trans_idx] = prev_val
        out[trans_idx:gap_end] = next_val
    return out


def _runs_of_true(b: np.ndarray) -> list[tuple[int, int]]:
    """Return a list of (start, end) for each contiguous True run; end is exclusive."""
    b = np.asarray(b, dtype=bool)
    if b.size == 0 or not b.any():
        return []
    pad = np.concatenate(([False], b, [False]))
    diff = np.diff(pad.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _probe_video(path: Path) -> dict:
    """Return {width, height, fps, n_frames} for a video file via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on {path}:\n{proc.stderr.decode(errors='replace')}"
        )
    data = json.loads(proc.stdout)["streams"][0]
    w, h = int(data["width"]), int(data["height"])
    r_fr = data["r_frame_rate"]
    if "/" in r_fr:
        num, den = r_fr.split("/")
        fps = float(num) / float(den) if float(den) else float(num)
    else:
        fps = float(r_fr)
    n_frames = None
    if data.get("nb_frames") not in (None, "N/A"):
        try:
            n_frames = int(data["nb_frames"])
        except (TypeError, ValueError):
            pass
    if n_frames is None and data.get("duration") not in (None, "N/A"):
        try:
            n_frames = int(round(float(data["duration"]) * fps))
        except (TypeError, ValueError):
            pass
    return {"width": w, "height": h, "fps": fps, "n_frames": n_frames}


def _write_csv(
    *,
    output_path: Path,
    result: DecodeResult,
    video_path: str,
    calibration_path: str,
    sidecar_path: str | None,
    scale: float,
    cal_channels: list[str],
    metadata: str | None,
) -> None:
    """Write the per-frame table as CSV with a `#`-prefixed metadata header."""
    lines = [
        "# Monitorio sync-tag decoder",
        f"# decoded_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"# source_video: {video_path}",
        f"# calibration: {calibration_path}",
        f"# sidecar: {sidecar_path or '(none -- defaults used)'}",
        f"# sample_rate_hz: {result.sample_rate}",
        f"# fps: {result.fps}",
        f"# scale_to_volts: {scale}",
        f"# sync_bit: {str(result.sync_bit).lower()}",
        f"# cycle: {result.cycle}",
        f"# n_pds: {result.n_pds}",
        f"# segment_samples: [{result.segment_start_sample},{result.segment_end_sample})",
        "# thresholds_v: " + ", ".join(
            f"{ch}={t:.6g}" for ch, t in zip(cal_channels, result.thresholds_v)
        ),
    ]
    if metadata is not None:
        # Allow multi-line user metadata.
        for line in str(metadata).splitlines():
            lines.append(f"# user_metadata: {line}")
    if result.warnings_:
        for w in result.warnings_:
            lines.append(f"# warning: {w}")
    lines.append("frame_number,sample_index")
    for row in result.frame_table:
        lines.append(f"{int(row[0])},{int(row[1])}")
    output_path.write_text("\n".join(lines) + "\n")
