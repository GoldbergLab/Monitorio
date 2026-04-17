"""High-level orchestration of the Monitorio calibration procedure.

Combines `calibration.display` and `calibration.daq` to run the real
measurements. This module grows piece-by-piece; at the moment it exposes
only the baseline characterization step (piece 3):

  - Measure noise floor on a fully black screen.
  - Measure response on a fully white screen.
  - Derive per-channel dynamic range, SNR, and a liveness flag.

Later pieces will add structured-light localization, centroid refinement,
rise-time measurement, and diameter sweep.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from calibration import gray
from calibration.daq import DAQ, Acquisition
from calibration.display import Display


# Default time between `Display.flip()` and the start of DAQ sampling. Gives
# the monitor's pixel response a chance to settle; LCDs want this on the
# order of 100-200 ms (slow pixel ramp + PWM backlight), OLEDs need only
# a few ms. Caller can override per-call or via the orchestration functions.
DEFAULT_SETTLE_TIME_S = 0.2

# Default DAQ sampling window for each per-pattern measurement.
DEFAULT_WINDOW_S = 0.1

# Default per-channel sample rate for DC-level measurements (baselines,
# localization, crosstalk). At 50 kHz per channel the PCIe-6343's mux has
# only ~2 us per channel to settle between neighbors, which leaves a few
# mV of residue from the previous channel on even a low-impedance input.
# 5 kHz gives 10x the settle budget and drops the residue below the
# noise floor, making liveness/localization results insensitive to how
# many channels are simultaneously scanned. Rise-time measurements don't
# use this default -- they need high rate and are run single-channel to
# sidestep the mux entirely.
DEFAULT_DC_SAMPLE_RATE = 5000.0

# Signal-to-noise ratio above which a channel is considered "live" --
# i.e. actually reading light off the screen, not a floating input.
# A channel with dynamic range < LIVENESS_NOISE_STDS * dark_noise_std is
# treated as dead. Very loose because at DEFAULT_DC_SAMPLE_RATE we expect
# a >1000x gap between connected channels and both unconnected channels
# and any residual mux ghosting.
DEFAULT_LIVENESS_THRESHOLD_STDS = 10.0


def measure_after_render(
    display: Display,
    daq: DAQ,
    draw: Callable[[Display], None],
    *,
    settle_time: float = DEFAULT_SETTLE_TIME_S,
    duration: float = DEFAULT_WINDOW_S,
    channels: tuple[str, ...] | list[str] | None = None,
    sample_rate: float | None = None,
) -> Acquisition:
    """Render a pattern, wait for pixel response to settle, then sample the DAQ.

    draw:        callable that issues Display draw commands. Must not call
                 flip() itself -- this function does the flip after draw.
    settle_time: seconds to wait between flip() and starting sampling.
    duration:    seconds of DAQ sampling.

    The monitor contents on return are whatever `draw` produced.
    """
    draw(display)
    display.flip()
    if settle_time > 0:
        time.sleep(settle_time)
    return daq.acquire(duration=duration, channels=channels, sample_rate=sample_rate)


@dataclass(frozen=True)
class BaselineResult:
    """Dark/bright baseline measurement, plus derived per-channel stats."""

    dark: Acquisition
    bright: Acquisition

    @property
    def channels(self) -> tuple[str, ...]:
        return self.dark.channels

    def dark_mean(self) -> np.ndarray:
        return self.dark.mean()

    def dark_std(self) -> np.ndarray:
        return self.dark.std()

    def bright_mean(self) -> np.ndarray:
        return self.bright.mean()

    def dynamic_range(self) -> np.ndarray:
        """Per-channel (bright_mean - dark_mean) in volts."""
        return self.bright.mean() - self.dark.mean()

    def snr(self) -> np.ndarray:
        """Per-channel dynamic_range / dark_std.

        dark_std is clamped at a small positive floor so a pathologically
        quiet floating input doesn't produce an Inf.
        """
        noise = np.maximum(self.dark_std(), 1e-9)
        return self.dynamic_range() / noise

    def liveness(
        self, threshold_stds: float = DEFAULT_LIVENESS_THRESHOLD_STDS,
    ) -> np.ndarray:
        """Bool mask: which channels respond to the screen state.

        A channel is "live" iff its dynamic range exceeds threshold_stds
        multiples of its own dark-noise std. With threshold_stds=10 and
        the >100x signal/noise gap we see between connected and floating
        channels, this is effectively binary.
        """
        return self.dynamic_range() > threshold_stds * self.dark_std()


# Finest Gray-code stripe bit used by localize_coarse(). Bit k produces
# 2**k -pixel stripes; if this drops below the photodiode's effective
# diameter on the screen, each PD averages across multiple stripes and
# the bit decode becomes random. A 1206 PD on a typical HD monitor spans
# roughly 10-20 pixels, so k_min=5 (32-px stripes) gives comfortable
# headroom. Localization uncertainty is +/- 2**(k_min-1) pixels, refined
# by the bar sweep in refine_locations().
DEFAULT_K_MIN = 5

# Settle time used by refine_locations()'s bar sweep. Each step changes
# only two columns (new column turns on, old column turns off) relative
# to the previous bar position, so the monitor's pixel-response ramp is
# much shorter than in the baselines/localization steps where the whole
# screen changes. 50 ms covers even slow LCDs; OLED would be happy at
# ~5 ms.
DEFAULT_REFINE_SETTLE_TIME_S = 0.05

# Default bar thickness for the fine sweep. Thinner than the PD's
# effective diameter so the per-position response looks like a clean
# trapezoid with a peak (not a long plateau), giving the weighted
# centroid a sharper maximum to lock onto.
DEFAULT_REFINE_BAR_WIDTH_PX = 4

# Extra pixels added to each PD's coarse +/- uncertainty window when the
# bar sweep runs, on both sides. Sized to be larger than the PD's
# expected sensitive radius so the sweep captures clean dark baseline
# on both flanks, not just the rising response. Without this margin, a
# PD whose true center sits near the edge of the coarse window sees
# the bar already overlapping its field from the very first sweep
# position, biasing the centroid and losing the left-flank FWHM.
# 16 px covers 1206 photodiodes on typical HD pixel pitches.
DEFAULT_REFINE_MARGIN_PX = 16


def characterize_baselines(
    display: Display,
    daq: DAQ,
    *,
    settle_time: float = DEFAULT_SETTLE_TIME_S,
    duration: float = DEFAULT_WINDOW_S,
    channels: tuple[str, ...] | list[str] | None = None,
    sample_rate: float | None = None,
) -> BaselineResult:
    """Measure per-channel noise floor (black screen) and full-scale response (white screen).

    Returns a BaselineResult from which the caller can derive dynamic
    range, SNR, and a liveness mask. Leaves the screen black when done so
    subsequent measurements start from a defined state.

    sample_rate defaults to DEFAULT_DC_SAMPLE_RATE (slower than the DAQ's
    default) because DC measurements don't benefit from fast scanning and
    the slower rate eliminates multiplexer settling crosstalk that would
    otherwise show up as small ghost responses on unconnected channels.
    """
    rate = sample_rate if sample_rate is not None else DEFAULT_DC_SAMPLE_RATE
    dark = measure_after_render(
        display, daq, lambda d: d.black(),
        settle_time=settle_time, duration=duration,
        channels=channels, sample_rate=rate,
    )
    bright = measure_after_render(
        display, daq, lambda d: d.white(),
        settle_time=settle_time, duration=duration,
        channels=channels, sample_rate=rate,
    )
    # Restore black so the next step doesn't leak white light into whatever
    # measurement follows.
    display.black()
    display.flip()
    return BaselineResult(dark=dark, bright=bright)


@dataclass(frozen=True)
class CoarseLocations:
    """Per-channel (x, y) pixel localization from structured-light patterns.

    x_pixels, y_pixels are integer pixel coordinates -- block centers from
    the Gray-code decode. True PD center lies within +/- uncertainty_px of
    the reported value (uniform distribution, assuming the PD is a point).

    min_confidence: per-channel lowest bit-decode confidence across all
    patterns, normalized to [0, 1] where 1 is cleanly above or below
    threshold and 0 is exactly at threshold. Channels with values near 0
    had at least one pattern where the PD straddled a stripe boundary;
    the bit could have gone either way, and the reported coordinate may
    be off by a block on that axis.
    """

    channels: tuple[str, ...]
    x_pixels: np.ndarray
    y_pixels: np.ndarray
    uncertainty_px: int
    min_confidence: np.ndarray


def localize_coarse(
    display: Display,
    daq: DAQ,
    baseline: BaselineResult,
    *,
    k_min: int = DEFAULT_K_MIN,
    channels: tuple[str, ...] | list[str] | None = None,
    settle_time: float = DEFAULT_SETTLE_TIME_S,
    duration: float = DEFAULT_WINDOW_S,
    sample_rate: float | None = None,
) -> CoarseLocations:
    """Locate each live photodiode to within 2**k_min pixels via structured light.

    For each axis, displays ceil(log2(axis_length)) - k_min + 1 Gray-coded
    stripe patterns at successively finer bit widths. Each PD's mean
    response to each pattern is thresholded at the midpoint of its
    baseline (dark_mean + bright_mean) / 2 to recover one bit of its
    Gray-coded position. The stack of bits is decoded into a block index,
    and the PD is reported at the center of that block.

    channels: which channels to localize. Defaults to baseline.liveness() --
        i.e. only channels that showed a real dark-to-bright response.
    k_min: finest pattern bit. See DEFAULT_K_MIN above for tuning.
    """
    rate = sample_rate if sample_rate is not None else DEFAULT_DC_SAMPLE_RATE

    # Pick channels to localize.
    if channels is None:
        live_mask = baseline.liveness()
        target_channels = tuple(
            c for c, live in zip(baseline.channels, live_mask) if live
        )
    else:
        target_channels = tuple(channels)
    if not target_channels:
        raise ValueError("No channels to localize (baseline found no live channels)")

    # Per-target dark/bright/threshold.
    baseline_idx = [baseline.channels.index(c) for c in target_channels]
    dark_m = baseline.dark_mean()[baseline_idx]
    bright_m = baseline.bright_mean()[baseline_idx]
    thresholds = (dark_m + bright_m) / 2.0
    half_range = np.maximum((bright_m - dark_m) / 2.0, 1e-9)

    n_live = len(target_channels)

    def scan_axis(axis: str, length: int):
        """Return (pixel_positions, min_confidence) for one axis."""
        K = math.ceil(math.log2(length)) - 1
        if K < k_min:
            raise ValueError(
                f"axis length {length} is too short for k_min={k_min} "
                f"(requires at least 2**{k_min + 1}={2 ** (k_min + 1)} pixels)"
            )
        n_bits = K - k_min + 1
        bits = np.zeros((n_live, n_bits), dtype=np.int64)
        confidences = np.zeros((n_live, n_bits))

        # Iterate coarsest bit (K) to finest (k_min). idx 0 is MSB of G_top.
        for idx, bit in enumerate(range(K, k_min - 1, -1)):
            acq = measure_after_render(
                display, daq,
                lambda d, b=bit: d.gray_stripes(axis, b),
                settle_time=settle_time, duration=duration,
                channels=target_channels, sample_rate=rate,
            )
            resp = acq.mean()  # (n_live,)
            bits[:, idx] = (resp > thresholds).astype(np.int64)
            confidences[:, idx] = np.abs(resp - thresholds) / half_range

        # Pack bits into G_top (idx 0 is MSB, idx n_bits-1 is LSB).
        g_top = np.zeros(n_live, dtype=np.int64)
        for idx in range(n_bits):
            g_top |= bits[:, idx] << (n_bits - 1 - idx)

        # G_top == grayEncode(block_index) -- see the derivation in
        # calibration.gray's module docstring.
        block_indices = gray.decode(g_top, n_bits=n_bits)

        # Report the center of each 2**k_min-wide block.
        positions = block_indices * (1 << k_min) + (1 << (k_min - 1))
        positions = np.clip(positions, 0, length - 1)

        return positions.astype(np.int64), confidences.min(axis=1)

    x_pixels, x_conf = scan_axis("x", display.width)
    y_pixels, y_conf = scan_axis("y", display.height)

    # Leave the screen black so subsequent steps start from a defined state.
    display.black()
    display.flip()

    return CoarseLocations(
        channels=target_channels,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        uncertainty_px=(1 << k_min) // 2,
        min_confidence=np.minimum(x_conf, y_conf),
    )


@dataclass(frozen=True)
class FineLocations:
    """Per-channel sub-pixel PD centers from a bar-sweep centroid refinement.

    x_pixels, y_pixels are floats (sub-pixel). A value of NaN means the
    refinement failed for that channel on that axis (typically: no clear
    peak was seen inside the coarse-location window).

    x_fwhm_px, y_fwhm_px are integer pixel counts at each PD's response
    above half-max on the respective axis. Useful as an estimate of the
    PD's sensitive diameter, which downstream steps (diameter sweep,
    circle placement) can seed from.

    x_sweeps, y_sweeps carry the raw bar-sweep data per channel as
    (positions_px, responses_volts) pairs -- the per-channel window
    slice used to compute that channel's centroid. Retained for
    diagnostic plotting.
    """

    channels: tuple[str, ...]
    x_pixels: np.ndarray
    y_pixels: np.ndarray
    x_fwhm_px: np.ndarray
    y_fwhm_px: np.ndarray
    x_sweeps: tuple[tuple[np.ndarray, np.ndarray], ...]
    y_sweeps: tuple[tuple[np.ndarray, np.ndarray], ...]


def refine_locations(
    display: Display,
    daq: DAQ,
    baseline: BaselineResult,
    coarse: CoarseLocations,
    *,
    bar_width: int = DEFAULT_REFINE_BAR_WIDTH_PX,
    margin_px: int = DEFAULT_REFINE_MARGIN_PX,
    settle_time: float = DEFAULT_REFINE_SETTLE_TIME_S,
    duration: float = DEFAULT_WINDOW_S,
    sample_rate: float | None = None,
    peak_fraction: float = 0.1,
) -> FineLocations:
    """Refine coarse (x, y) estimates to sub-pixel via a bar-sweep centroid.

    For each axis, sweeps a `bar_width`-px bar across the UNION of each
    PD's coarse +/- (uncertainty + margin_px) window, acquires every
    live PD's response at every bar position, then extracts each PD's
    center from its own window slice by a noise-rejected weighted
    centroid.

    margin_px: extra pixels added to each window on both sides beyond
        coarse.uncertainty_px. Needs to be larger than the PD's
        expected sensitive radius so the sweep sees clean dark baseline
        on each flank -- otherwise the bar is already overlapping the
        PD from the first sweep position, biasing the centroid and
        truncating the left-flank FWHM.

    The centroid weighting is max(response - peak_fraction * peak, 0),
    which zeros out readings below `peak_fraction` of the peak height
    on the positive side while leaving the peak region intact -- so a
    flat background doesn't pull the center off.

    Bar width should be smaller than the PD's effective diameter; the
    response curve is then a clean trapezoid with a peak at the PD
    center, giving the centroid a sharp maximum. Wider bars also work
    (flat-top centroid is still centered) but give a flatter peak.
    """
    rate = sample_rate if sample_rate is not None else DEFAULT_DC_SAMPLE_RATE

    # Match target_channels to baseline so we can pull per-channel darks.
    baseline_idx = [baseline.channels.index(c) for c in coarse.channels]
    dark_m = baseline.dark_mean()[baseline_idx]
    dynamic_range = baseline.dynamic_range()[baseline_idx]
    n_live = len(coarse.channels)
    half_width = int(coarse.uncertainty_px) + int(margin_px)

    def sweep_axis(axis: str, length: int, coarse_pos: np.ndarray):
        """Return (sub-pixel centers, FWHM-in-pixels) per channel for one axis."""
        # Per-PD windows, clipped to screen bounds. half_width includes both
        # the coarse-localization uncertainty and a margin for the PD's own
        # sensitive radius.
        windows = [
            (max(0, int(c) - half_width), min(length - 1, int(c) + half_width))
            for c in coarse_pos
        ]
        # Union of windows as a sorted list of unique positions to display.
        positions_set: set[int] = set()
        for lo, hi in windows:
            positions_set.update(range(lo, hi + 1))
        positions = sorted(positions_set)
        if not positions:
            return (
                np.full(n_live, np.nan),
                np.zeros(n_live, dtype=np.int64),
            )

        responses = np.zeros((n_live, len(positions)), dtype=np.float64)
        half_bar = bar_width // 2
        for i, pos in enumerate(positions):
            # Draw bar centered on `pos`. The bar's left/top edge is pos-half_bar.
            start = pos - half_bar
            if axis == "x":
                draw = lambda d, s=start: d.vertical_bar(s, bar_width)
            else:
                draw = lambda d, s=start: d.horizontal_bar(s, bar_width)
            acq = measure_after_render(
                display, daq, draw,
                settle_time=settle_time, duration=duration,
                channels=coarse.channels, sample_rate=rate,
            )
            # Subtract per-channel dark so "response" is the above-dark signal.
            responses[:, i] = acq.mean() - dark_m

        positions_arr = np.asarray(positions, dtype=np.float64)
        centers = np.full(n_live, np.nan, dtype=np.float64)
        fwhms = np.zeros(n_live, dtype=np.int64)
        per_pd_sweeps: list[tuple[np.ndarray, np.ndarray]] = []

        # Minimum peak height (in volts above dark) to trust the refinement:
        # require the PD to have lit up to at least 10% of its baseline range.
        min_peak = 0.10 * dynamic_range

        for j in range(n_live):
            lo, hi = windows[j]
            mask = (positions_arr >= lo) & (positions_arr <= hi)
            pos_j = positions_arr[mask]
            resp_j = responses[j, mask]
            per_pd_sweeps.append((pos_j.copy(), resp_j.copy()))
            if resp_j.size == 0:
                continue
            peak = float(resp_j.max())
            if peak < min_peak[j]:
                continue  # leave as NaN

            # Noise-rejected weighted centroid.
            weights = np.maximum(resp_j - peak_fraction * peak, 0.0)
            wsum = weights.sum()
            if wsum > 0:
                centers[j] = float((pos_j * weights).sum() / wsum)

            # FWHM (pixel count above half-peak; integer estimate of PD diameter).
            fwhms[j] = int(np.count_nonzero(resp_j >= peak / 2.0))

        return centers, fwhms, tuple(per_pd_sweeps)

    x_fine, x_fwhm, x_sweeps = sweep_axis("x", display.width, coarse.x_pixels)
    y_fine, y_fwhm, y_sweeps = sweep_axis("y", display.height, coarse.y_pixels)

    display.black()
    display.flip()

    return FineLocations(
        channels=coarse.channels,
        x_pixels=x_fine,
        y_pixels=y_fine,
        x_fwhm_px=x_fwhm,
        y_fwhm_px=y_fwhm,
        x_sweeps=x_sweeps,
        y_sweeps=y_sweeps,
    )


# High-rate single-channel default for rise-time measurements. Single
# channel sidesteps the multiplexer, so we can run near the card's
# single-channel max without any ghost worry. 50 kHz gives 20 us
# sample spacing -- fine for any LCD (~1-10 ms rise times) and adequate
# for OLED (~100 us). Bump up for pathological-fast OLEDs if needed.
DEFAULT_RISE_SAMPLE_RATE = 50_000.0

# Window captured per transition (seconds). Includes the pre-flip
# baseline period, the actual transition, and a post-transition tail
# from which the plateau mean is measured.
DEFAULT_RISE_WINDOW_S = 0.1

# How long into the acquisition to delay before triggering the screen
# flip. Gives enough pre-flip samples to estimate a clean baseline.
DEFAULT_RISE_PRE_FLIP_S = 0.02


# Minimum duration the trace must remain past a 10/90 threshold for a
# crossing to count. Filters sub-millisecond spikes (scan-line
# artifacts, brief OLED refresh glitches) that would otherwise trigger
# a false early 10% crossing and dominate the reported transition
# duration. 1 ms safely exceeds any single-sample transient while
# staying far below typical monitor rise/fall times (tens of us on
# OLED, single ms on LCD).
DEFAULT_RISE_SUSTAIN_S = 0.001


@dataclass(frozen=True)
class RiseTimeResult:
    """Per-channel transition timings for black<->white flips at each PD.

    `*_duration_s` is the 10-90% transition time itself -- how long the
    PD spends ramping between states. Bounded below by the monitor's
    actual pixel response; a sensible metric for "how blurry is a bit
    flip at a sampling DAQ."

    `*_latency_s` is the time from the start of the capture window
    (which begins the moment the DAQ task starts) to the sustained 10%
    crossing. This folds in (a) the pre-flip delay, (b) pygame.flip()
    overhead, (c) the monitor's frame pipeline. On a 60 Hz display it's
    typically quantized to multiples of 16.7 ms. The difference between
    rise_latency and fall_latency is a useful asymmetry diagnostic.

    Traces are retained so the caller can plot them for sanity check.
    """

    channels: tuple[str, ...]
    rise_duration_s: np.ndarray
    fall_duration_s: np.ndarray
    rise_latency_s: np.ndarray
    fall_latency_s: np.ndarray
    sample_rate: float
    pre_flip_s: float
    sustain_s: float
    rise_trace: np.ndarray  # (n_channels, n_samples) volts
    fall_trace: np.ndarray


def _first_sustained_crossing(mask: np.ndarray, n_sustain: int) -> int:
    """Return the first index i where mask[i : i+n_sustain] is all True, else -1.

    Vectorized via a cumulative-sum window count: counts[i] is the number
    of True values in mask[i : i+n_sustain]; the earliest i where that
    equals n_sustain is our sustained-crossing index.
    """
    if n_sustain <= 1:
        return int(np.argmax(mask)) if mask.any() else -1
    int_mask = mask.astype(np.int64)
    csum = np.concatenate(([0], np.cumsum(int_mask)))
    counts = csum[n_sustain:] - csum[:-n_sustain]
    valid = np.where(counts == n_sustain)[0]
    return int(valid[0]) if valid.size > 0 else -1


def _detect_transition(
    trace: np.ndarray, sample_rate: float, sustain_s: float,
) -> tuple[float, float]:
    """Return (latency_s, duration_10_90_s) for a trace with one transition.

    Normalizes trace to [0, 1] on (baseline mean, plateau mean) so the
    same code handles rise and fall.

    The 10% crossing uses `sustain_s` to reject sub-frame dips from
    baseline (e.g. scan-line artifacts) that would otherwise fake out
    the detector. The 90% crossing takes the FIRST sample at or past
    90% on or after the 10% crossing -- no sustain. Sub-frame noise
    almost never rises from baseline all the way to 90% of plateau, so
    sustained matching isn't buying us anything at that level; and on
    some displays the initial rising edge briefly overshoots above 90%
    before decaying as part of the frame's normal brightness ramp,
    which would leave a sustained search stuck hunting for the next
    refresh peak.
    """
    n = trace.size
    n_pre = max(20, n // 20)    # first 5% -> baseline
    n_post = max(20, n // 10)   # last 10% -> plateau
    baseline = float(trace[:n_pre].mean())
    plateau = float(trace[-n_post:].mean())
    span = plateau - baseline
    if abs(span) < 0.01:  # <10 mV -- no real transition captured
        return float("nan"), float("nan")

    norm = (trace - baseline) / span  # 0 at baseline, 1 at plateau
    sustain_n = max(1, int(round(sustain_s * sample_rate)))
    i10 = _first_sustained_crossing(norm >= 0.1, sustain_n)
    if i10 < 0:
        return float("nan"), float("nan")
    # First 90%+ sample at or after the sustained 10% crossing. Allow
    # i90 == i10: that just means the transition took less than one
    # sample period to complete (a legitimate, resolution-limited 0).
    post_i10 = norm[i10:] >= 0.9
    if not post_i10.any():
        return float("nan"), float("nan")
    i90 = i10 + int(np.argmax(post_i10))
    return i10 / sample_rate, (i90 - i10) / sample_rate


def measure_rise_times(
    display: Display,
    daq: DAQ,
    fine: FineLocations,
    *,
    radius_padding_px: int = 4,
    duration: float = DEFAULT_RISE_WINDOW_S,
    pre_flip_s: float = DEFAULT_RISE_PRE_FLIP_S,
    sample_rate: float = DEFAULT_RISE_SAMPLE_RATE,
    settle_time: float = DEFAULT_SETTLE_TIME_S,
    sustain_s: float = DEFAULT_RISE_SUSTAIN_S,
) -> RiseTimeResult:
    """Measure 10-90% rise and fall time at each PD's fine position.

    For each channel in `fine.channels`:
      1. Draw a circle of matching polarity over the PD (radius derived
         from the PD's FWHM plus `radius_padding_px` so the circle
         saturates the PD).
      2. Settle.
      3. Start a single-channel, high-rate finite acquisition.
      4. After pre_flip_s, flip to the opposite polarity.
      5. Read the trace; detect 10-90% transition time.
    Each PD gets one rise measurement (black->white) and one fall
    measurement (white->black); single-channel acquisition avoids mux
    crosstalk at high rate.
    """
    import time as _time  # local import: not used by the module elsewhere

    n_chan = len(fine.channels)
    rise_durations = np.full(n_chan, np.nan)
    fall_durations = np.full(n_chan, np.nan)
    rise_latencies = np.full(n_chan, np.nan)
    fall_latencies = np.full(n_chan, np.nan)
    n_samples = int(round(duration * sample_rate))
    rise_traces = np.zeros((n_chan, n_samples), dtype=np.float64)
    fall_traces = np.zeros((n_chan, n_samples), dtype=np.float64)

    for i, chan in enumerate(fine.channels):
        xi, yi = fine.x_pixels[i], fine.y_pixels[i]
        if np.isnan(xi) or np.isnan(yi):
            continue
        cx, cy = int(round(float(xi))), int(round(float(yi)))
        radius = int(max(fine.x_fwhm_px[i], fine.y_fwhm_px[i]) / 2) + radius_padding_px

        # --- Rise: screen is fully black, flip to "white disk at PD". ---
        display.black()
        display.flip()
        _time.sleep(settle_time)

        def flip_to_white(cx=cx, cy=cy, r=radius):
            _time.sleep(pre_flip_s)
            display.circle(cx, cy, r)  # bg=0 fg=255
            display.flip()

        acq = daq.acquire_with_action(
            duration, flip_to_white,
            channels=(chan,), sample_rate=sample_rate,
        )
        trace = acq.samples[0]
        n_eff = min(trace.size, n_samples)
        rise_traces[i, :n_eff] = trace[:n_eff]
        rise_latencies[i], rise_durations[i] = _detect_transition(
            trace, sample_rate, sustain_s,
        )

        # --- Fall: white disk on black, flip to all-black. ---
        display.circle(cx, cy, radius)
        display.flip()
        _time.sleep(settle_time)

        def flip_to_black():
            _time.sleep(pre_flip_s)
            display.black()
            display.flip()

        acq = daq.acquire_with_action(
            duration, flip_to_black,
            channels=(chan,), sample_rate=sample_rate,
        )
        trace = acq.samples[0]
        n_eff = min(trace.size, n_samples)
        fall_traces[i, :n_eff] = trace[:n_eff]
        fall_latencies[i], fall_durations[i] = _detect_transition(
            trace, sample_rate, sustain_s,
        )

    display.black()
    display.flip()

    return RiseTimeResult(
        channels=fine.channels,
        rise_duration_s=rise_durations,
        fall_duration_s=fall_durations,
        rise_latency_s=rise_latencies,
        fall_latency_s=fall_latencies,
        sample_rate=sample_rate,
        pre_flip_s=pre_flip_s,
        sustain_s=sustain_s,
        rise_trace=rise_traces,
        fall_trace=fall_traces,
    )


# --- bit-circle radius selection + crosstalk verification ---------------

# Radius chosen so the circle a) saturates the PD's sensitive field and
# b) stays well clear of the next PD's field. Saturation needs radius at
# least ~FWHM/2; to cover the tails we bump that up by saturate_factor.
# Circle mustn't approach a neighbor closer than `neighbor_clearance`
# fraction of the neighbor distance, which for a typical Monitorio layout
# with ~64 px PD spacing and ~24 px FWHM leaves a comfortable gap.
DEFAULT_SATURATE_FACTOR = 1.5
DEFAULT_NEIGHBOR_CLEARANCE = 0.4

# Fraction of own dynamic range that off-target channels may show before
# a crosstalk result is flagged unacceptable. 5% means a lit PD can
# bleed into a neighbor at up to 5% of full bright before the script
# warns -- well above the mux-ghost floor we established earlier
# (<<0.1% at 5 kHz DC rate) but below anything that would meaningfully
# corrupt a thresholded bit decode.
DEFAULT_CROSSTALK_THRESHOLD = 0.05


# Default "how far out the PD still sees light" threshold. A position
# at which the bar-sweep response drops below this fraction of the PD's
# baseline dynamic range is considered outside the PD's sensitive field.
# 1% means a white pixel there contributes <=1% of a full-bright
# reading to the PD -- negligible against the ~75% bit-on vs. ~0% bit-off
# separation we see in the crosstalk measurement.
DEFAULT_BACKGROUND_RESPONSE_THRESHOLD = 0.01


def pick_background_radius_px(
    fine: FineLocations,
    baseline: BaselineResult,
    *,
    response_threshold: float = DEFAULT_BACKGROUND_RESPONSE_THRESHOLD,
    extra_margin_px: int = 4,
) -> np.ndarray:
    """Pick the black-background disk radius per PD.

    The background disk is the larger black region drawn behind each
    bit circle in addVideoSyncTags, whose job is to prevent the
    underlying video frame from bleeding into the photodiode during a
    bit-off read. This function sizes it to be large enough that the
    PD doesn't see past its edge.

    Procedure: use the fine-refinement bar sweeps (already captured in
    `fine.x_sweeps` / `fine.y_sweeps`) to find, per axis, the max
    distance from the PD's fine center at which the sweep response
    still exceeds `response_threshold * baseline.dynamic_range`. Take
    the max across axes, add `extra_margin_px` for safety.

    Returns an int array, one radius per channel. A value of 0 means
    the sweep never exceeded the threshold (typically because the PD
    wasn't responsive or the sweep window was entirely past the PD).

    Caveats:
    - The sweep window extends only `coarse.uncertainty_px + margin_px`
      from the coarse position; if the PD has response tails beyond
      that, this function reports the window edge, understating the
      true background radius. On a typical rig (FWHM ~24 px, sweep
      half-width 32 px) the tails have decayed to ~0 well within the
      window.
    - extra_margin_px protects against small measurement noise at the
      threshold crossing and minor PD misalignment vs. screen pixels.
    """
    baseline_idx = [baseline.channels.index(c) for c in fine.channels]
    dyn_range = baseline.dynamic_range()[baseline_idx]

    n = len(fine.channels)
    radii = np.zeros(n, dtype=np.int64)

    for i in range(n):
        thresh_abs = float(response_threshold * dyn_range[i])
        half_widths: list[float] = []
        for (center, sweeps) in (
            (fine.x_pixels[i], fine.x_sweeps),
            (fine.y_pixels[i], fine.y_sweeps),
        ):
            if np.isnan(center):
                continue
            pos, resp = sweeps[i]
            above = resp > thresh_abs
            if not above.any():
                continue
            dists = np.abs(pos[above] - float(center))
            half_widths.append(float(dists.max()))
        if not half_widths:
            radii[i] = 0
            continue
        radii[i] = int(np.ceil(max(half_widths))) + int(extra_margin_px)
    return radii


def pick_bit_radius_px(
    fine: FineLocations,
    *,
    saturate_factor: float = DEFAULT_SATURATE_FACTOR,
    neighbor_clearance: float = DEFAULT_NEIGHBOR_CLEARANCE,
    min_radius_px: int = 4,
) -> np.ndarray:
    """Pick a bit-circle radius (px) per PD based on FWHM and neighbor spacing.

    saturate_factor: multiplier on FWHM/2. 1.5 gives roughly 0.75 x FWHM,
        which in practice saturates a Gaussian-like PD response (covers
        it past the ~10% shoulders).
    neighbor_clearance: circle radius may be at most this fraction of
        the nearest-neighbor distance. 0.4 leaves a ~20% gap between
        adjacent circles.

    Returns int array of shape (n_channels,). When the two bounds
    conflict (PDs too densely packed), the neighbor-clearance bound
    wins and the returned radius may undersaturate the PD.
    """
    n = len(fine.channels)
    positions = np.stack([fine.x_pixels, fine.y_pixels], axis=1)  # (n, 2)
    fwhms = np.maximum(fine.x_fwhm_px, fine.y_fwhm_px).astype(np.float64)
    target = np.ceil(fwhms / 2.0 * saturate_factor).astype(np.int64)

    radii = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if np.isnan(positions[i, 0]) or np.isnan(positions[i, 1]):
            radii[i] = min_radius_px
            continue
        if n == 1:
            upper = target[i]
        else:
            others = np.delete(positions, i, axis=0)
            # Skip any neighbor with NaN position so it doesn't pollute the min.
            valid = ~np.isnan(others).any(axis=1)
            others = others[valid]
            if others.shape[0] == 0:
                upper = target[i]
            else:
                dists = np.sqrt(((others - positions[i]) ** 2).sum(axis=1))
                upper = int(np.floor(dists.min() * neighbor_clearance))
        radii[i] = max(min_radius_px, min(int(target[i]), upper))
    return radii


@dataclass(frozen=True)
class CrosstalkResult:
    """Normalized crosstalk matrix after lighting each PD's chosen circle.

    matrix[i, j] is channel j's dark-subtracted response, normalized to
    its own (baseline bright - baseline dark), when PD i's circle is
    illuminated. Diagonal entries are ideally close to 1.0 (own PD fully
    saturated); off-diagonal entries measure how much a lit PD spills
    into every other channel.

    max_crosstalk[i] is the max absolute off-diagonal entry in row i --
    the worst channel-to-channel leak for circle i.

    `acceptable` is True iff every max_crosstalk < warn_threshold.
    """

    channels: tuple[str, ...]
    radii_px: np.ndarray
    matrix: np.ndarray
    max_crosstalk: np.ndarray
    warn_threshold: float
    acceptable: bool


def measure_crosstalk(
    display: Display,
    daq: DAQ,
    fine: FineLocations,
    baseline: BaselineResult,
    *,
    radii_px: np.ndarray | None = None,
    settle_time: float = DEFAULT_SETTLE_TIME_S,
    duration: float = DEFAULT_WINDOW_S,
    sample_rate: float | None = None,
    warn_threshold: float = DEFAULT_CROSSTALK_THRESHOLD,
) -> CrosstalkResult:
    """Light each PD's chosen circle in turn and read every channel.

    Fast sanity check that the picked circle radii don't cause neighbor
    bleed. One measurement per PD (n_live total), so a full four-PD rig
    finishes in well under 10 seconds.
    """
    import time as _time

    rate = sample_rate if sample_rate is not None else DEFAULT_DC_SAMPLE_RATE
    if radii_px is None:
        radii_px = pick_bit_radius_px(fine)
    radii = np.asarray(radii_px, dtype=np.int64)

    target_channels = fine.channels
    n = len(target_channels)
    baseline_idx = [baseline.channels.index(c) for c in target_channels]
    dark_m = baseline.dark_mean()[baseline_idx]
    dyn_range = np.maximum(baseline.dynamic_range()[baseline_idx], 1e-9)

    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xi, yi = fine.x_pixels[i], fine.y_pixels[i]
        if np.isnan(xi) or np.isnan(yi):
            matrix[i, :] = np.nan
            continue
        cx, cy = int(round(float(xi))), int(round(float(yi)))
        display.circle(cx, cy, int(radii[i]))
        display.flip()
        _time.sleep(settle_time)
        acq = daq.acquire(duration=duration, channels=target_channels, sample_rate=rate)
        response = acq.mean() - dark_m  # per channel, V above dark
        matrix[i, :] = response / dyn_range

    # Off-diagonal max per illuminated PD (absolute so a suspiciously
    # negative number counts too).
    max_crosstalk = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = np.abs(matrix[i].copy())
        row[i] = 0.0
        max_crosstalk[i] = float(row.max()) if row.size > 0 else 0.0

    acceptable = bool(np.all(max_crosstalk < warn_threshold))

    display.black()
    display.flip()

    return CrosstalkResult(
        channels=target_channels,
        radii_px=radii,
        matrix=matrix,
        max_crosstalk=max_crosstalk,
        warn_threshold=warn_threshold,
        acceptable=acceptable,
    )
