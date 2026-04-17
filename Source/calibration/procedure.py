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

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from calibration.daq import DAQ, Acquisition
from calibration.display import Display


# Default time between `Display.flip()` and the start of DAQ sampling. Gives
# the monitor's pixel response a chance to settle; LCDs want this on the
# order of 100-200 ms (slow pixel ramp + PWM backlight), OLEDs need only
# a few ms. Caller can override per-call or via the orchestration functions.
DEFAULT_SETTLE_TIME_S = 0.2

# Default DAQ sampling window for each per-pattern measurement.
DEFAULT_WINDOW_S = 0.1

# Signal-to-noise ratio above which a channel is considered "live" --
# i.e. actually reading light off the screen, not a floating input.
# A channel with dynamic range < LIVENESS_NOISE_STDS * dark_noise_std is
# treated as dead. Very loose because we expect a >100x gap between
# connected and unconnected channels.
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
    """
    dark = measure_after_render(
        display, daq, lambda d: d.black(),
        settle_time=settle_time, duration=duration,
        channels=channels, sample_rate=sample_rate,
    )
    bright = measure_after_render(
        display, daq, lambda d: d.white(),
        settle_time=settle_time, duration=duration,
        channels=channels, sample_rate=sample_rate,
    )
    # Restore black so the next step doesn't leak white light into whatever
    # measurement follows.
    display.black()
    display.flip()
    return BaselineResult(dark=dark, bright=bright)
