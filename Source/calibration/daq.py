"""NI-DAQmx wrapper for photodiode analog input acquisition.

Exposes a single `DAQ` class that knows nothing about the calibration
procedure: it just connects to a device, samples a time window of
analog-input channels, and returns an `Acquisition` with the raw
voltage samples plus per-channel statistics.

Channel selection and live-channel detection are the caller's job --
they belong to the procedure layer, not the hardware wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.system import System


# Default analog input voltage range. The Monitorio board's TIA runs on a
# single 5 V supply, so signals are 0-5 V; we use bipolar +/-5 V on the DAQ
# because most NI AI ranges are symmetric around 0 and +/-5 V is the
# tightest range that contains 0-5 V.
DEFAULT_VOLTAGE_RANGE = (-5.0, 5.0)

# Default per-channel sample rate. User-overridable in DAQ constructor and
# per-acquisition.
DEFAULT_SAMPLE_RATE = 50_000.0


@dataclass(frozen=True)
class Acquisition:
    """Result of a single finite analog-input acquisition.

    samples:      (n_channels, n_samples) volts
    channels:     physical channel names in same order as samples' first axis
    sample_rate:  actual per-channel sample rate used (Hz)
    """

    samples: np.ndarray
    channels: tuple[str, ...]
    sample_rate: float

    @property
    def n_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate

    def times(self) -> np.ndarray:
        """Time vector in seconds, aligned to each sample's start."""
        return np.arange(self.n_samples) / self.sample_rate

    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=1)

    def std(self) -> np.ndarray:
        return self.samples.std(axis=1)

    def min(self) -> np.ndarray:
        return self.samples.min(axis=1)

    def max(self) -> np.ndarray:
        return self.samples.max(axis=1)


class DAQ:
    """Minimal wrapper around an NI-DAQmx device for repeated finite AI reads.

    Typical use:

        with DAQ() as daq:
            acq = daq.acquire(duration=0.1)
            print(acq.mean())

    The device, sample rate, and voltage range are set at construction.
    `acquire()` creates a fresh finite task each call; this costs a few ms
    of overhead per read but keeps the class stateless between
    acquisitions (no risk of a stale task configuration).
    """

    def __init__(
        self,
        device_name: str | None = None,
        *,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        voltage_range: tuple[float, float] = DEFAULT_VOLTAGE_RANGE,
        terminal_config: TerminalConfiguration = TerminalConfiguration.RSE,
    ):
        devices = list_devices()
        if not devices:
            raise RuntimeError(
                "No NI-DAQmx devices found. Check that the NI-DAQmx driver "
                "is installed and a device is connected."
            )
        if device_name is None:
            device_name = devices[0]
        elif device_name not in devices:
            raise ValueError(
                f"Device {device_name!r} not found. Available: {devices}"
            )

        self._device_name = device_name
        self._device = System.local().devices[device_name]
        self._sample_rate = float(sample_rate)
        self._voltage_range = voltage_range
        self._terminal_config = terminal_config
        # Physical channels usable in this terminal config. In DIFF/PSEUDO_DIFF
        # only half the AI pins are addressable as separate channels (the rest
        # are reserved as their negative-input partners), so we filter by the
        # device's own self-reported per-channel ai_term_cfgs rather than
        # assuming all pins map to all configs.
        self._all_channels = tuple(
            c.name for c in self._device.ai_physical_chans
            if terminal_config in c.ai_term_cfgs
        )

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def product_type(self) -> str:
        return self._device.product_type

    @property
    def channels(self) -> tuple[str, ...]:
        """All AI physical channels on this device, e.g. ('Dev1/ai0', ...)."""
        return self._all_channels

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def max_multi_channel_rate(self) -> float:
        """Aggregate AI sample rate across all channels (Hz)."""
        return float(self._device.ai_max_multi_chan_rate)

    def close(self):
        # Nothing to release -- tasks are created/destroyed per-acquire.
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    # -------- acquisition --------

    def acquire(
        self,
        duration: float = 0.1,
        *,
        channels: tuple[str, ...] | list[str] | None = None,
        sample_rate: float | None = None,
    ) -> Acquisition:
        """Read `duration` seconds of samples from `channels` at `sample_rate`.

        channels:    iterable of physical channel names (e.g. 'Dev1/ai0').
                     Defaults to all AI channels on this device.
        sample_rate: per-channel rate in Hz. Defaults to self.sample_rate.
                     Must satisfy n_channels * sample_rate <= device's
                     aggregate AI rate, else nidaqmx raises.

        Returns an Acquisition with voltages as a (n_channels, n_samples)
        numpy array. The last channel's last sample is included
        (i.e. n_samples == round(duration * sample_rate)).
        """
        if channels is None:
            channels = self._all_channels
        else:
            channels = tuple(channels)
        if len(channels) == 0:
            raise ValueError("channels must be non-empty")

        rate = float(sample_rate) if sample_rate is not None else self._sample_rate
        n_samples = max(2, int(round(duration * rate)))

        # Sanity check aggregate rate before we even talk to the driver,
        # so the error message names the offending knob rather than a
        # generic DAQmx error code.
        max_per_chan = self.max_multi_channel_rate / len(channels)
        if rate > max_per_chan:
            raise ValueError(
                f"sample_rate={rate} Hz exceeds per-channel max "
                f"{max_per_chan:.0f} Hz for {len(channels)} simultaneous "
                f"channels on {self._device.product_type} "
                f"(aggregate max = {self.max_multi_channel_rate:.0f} Hz)."
            )

        v_min, v_max = self._voltage_range
        buffer = np.empty((len(channels), n_samples), dtype=np.float64)

        with nidaqmx.Task() as task:
            for chan in channels:
                task.ai_channels.add_ai_voltage_chan(
                    chan,
                    terminal_config=self._terminal_config,
                    min_val=v_min,
                    max_val=v_max,
                )
            task.timing.cfg_samp_clk_timing(
                rate=rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=n_samples,
            )
            reader = AnalogMultiChannelReader(task.in_stream)
            reader.read_many_sample(
                buffer, number_of_samples_per_channel=n_samples,
                timeout=duration + 5.0,
            )

        return Acquisition(
            samples=buffer, channels=channels, sample_rate=rate,
        )

    def acquire_with_action(
        self,
        duration: float,
        action,
        *,
        channels: tuple[str, ...] | list[str] | None = None,
        sample_rate: float | None = None,
    ) -> Acquisition:
        """Start a finite AI acquisition, fire `action()`, then read the samples.

        Used when you need to trigger an external stimulus (e.g. a screen
        flip) AFTER the DAQ has begun sampling but BEFORE the samples are
        read back -- so the captured time series contains both the
        pre-stimulus baseline and the transition.

        `action` is a zero-arg callable run immediately after task.start().
        Any pre-flip delay should be inside `action` (a time.sleep() is
        fine; the hardware keeps sampling while Python sleeps).
        """
        if channels is None:
            channels = self._all_channels
        else:
            channels = tuple(channels)
        if len(channels) == 0:
            raise ValueError("channels must be non-empty")

        rate = float(sample_rate) if sample_rate is not None else self._sample_rate
        n_samples = max(2, int(round(duration * rate)))

        max_per_chan = self.max_multi_channel_rate / len(channels)
        if rate > max_per_chan:
            raise ValueError(
                f"sample_rate={rate} Hz exceeds per-channel max "
                f"{max_per_chan:.0f} Hz for {len(channels)} simultaneous "
                f"channels on {self._device.product_type}."
            )

        v_min, v_max = self._voltage_range
        buffer = np.empty((len(channels), n_samples), dtype=np.float64)

        with nidaqmx.Task() as task:
            for chan in channels:
                task.ai_channels.add_ai_voltage_chan(
                    chan,
                    terminal_config=self._terminal_config,
                    min_val=v_min,
                    max_val=v_max,
                )
            task.timing.cfg_samp_clk_timing(
                rate=rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=n_samples,
            )
            reader = AnalogMultiChannelReader(task.in_stream)
            task.start()
            action()
            reader.read_many_sample(
                buffer, number_of_samples_per_channel=n_samples,
                timeout=duration + 5.0,
            )

        return Acquisition(
            samples=buffer, channels=channels, sample_rate=rate,
        )


def list_devices() -> list[str]:
    """Return the names of every NI-DAQmx device currently visible, e.g. ['Dev1']."""
    return [d.name for d in System.local().devices]


def list_ai_channels(
    device_name: str,
    *,
    terminal_config: TerminalConfiguration | None = None,
) -> list[str]:
    """Return AI physical channel names on the named device.

    If `terminal_config` is given, only channels that support that
    configuration are returned. This matters in DIFF (and PSEUDO_DIFF):
    half the physical AI pins are not addressable as separate channels
    in those modes -- they're reserved as the negative-input partners of
    the other half. On a 32-pin X-series device, RSE/NRSE return all 32;
    DIFF returns 16 (the positive-input pins). The exact pinout is
    device-dependent, so we ask the device itself rather than assuming.
    """
    dev = System.local().devices[device_name]
    chans = list(dev.ai_physical_chans)
    if terminal_config is not None:
        chans = [c for c in chans if terminal_config in c.ai_term_cfgs]
    return [c.name for c in chans]


# CLI-friendly names for the nidaqmx TerminalConfiguration enum. Excludes
# DEFAULT because that's just "whatever the card's own default is" and
# surprises downstream -- our code should always pass an explicit
# configuration.
TERMINAL_CONFIG_CHOICES = ("RSE", "DIFF", "NRSE", "PSEUDO_DIFF")


def terminal_config_from_name(name: str) -> TerminalConfiguration:
    """Resolve a case-insensitive name to a nidaqmx TerminalConfiguration.

    Accepts: 'RSE' (referenced single-ended, default), 'DIFF'
    (differential), 'NRSE' (non-referenced single-ended), or
    'PSEUDO_DIFF'. Raises ValueError for any other string.
    """
    m = {
        "RSE": TerminalConfiguration.RSE,
        "DIFF": TerminalConfiguration.DIFF,
        "NRSE": TerminalConfiguration.NRSE,
        "PSEUDO_DIFF": TerminalConfiguration.PSEUDO_DIFF,
    }
    key = name.upper()
    if key not in m:
        raise ValueError(
            f"Unknown terminal config {name!r}; expected one of "
            f"{TERMINAL_CONFIG_CHOICES}"
        )
    return m[key]
