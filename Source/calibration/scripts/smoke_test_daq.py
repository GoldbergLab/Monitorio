"""Manual smoke test for calibration.daq.

Phase 1: enumerate devices, list AI channels, and take a short baseline
         acquisition with no stimulus -- reports the noise floor and
         unconnected-channel behavior.
Phase 2: (optional) acquire for a longer window while the operator
         covers/uncovers photodiodes by hand, to verify that the DAQ
         sees signal variation in response to light.

Usage:
    venv\\Scripts\\python Source\\calibration\\scripts\\smoke_test_daq.py [device_name] [n_channels]

If device_name is omitted, the first detected device is used.
If n_channels is omitted, the script uses as many AI channels as can run
at the default sample rate without exceeding the DAQ's aggregate AI rate
(i.e. floor(aggregate_rate / sample_rate)), capped by the physical
channel count. Always in single-ended (RSE) mode.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the calibration package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from calibration.daq import DAQ, list_devices, list_ai_channels


def _print_stats_table(label: str, channels, stats_rows):
    """stats_rows: iterable of (name, value_array) pairs, one row per stat."""
    print(f"\n{label}")
    header = "  channel".ljust(14) + "".join(f"{n:>14}" for n, _ in stats_rows)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, ch in enumerate(channels):
        row = f"  {ch:<12}" + "".join(
            f"{vals[i]:>14.5f}" for _, vals in stats_rows
        )
        print(row)


def main() -> int:
    devices = list_devices()
    print(f"Detected {len(devices)} NI-DAQmx device(s):")
    for d in devices:
        chans = list_ai_channels(d)
        print(f"  {d}  (AI channels: {len(chans)})")
    if not devices:
        print("\nNo device found. Is the NI-DAQmx driver installed and the DAQ connected?")
        return 1

    device_name = sys.argv[1] if len(sys.argv) > 1 else devices[0]
    all_chans = list_ai_channels(device_name)

    with DAQ(device_name) as daq:
        # Default: as many channels as the aggregate AI rate allows at the
        # DAQ's default per-channel sample rate, capped at all physical
        # channels. For the PCIe-6343 (500 kS/s aggregate) @ 50 kHz that
        # works out to 10 channels; for faster rates the count shrinks
        # automatically. The user can still override via CLI arg.
        max_at_default_rate = int(daq.max_multi_channel_rate // daq.sample_rate)
        default_n = min(len(all_chans), max_at_default_rate)
        requested_n = int(sys.argv[2]) if len(sys.argv) > 2 else default_n
        n_channels = min(requested_n, len(all_chans))
        channels = tuple(all_chans[:n_channels])
        print(
            f"\nUsing {device_name}, first {n_channels} AI channel(s) in "
            f"single-ended (RSE) mode: {list(channels)}"
        )
        print(
            f"  product: {daq.product_type}\n"
            f"  default per-channel rate: {daq.sample_rate:.0f} Hz\n"
            f"  aggregate AI max rate: {daq.max_multi_channel_rate:.0f} Hz\n"
            f"  per-channel max with {n_channels} channels: "
            f"{daq.max_multi_channel_rate / n_channels:.0f} Hz"
        )

        # ---- Phase 1: baseline noise-floor acquisition ----
        duration = 0.1
        print(
            f"\nPhase 1: baseline acquisition ({duration*1000:.0f} ms "
            f"@ {daq.sample_rate:.0f} Hz, no stimulus)..."
        )
        t0 = time.perf_counter()
        acq = daq.acquire(duration=duration, channels=channels)
        dt = time.perf_counter() - t0
        print(
            f"  acquired {acq.n_samples} samples/channel "
            f"in {dt*1000:.1f} ms (wall time)"
        )
        _print_stats_table(
            "  Baseline per-channel statistics (volts):",
            channels,
            [("mean", acq.mean()), ("std", acq.std()),
             ("min", acq.min()), ("max", acq.max())],
        )

        # ---- Phase 2: interactive variation test ----
        phase2_duration = 4.0
        print(
            f"\nPhase 2: cover/uncover the photodiodes by hand for the next "
            f"{phase2_duration:.0f} seconds. Press ENTER to begin (or Ctrl-C to skip)."
        )
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            print("  skipped.")
            return 0

        print(f"  Sampling for {phase2_duration:.0f} seconds...")
        acq2 = daq.acquire(duration=phase2_duration, channels=channels)
        print(f"  acquired {acq2.n_samples} samples/channel at {acq2.sample_rate:.0f} Hz")
        _print_stats_table(
            "  Variation-test per-channel statistics (volts):",
            channels,
            [("mean", acq2.mean()), ("std", acq2.std()),
             ("min", acq2.min()), ("max", acq2.max()),
             ("range", acq2.max() - acq2.min())],
        )
        print(
            "\n  Channels with a meaningful 'range' in Phase 2 that was "
            "~flat in Phase 1 are picking up real light variation."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
