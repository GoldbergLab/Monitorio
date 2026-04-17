"""Manual smoke test: combined Display + DAQ baseline characterization (piece 3).

Measures noise floor (black screen) and full-scale response (white screen)
per channel, reports dynamic range / SNR / liveness. On single-monitor
rigs all instructions and results are shown fullscreen between measurements
as well as printed to the terminal.

Usage:
    venv\\Scripts\\python Source\\calibration\\scripts\\smoke_test_baselines.py \\
        [--display N] [--device NAME] [--channels N] [--sample-rate HZ]

Defaults: display 0, first detected DAQ device, as many channels as fit at
the DAQ's default sample rate, DAQ's default sample rate.

Lowering --sample-rate is useful for diagnosing multiplexer settling
crosstalk on NI DAQs: at low rates (e.g. 5000 Hz) the mux has more
time per channel to settle, and residual voltage from neighboring
channels drops substantially. DC-level measurements like baselines
don't benefit from high rate, so a slower rate is the better default
for this kind of measurement anyway.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the calibration package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from calibration.daq import DAQ, list_ai_channels, list_devices
from calibration.display import Display, list_displays
from calibration.procedure import characterize_baselines


def _format_table(channels, rows) -> str:
    """rows: list of (label, np.ndarray, fmt_string) producing aligned columns."""
    header = f"{'channel':<12}" + "".join(f"{lbl:>12}" for lbl, _, _ in rows)
    out = [header, "-" * len(header)]
    for i, ch in enumerate(channels):
        line = f"{ch:<12}" + "".join(
            (fmt.format(arr[i]) if not isinstance(arr[i], (bool, np.bool_))
             else ("YES" if arr[i] else "no ")).rjust(12)
            for _, arr, fmt in rows
        )
        out.append(line)
    return "\n".join(out)


def _both(display, msg, *, duration_hint: str = ""):
    """Show `msg` on screen (centered) and also print to terminal."""
    print(msg if not duration_hint else f"{msg}  [{duration_hint}]")
    display.message(msg + "\n\nPress any key to continue. ESC to quit.")
    display.flip()
    return display.wait_for_key()  # True on ESC


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--display", type=int, default=0, help="display index")
    p.add_argument("--device", type=str, default=None, help="DAQ device name (e.g. Dev1)")
    p.add_argument("--channels", type=int, default=None, help="number of AI channels to test")
    p.add_argument(
        "--sample-rate", type=float, default=None, dest="sample_rate",
        help="per-channel sample rate in Hz (defaults to DAQ's default)",
    )
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args(sys.argv[1:])

    displays = list_displays()
    devices = list_devices()
    print("Displays:")
    for d in displays:
        print(f"  [{d.index}] {d.width}x{d.height}")
    print("DAQ devices:")
    for d in devices:
        print(f"  {d}")
    if not devices:
        print("\nNo DAQ device found.")
        return 1
    if not displays:
        print("\nNo display found.")
        return 1

    display_index = args.display
    device_name = args.device if args.device is not None else devices[0]

    all_chans = list_ai_channels(device_name)

    # If --sample-rate was given, configure the DAQ for it up front so the
    # aggregate-rate sanity check at acquire time uses the right rate.
    daq_kwargs = {}
    if args.sample_rate is not None:
        daq_kwargs["sample_rate"] = args.sample_rate

    with DAQ(device_name, **daq_kwargs) as daq:
        max_at_rate = int(daq.max_multi_channel_rate // daq.sample_rate)
        default_n = min(len(all_chans), max_at_rate)
        n_channels = args.channels if args.channels is not None else default_n
        n_channels = min(n_channels, len(all_chans))
        channels = tuple(all_chans[:n_channels])

        print(
            f"\nRunning with display {display_index}, DAQ {device_name} "
            f"({daq.product_type}), {n_channels} AI channel(s) @ "
            f"{daq.sample_rate:.0f} Hz."
        )

        with Display(display_index) as display:
            if _both(
                display,
                "Baseline characterization\n\n"
                "About to measure noise floor (black screen) then\n"
                "full-scale response (white screen).\n\n"
                "Do not cover the photodiodes during this test.",
            ):
                return 0

            print("\nMeasuring dark and bright baselines...")
            result = characterize_baselines(display, daq, channels=channels)

            # Terminal report.
            rows = [
                ("dark mean",  result.dark_mean(),  "{:>12.4f}"),
                ("dark std",   result.dark_std(),   "{:>12.5f}"),
                ("bright mean", result.bright_mean(), "{:>12.4f}"),
                ("range",     result.dynamic_range(), "{:>12.4f}"),
                ("SNR",       result.snr(),         "{:>12.0f}"),
                ("live?",     result.liveness(),    ""),
            ]
            table = _format_table(result.channels, rows)
            print("\nPer-channel baselines (volts):\n" + table)
            live_idx = np.flatnonzero(result.liveness())
            print(
                f"\nLive channels: {len(live_idx)} of {n_channels} "
                f"({[result.channels[i] for i in live_idx]})"
            )

            # On-screen report.
            lines = ["Baselines complete"]
            lines.append(
                f"{'ch':<10}{'dark':>10}{'bright':>10}{'range':>10}{'SNR':>8} live"
            )
            for i, ch in enumerate(result.channels):
                short = ch.split("/")[-1]
                lines.append(
                    f"{short:<10}"
                    f"{result.dark_mean()[i]:>10.3f}"
                    f"{result.bright_mean()[i]:>10.3f}"
                    f"{result.dynamic_range()[i]:>10.3f}"
                    f"{result.snr()[i]:>8.0f}"
                    f"{'  YES' if result.liveness()[i] else '   no'}"
                )
            lines.append("")
            lines.append(f"Live channels: {len(live_idx)} of {n_channels}")
            lines.append("")
            lines.append("Press any key to exit.")
            # Smaller font: 10+ rows need tighter spacing on shorter monitors.
            display.message("\n".join(lines), size=max(16, display.height // 42))
            display.flip()
            display.wait_for_key()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
