"""Manual smoke test: full pipeline + rise/fall time (piece 6).

Runs baselines -> coarse -> fine -> rise-time measurement, reports per-PD
10-90% rise and fall times, and saves the raw traces to a timestamped
.npz so the actual transition waveforms can be inspected.

Usage:
    venv\\Scripts\\python Source\\calibration\\scripts\\smoke_test_rise_time.py \\
        [--display N] [--device NAME] [--rise-rate HZ] [--rise-window SEC] \\
        [--pre-flip SEC] [--save-traces PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the calibration package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from calibration.daq import DAQ, list_ai_channels, list_devices
from calibration.display import Display, list_displays
from calibration.io import get_or_measure_pipeline
from calibration.procedure import (
    DEFAULT_DC_SAMPLE_RATE,
    DEFAULT_RISE_PRE_FLIP_S,
    DEFAULT_RISE_SAMPLE_RATE,
    DEFAULT_RISE_WINDOW_S,
    measure_rise_times,
)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--display", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--rise-rate", type=float, default=DEFAULT_RISE_SAMPLE_RATE,
        dest="rise_rate",
        help="per-channel sample rate for rise/fall acquisition (Hz)",
    )
    p.add_argument(
        "--rise-window", type=float, default=DEFAULT_RISE_WINDOW_S,
        dest="rise_window",
        help="seconds per rise/fall acquisition",
    )
    p.add_argument(
        "--pre-flip", type=float, default=DEFAULT_RISE_PRE_FLIP_S,
        dest="pre_flip",
        help="seconds to delay after acquisition start before flipping",
    )
    p.add_argument(
        "--save-traces", type=str, default=None, dest="save_traces",
        help="path to .npz file for the rise/fall traces (default: "
             "rise_time_traces_<timestamp>.npz in CWD)",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="show a matplotlib figure of the rise and fall traces after "
             "measurement completes",
    )
    p.add_argument(
        "--cache", type=str, default=None,
        help="path to .npz cache of baselines+coarse+fine. Loaded if it "
             "exists; written after measurement if it doesn't. Skips the "
             "slow (~1-2 min) localization step on subsequent runs.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="ignore any existing --cache and re-measure baselines+coarse+fine",
    )
    return p.parse_args(argv)


def _both(display, msg):
    print(msg)
    display.message(msg + "\n\nPress any key to continue. ESC to quit.")
    display.flip()
    return display.wait_for_key()  # True on ESC


def main() -> int:
    args = _parse_args(sys.argv[1:])

    if not list_devices() or not list_displays():
        print("No DAQ or display found.")
        return 1

    device_name = args.device if args.device is not None else list_devices()[0]
    chans = list_ai_channels(device_name)[:10]

    with DAQ(device_name) as daq:
        with Display(args.display) as display:
            print(
                f"Display {args.display}, DAQ {device_name} ({daq.product_type}); "
                f"rise rate={args.rise_rate:.0f} Hz, window={args.rise_window*1000:.0f} ms, "
                f"pre-flip={args.pre_flip*1000:.0f} ms"
            )
            if _both(
                display,
                "Rise-time measurement\n\n"
                "Will run the full localization pipeline first,\n"
                "then measure 10-90% rise and fall time per PD\n"
                "using single-channel high-rate acquisition.\n\n"
                "Do not cover the photodiodes.",
            ):
                return 0

            state = get_or_measure_pipeline(
                display, daq, cache_path=args.cache, force=args.force,
                channels=chans,
            )
            fine = state.fine
            print("rise times...")
            rt = measure_rise_times(
                display, daq, fine,
                duration=args.rise_window,
                pre_flip_s=args.pre_flip,
                sample_rate=args.rise_rate,
            )

            # Terminal report: duration (10-90%) and latency (start -> 10%).
            print("\nResults (ms):")
            hdr = (
                f"  {'channel':<12}"
                f"{'rise dur':>10}{'rise lat':>10}"
                f"{'fall dur':>10}{'fall lat':>10}"
            )
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for i, c in enumerate(rt.channels):
                print(
                    f"  {c:<12}"
                    f"{rt.rise_duration_s[i]*1000:>10.3f}"
                    f"{rt.rise_latency_s[i]*1000:>10.2f}"
                    f"{rt.fall_duration_s[i]*1000:>10.3f}"
                    f"{rt.fall_latency_s[i]*1000:>10.2f}"
                )

            # Save traces for external plotting if requested.
            if args.save_traces:
                out_path = Path(args.save_traces)
            else:
                ts = time.strftime("%Y%m%d-%H%M%S")
                out_path = Path(f"rise_time_traces_{ts}.npz")
            np.savez(
                out_path,
                channels=np.array(rt.channels),
                rise_duration_s=rt.rise_duration_s,
                fall_duration_s=rt.fall_duration_s,
                rise_latency_s=rt.rise_latency_s,
                fall_latency_s=rt.fall_latency_s,
                sample_rate=rt.sample_rate,
                pre_flip_s=rt.pre_flip_s,
                sustain_s=rt.sustain_s,
                rise_trace=rt.rise_trace,
                fall_trace=rt.fall_trace,
            )
            print(f"\nSaved raw traces to: {out_path.resolve()}")

            # On-screen summary.
            lines = ["Rise-time measurement complete", ""]
            lines.append(f"{'ch':<8}{'r_dur':>10}{'r_lat':>10}{'f_dur':>10}{'f_lat':>10}")
            for i, c in enumerate(rt.channels):
                short = c.split("/")[-1]
                lines.append(
                    f"{short:<8}"
                    f"{rt.rise_duration_s[i]*1000:>10.2f}"
                    f"{rt.rise_latency_s[i]*1000:>10.1f}"
                    f"{rt.fall_duration_s[i]*1000:>10.2f}"
                    f"{rt.fall_latency_s[i]*1000:>10.1f}"
                )
            lines.append("(all values in ms)")
            lines += ["", f"Traces: {out_path.name}", "", "Press any key to exit."]
            display.message("\n".join(lines), size=max(20, display.height // 32))
            display.flip()
            display.wait_for_key()

    # Plot AFTER the Display context closes so the pygame fullscreen window
    # doesn't overlap with the matplotlib window.
    if args.plot:
        from calibration.plot import plot_rise_time
        plot_rise_time(rt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
