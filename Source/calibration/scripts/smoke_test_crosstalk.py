"""Manual smoke test: full pipeline + crosstalk verification (piece 7).

Runs baselines -> coarse -> fine, picks a bit-circle radius per PD from
the measured FWHM clipped by the nearest-neighbor distance, then lights
each PD's circle in turn and records every channel to build the
normalized crosstalk matrix.

Usage:
    venv\\Scripts\\python Source\\calibration\\scripts\\smoke_test_crosstalk.py \\
        [--display N] [--device NAME] [--threshold PCT] [--plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from calibration.daq import DAQ, list_ai_channels, list_devices
from calibration.display import Display, list_displays
from calibration.io import get_or_measure_pipeline
from calibration.procedure import (
    DEFAULT_CROSSTALK_THRESHOLD,
    DEFAULT_DC_SAMPLE_RATE,
    measure_crosstalk,
    pick_bit_radius_px,
)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--display", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_CROSSTALK_THRESHOLD,
        help="max acceptable off-diagonal fraction of own dynamic range",
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
    p.add_argument("--plot", action="store_true")
    return p.parse_args(argv)


def _both(display, msg):
    print(msg)
    display.message(msg + "\n\nPress any key to continue. ESC to quit.")
    display.flip()
    return display.wait_for_key()


def main() -> int:
    args = _parse_args(sys.argv[1:])
    if not list_devices() or not list_displays():
        print("No DAQ or display found.")
        return 1

    device_name = args.device if args.device is not None else list_devices()[0]
    chans = list_ai_channels(device_name)[:10]

    with DAQ(device_name) as daq:
        with Display(args.display) as display:
            if _both(
                display,
                "Crosstalk verification\n\n"
                "Pipeline: baselines -> coarse -> fine,\n"
                "then per-PD illuminate + read all channels.\n\n"
                "Do not cover the photodiodes.",
            ):
                return 0

            state = get_or_measure_pipeline(
                display, daq, cache_path=args.cache, force=args.force,
                channels=chans,
            )
            baseline, fine = state.baseline, state.fine
            radii = pick_bit_radius_px(fine)
            print(f"picked radii (px): {list(radii)}")
            print("crosstalk...")
            xt = measure_crosstalk(
                display, daq, fine, baseline,
                radii_px=radii, warn_threshold=args.threshold,
            )

            short = [c.split("/")[-1] for c in xt.channels]
            print("\nNormalized crosstalk matrix (rows = lit PD, cols = channel read):")
            header = "  lit\\read " + "".join(f"{s:>10}" for s in short)
            print(header)
            for i in range(len(short)):
                row = "  " + f"{short[i]:<10}" + "".join(
                    f"{xt.matrix[i, j]:>10.3f}" for j in range(len(short))
                )
                print(row)
            print()
            for i, ch in enumerate(short):
                flag = "" if xt.max_crosstalk[i] < xt.warn_threshold else "  <-- exceeds threshold"
                print(
                    f"  {ch}: radius={xt.radii_px[i]} px, "
                    f"max off-diagonal = {xt.max_crosstalk[i]:.3f}{flag}"
                )
            verdict = "acceptable" if xt.acceptable else f"EXCEEDS {xt.warn_threshold:.1%} THRESHOLD"
            print(f"\nOverall: {verdict}")

            lines = [f"Crosstalk: {verdict}", ""]
            lines.append(f"{'ch':<8}{'radius':>8}{'max xt':>10}")
            for i, ch in enumerate(short):
                lines.append(f"{ch:<8}{xt.radii_px[i]:>8d}{xt.max_crosstalk[i]:>10.3f}")
            lines += ["", "Press any key to exit."]
            display.message("\n".join(lines), size=max(20, display.height // 36))
            display.flip()
            display.wait_for_key()

    if args.plot:
        from calibration.plot import plot_crosstalk
        plot_crosstalk(xt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
