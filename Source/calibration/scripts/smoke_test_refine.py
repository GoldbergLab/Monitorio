"""Manual smoke test: full localization pipeline, coarse + fine (piece 5).

Runs baselines -> structured-light coarse localization -> bar-sweep
centroid refinement, then shows the fine (sub-pixel) positions as small
circles over the physical photodiodes.

Usage:
    venv\\Scripts\\python Source\\calibration\\scripts\\smoke_test_refine.py \\
        [--display N] [--device NAME] [--channels N] [--sample-rate HZ] \\
        [--k-min K] [--bar-width W] [--settle SEC] [--refine-settle SEC]

Most defaults come from calibration.procedure. --bar-width sets the
thickness (px) of the sweep bar -- should be smaller than the PD's
effective diameter (~20 px on the test rig).
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
from calibration.procedure import (
    DEFAULT_DC_SAMPLE_RATE,
    DEFAULT_K_MIN,
    DEFAULT_REFINE_BAR_WIDTH_PX,
    DEFAULT_REFINE_MARGIN_PX,
    DEFAULT_REFINE_SETTLE_TIME_S,
    DEFAULT_SETTLE_TIME_S,
    characterize_baselines,
    localize_coarse,
    refine_locations,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--display", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument(
        "--sample-rate", type=float, default=None, dest="sample_rate",
    )
    p.add_argument("--k-min", type=int, default=DEFAULT_K_MIN, dest="k_min")
    p.add_argument(
        "--bar-width", type=int, default=DEFAULT_REFINE_BAR_WIDTH_PX,
        dest="bar_width",
    )
    p.add_argument(
        "--margin", type=int, default=DEFAULT_REFINE_MARGIN_PX, dest="margin",
        help="extra px beyond the coarse uncertainty window to sweep "
             "(must exceed the PD's sensitive radius)",
    )
    p.add_argument(
        "--settle", type=float, default=DEFAULT_SETTLE_TIME_S,
        help="settle time for baselines + coarse localization (s)",
    )
    p.add_argument(
        "--refine-settle", type=float, default=DEFAULT_REFINE_SETTLE_TIME_S,
        dest="refine_settle",
        help="settle time for the bar sweep (s); typically 5-10x shorter than --settle",
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
    all_chans = list_ai_channels(device_name)
    effective_rate = (
        args.sample_rate if args.sample_rate is not None
        else DEFAULT_DC_SAMPLE_RATE
    )

    with DAQ(device_name) as daq:
        max_at_rate = int(daq.max_multi_channel_rate // effective_rate)
        default_n = min(len(all_chans), max_at_rate)
        n_channels = args.channels if args.channels is not None else default_n
        n_channels = min(n_channels, len(all_chans))
        channels = tuple(all_chans[:n_channels])

        print(
            f"Display {args.display}, DAQ {device_name} ({daq.product_type}), "
            f"{n_channels} channel(s) @ {effective_rate:.0f} Hz, "
            f"k_min={args.k_min}, bar_width={args.bar_width}"
        )

        with Display(args.display) as display:
            if _both(
                display,
                "Full localization pipeline\n\n"
                "1. Baselines (black + white screens)\n"
                "2. Gray-coded stripe patterns (coarse, +/- 16 px)\n"
                "3. Bar sweep across each PD's coarse window\n"
                "   (fine, sub-pixel)\n\n"
                "Do not cover the photodiodes.",
            ):
                return 0

            print("\nRunning baselines...")
            baseline = characterize_baselines(
                display, daq, channels=channels,
                settle_time=args.settle, sample_rate=effective_rate,
            )
            live = [c for c, l in zip(baseline.channels, baseline.liveness()) if l]
            if not live:
                display.message("No live channels found. Aborting.")
                display.flip()
                display.wait_for_key()
                return 1
            print(f"  {len(live)} live: {live}")

            print("\nRunning coarse localization...")
            coarse = localize_coarse(
                display, daq, baseline,
                k_min=args.k_min, channels=live,
                settle_time=args.settle, sample_rate=effective_rate,
            )
            for i, c in enumerate(coarse.channels):
                print(
                    f"  {c}: coarse=({coarse.x_pixels[i]:4d}, {coarse.y_pixels[i]:4d})"
                    f"  conf={coarse.min_confidence[i]:.2f}"
                )

            print("\nRunning fine refinement...")
            fine = refine_locations(
                display, daq, baseline, coarse,
                bar_width=args.bar_width,
                margin_px=args.margin,
                settle_time=args.refine_settle,
                sample_rate=effective_rate,
            )

            # Terminal report: fine (x, y), FWHM, and shift from coarse.
            print(f"\nFinal localization:")
            header = (
                f"  {'channel':<12}{'fine x':>10}{'fine y':>10}"
                f"{'dx':>8}{'dy':>8}{'fwhm_x':>8}{'fwhm_y':>8}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for i, ch in enumerate(fine.channels):
                dx = fine.x_pixels[i] - coarse.x_pixels[i]
                dy = fine.y_pixels[i] - coarse.y_pixels[i]
                print(
                    f"  {ch:<12}"
                    f"{fine.x_pixels[i]:>10.2f}"
                    f"{fine.y_pixels[i]:>10.2f}"
                    f"{dx:>8.2f}"
                    f"{dy:>8.2f}"
                    f"{fine.x_fwhm_px[i]:>8d}"
                    f"{fine.y_fwhm_px[i]:>8d}"
                )

            # Visual verification: small circles at the fine positions.
            short = [c.split("/")[-1] for c in fine.channels]
            # Skip any channel whose refinement failed (NaN).
            points: list[tuple[int, int, str]] = []
            for i in range(len(fine.channels)):
                xi, yi = fine.x_pixels[i], fine.y_pixels[i]
                if np.isnan(xi) or np.isnan(yi):
                    continue
                points.append((int(round(xi)), int(round(yi)), short[i]))
            # Radius ~ half the PD's FWHM, floored at 6 px so labels don't overlap the dot.
            radius = max(6, int(np.nanmedian(np.concatenate(
                [fine.x_fwhm_px, fine.y_fwhm_px]
            )) // 2)) if points else 10
            display.annotated_points(points, radius=radius)
            display.flip()
            print(
                "\nVisual check: each labeled circle should sit directly on "
                "its photodiode.\n(Press any key to continue.)"
            )
            display.wait_for_key()

            # On-screen summary.
            lines = ["Fine localization complete", ""]
            lines.append(f"{'channel':<10}{'x':>9}{'y':>9}{'fwhm':>8}")
            for i, ch in enumerate(fine.channels):
                mean_fwhm = (fine.x_fwhm_px[i] + fine.y_fwhm_px[i]) / 2
                lines.append(
                    f"{short[i]:<10}"
                    f"{fine.x_pixels[i]:>9.2f}"
                    f"{fine.y_pixels[i]:>9.2f}"
                    f"{mean_fwhm:>8.1f}"
                )
            lines += ["", "Press any key to exit."]
            display.message("\n".join(lines), size=max(18, display.height // 38))
            display.flip()
            display.wait_for_key()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
