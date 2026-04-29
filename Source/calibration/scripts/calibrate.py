"""Monitorio calibration: one-shot end-user entry point (piece 8).

Runs the full calibration pipeline end to end:

  1. baseline characterization  (or load from --cache)
  2. coarse structured-light localization  (slow, cached)
  3. fine bar-sweep refinement  (slow, cached)
  4. pick bit-circle radius per PD  (from FWHM + neighbor spacing)
  5. pick black-background radius per PD  (from sweep tails)
  6. rise/fall-time measurement
  7. crosstalk matrix verification

Then pops up three matplotlib figures (spatial response, temporal
response, crosstalk heatmap) for you to eyeball, asks whether to save,
and writes a MATLAB-loadable JSON with everything addVideoSyncTags
needs: per-PD center (x, y), bit and background radii, plus per-PD
baselines, rise/fall times, and the full crosstalk matrix as metadata.

Usage:
    venv\\Scripts\\python Source\\calibration\\scripts\\calibrate.py \\
        [--display N] [--device NAME] [--cache PATH] [--output PATH] \\
        [--no-plot] [--no-confirm] [--crosstalk-threshold PCT] [--force]

Re-running with the same --cache skips the ~1 min localization step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from calibration.daq import (
    DAQ,
    TERMINAL_CONFIG_CHOICES,
    list_ai_channels,
    list_devices,
    terminal_config_from_name,
)
from calibration.display import Display, list_displays
from calibration.io import get_or_measure_pipeline
from calibration.procedure import (
    DEFAULT_CROSSTALK_THRESHOLD,
    DEFAULT_DC_SAMPLE_RATE,
    measure_crosstalk,
    measure_rise_times,
    pick_background_radius_px,
    pick_bit_radius_px,
)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--display", type=int, default=0, help="display index")
    p.add_argument("--device", type=str, default=None, help="NI DAQ device name (e.g. Dev1)")
    p.add_argument(
        "--cache", type=str, default="calibration_cache.npz",
        help="path to .npz cache of baselines+coarse+fine. Loaded if it "
             "exists; written if not. Default: calibration_cache.npz in CWD",
    )
    p.add_argument(
        "--force", action="store_true",
        help="ignore any existing --cache and re-measure from scratch",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="path to write calibration JSON. Default: calibration_<timestamp>.json",
    )
    p.add_argument("--no-plot", action="store_true", help="skip matplotlib plots")
    p.add_argument(
        "--no-confirm", action="store_true",
        help="save without asking for confirmation",
    )
    p.add_argument(
        "--crosstalk-threshold", type=float, default=DEFAULT_CROSSTALK_THRESHOLD,
        dest="crosstalk_threshold",
    )
    p.add_argument(
        "--terminal-config", type=str.upper, default="RSE",
        choices=TERMINAL_CONFIG_CHOICES, dest="terminal_config",
        help="AI terminal configuration (default: RSE)",
    )
    return p.parse_args(argv)


def _intro(display: Display) -> bool:
    msg = (
        "Monitorio calibration\n\n"
        "Will:\n"
        "  1. Locate each photodiode on the screen\n"
        "  2. Measure monitor rise/fall time\n"
        "  3. Check for channel crosstalk\n"
        "  4. Compute bit and background circle sizes\n\n"
        "Do not cover the photodiodes during measurement."
    )
    print(msg)
    display.message(msg + "\n\nPress any key to continue. ESC to abort.")
    display.flip()
    return display.wait_for_key()  # True on ESC


def _summary_lines(fine, bit_radii, bg_radii, rt, xt) -> list[str]:
    short = [c.split("/")[-1] for c in fine.channels]
    lines = [
        "Calibration complete", "",
        f"{'ch':<6}{'x':>8}{'y':>9}{'r_bit':>7}{'r_bg':>6}"
        f"{'rise ms':>10}{'xt':>8}",
    ]
    for i in range(len(fine.channels)):
        rise_ms = (rt.rise_duration_s[i] * 1000.0
                   if np.isfinite(rt.rise_duration_s[i]) else float("nan"))
        lines.append(
            f"{short[i]:<6}"
            f"{fine.x_pixels[i]:>8.1f}"
            f"{fine.y_pixels[i]:>9.1f}"
            f"{bit_radii[i]:>7d}"
            f"{bg_radii[i]:>6d}"
            f"{rise_ms:>10.3f}"
            f"{xt.max_crosstalk[i]:>8.4f}"
        )
    lines += [
        "",
        f"Crosstalk: {'OK' if xt.acceptable else 'FAIL (exceeds ' + f'{xt.warn_threshold:.1%}' + ')'}",
    ]
    return lines


def _build_json(
    *, state, rt, xt, bit_radii, bg_radii,
    display_index: int, display_w: int, display_h: int,
    device_name: str, product_type: str, sample_rate_hz: float,
    terminal_config: str,
) -> dict:
    def _nan_to_none(x):
        x = float(x)
        return None if not np.isfinite(x) else x

    baseline = state.baseline
    fine = state.fine

    photodiodes = []
    for i, ch in enumerate(fine.channels):
        bi = baseline.channels.index(ch)
        photodiodes.append({
            "channel": ch,
            "x_px": float(fine.x_pixels[i]),
            "y_px": float(fine.y_pixels[i]),
            "fwhm_x_px": int(fine.x_fwhm_px[i]),
            "fwhm_y_px": int(fine.y_fwhm_px[i]),
            "bit_radius_px": int(bit_radii[i]),
            "background_radius_px": int(bg_radii[i]),
            "baseline_dark_v": float(baseline.dark_mean()[bi]),
            "baseline_bright_v": float(baseline.bright_mean()[bi]),
            "dynamic_range_v": float(baseline.dynamic_range()[bi]),
            "rise_duration_s": _nan_to_none(rt.rise_duration_s[i]),
            "fall_duration_s": _nan_to_none(rt.fall_duration_s[i]),
            "rise_latency_s": _nan_to_none(rt.rise_latency_s[i]),
            "fall_latency_s": _nan_to_none(rt.fall_latency_s[i]),
        })

    return {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monitor": {
            "index": int(display_index),
            "width": int(display_w),
            "height": int(display_h),
        },
        "daq": {
            "device": device_name,
            "product_type": product_type,
            "terminal_config": terminal_config,
            "dc_sample_rate_hz": float(sample_rate_hz),
            "rise_time_sample_rate_hz": float(rt.sample_rate),
        },
        "photodiodes": photodiodes,
        "crosstalk": {
            "channels": list(xt.channels),
            "matrix": [[float(v) for v in row] for row in xt.matrix],
            "max_off_diagonal": [float(v) for v in xt.max_crosstalk],
            "threshold": float(xt.warn_threshold),
            "acceptable": bool(xt.acceptable),
        },
    }


def main() -> int:
    args = _parse_args(sys.argv[1:])
    displays = list_displays()
    devices = list_devices()
    if not devices or not displays:
        print("No DAQ or display found.")
        return 1

    device_name = args.device if args.device is not None else devices[0]
    tc = terminal_config_from_name(args.terminal_config)
    # Filter by terminal config: in DIFF mode the device only exposes half its
    # AI pins as addressable channels, so we must query usable channels under
    # the chosen config rather than blindly taking the first 10 physical pins.
    chans = list_ai_channels(device_name, terminal_config=tc)[:10]

    output_path = Path(
        args.output or f"calibration_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )

    # --- Hardware phase: measure everything. -------------------------
    print(f"Terminal config: {args.terminal_config}")
    with DAQ(device_name, terminal_config=tc) as daq:
        with Display(args.display) as display:
            product_type = daq.product_type
            display_w, display_h = display.width, display.height

            if _intro(display):
                return 0

            state = get_or_measure_pipeline(
                display, daq,
                cache_path=args.cache, force=args.force, channels=chans,
            )
            print(f"\n{len(state.fine.channels)} live channel(s): {list(state.fine.channels)}")

            bit_radii = pick_bit_radius_px(state.fine)
            bg_radii = pick_background_radius_px(state.fine, state.baseline)
            print(f"bit radii (px):        {list(bit_radii)}")
            print(f"background radii (px): {list(bg_radii)}")

            print("\nmeasuring rise time...")
            rt = measure_rise_times(display, daq, state.fine)

            print("measuring crosstalk...")
            xt = measure_crosstalk(
                display, daq, state.fine, state.baseline,
                radii_px=bit_radii, warn_threshold=args.crosstalk_threshold,
            )

            summary = _summary_lines(state.fine, bit_radii, bg_radii, rt, xt)
            for line in summary:
                print(line)

            display.message(
                "\n".join(summary + ["", "Press any key to continue."]),
                size=max(18, display.height // 38),
            )
            display.flip()
            display.wait_for_key()

    # --- Inspection phase: out of fullscreen, show plots. ------------
    if not args.no_plot:
        from calibration.plot import plot_crosstalk, plot_refine, plot_rise_time
        print("\nShowing spatial response. Close the window to continue.")
        plot_refine(state.fine)
        print("Showing temporal response. Close the window to continue.")
        plot_rise_time(rt)
        print("Showing crosstalk matrix. Close the window to continue.")
        plot_crosstalk(xt)

    # --- Save phase. --------------------------------------------------
    payload = _build_json(
        state=state, rt=rt, xt=xt,
        bit_radii=bit_radii, bg_radii=bg_radii,
        display_index=args.display, display_w=display_w, display_h=display_h,
        device_name=device_name, product_type=product_type,
        sample_rate_hz=DEFAULT_DC_SAMPLE_RATE,
        terminal_config=args.terminal_config,
    )

    if not args.no_confirm:
        resp = input(f"\nSave calibration to {output_path}? [Y/n]: ").strip().lower()
        if resp not in ("", "y", "yes"):
            print("Not saved.")
            return 0

    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
