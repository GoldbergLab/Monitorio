"""Persistence for the slow half of the calibration pipeline.

Running baselines -> coarse -> fine takes ~1-2 minutes. The PDs don't
move between runs on the same rig, so caching those three results to
disk and reloading on subsequent runs lets us jump straight to the
fast measurements (rise time, crosstalk).

`get_or_measure_pipeline()` is the one entry point most callers need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibration.daq import DAQ, Acquisition
from calibration.display import Display
from calibration.procedure import (
    BaselineResult,
    CoarseLocations,
    FineLocations,
    characterize_baselines,
    localize_coarse,
    refine_locations,
)


@dataclass(frozen=True)
class PipelineState:
    baseline: BaselineResult
    coarse: CoarseLocations
    fine: FineLocations


def save_pipeline_state(path: str | Path, state: PipelineState) -> None:
    """Write a PipelineState to an .npz file.

    Raw baseline time series are stored so BaselineResult is reconstructed
    exactly (including std and liveness). Fine sweeps are concatenated
    with a per-channel length index so variable-length per-PD windows
    survive the round-trip.
    """
    data: dict[str, np.ndarray] = {
        # Baseline
        "baseline_dark_samples": state.baseline.dark.samples,
        "baseline_dark_channels": np.array(state.baseline.dark.channels),
        "baseline_dark_rate": np.float64(state.baseline.dark.sample_rate),
        "baseline_bright_samples": state.baseline.bright.samples,
        "baseline_bright_channels": np.array(state.baseline.bright.channels),
        "baseline_bright_rate": np.float64(state.baseline.bright.sample_rate),
        # Coarse
        "coarse_channels": np.array(state.coarse.channels),
        "coarse_x_pixels": state.coarse.x_pixels,
        "coarse_y_pixels": state.coarse.y_pixels,
        "coarse_uncertainty_px": np.int64(state.coarse.uncertainty_px),
        "coarse_min_confidence": state.coarse.min_confidence,
        # Fine
        "fine_channels": np.array(state.fine.channels),
        "fine_x_pixels": state.fine.x_pixels,
        "fine_y_pixels": state.fine.y_pixels,
        "fine_x_fwhm_px": state.fine.x_fwhm_px,
        "fine_y_fwhm_px": state.fine.y_fwhm_px,
    }
    for prefix, sweeps in (
        ("fine_x_sweeps", state.fine.x_sweeps),
        ("fine_y_sweeps", state.fine.y_sweeps),
    ):
        lengths = np.array([len(p) for p, _ in sweeps], dtype=np.int64)
        positions = (
            np.concatenate([p for p, _ in sweeps])
            if sweeps else np.empty(0, dtype=np.float64)
        )
        responses = (
            np.concatenate([r for _, r in sweeps])
            if sweeps else np.empty(0, dtype=np.float64)
        )
        data[f"{prefix}_lengths"] = lengths
        data[f"{prefix}_positions"] = positions
        data[f"{prefix}_responses"] = responses
    np.savez(str(path), **data)


def load_pipeline_state(path: str | Path) -> PipelineState:
    """Inverse of save_pipeline_state. Raises on missing keys."""
    f = np.load(str(path), allow_pickle=False)

    dark = Acquisition(
        samples=f["baseline_dark_samples"],
        channels=tuple(str(c) for c in f["baseline_dark_channels"]),
        sample_rate=float(f["baseline_dark_rate"]),
    )
    bright = Acquisition(
        samples=f["baseline_bright_samples"],
        channels=tuple(str(c) for c in f["baseline_bright_channels"]),
        sample_rate=float(f["baseline_bright_rate"]),
    )
    baseline = BaselineResult(dark=dark, bright=bright)

    coarse = CoarseLocations(
        channels=tuple(str(c) for c in f["coarse_channels"]),
        x_pixels=f["coarse_x_pixels"],
        y_pixels=f["coarse_y_pixels"],
        uncertainty_px=int(f["coarse_uncertainty_px"]),
        min_confidence=f["coarse_min_confidence"],
    )

    def _load_sweeps(prefix: str):
        lengths = f[f"{prefix}_lengths"]
        positions = f[f"{prefix}_positions"]
        responses = f[f"{prefix}_responses"]
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        return tuple(
            (positions[offsets[i]:offsets[i + 1]],
             responses[offsets[i]:offsets[i + 1]])
            for i in range(len(lengths))
        )

    fine = FineLocations(
        channels=tuple(str(c) for c in f["fine_channels"]),
        x_pixels=f["fine_x_pixels"],
        y_pixels=f["fine_y_pixels"],
        x_fwhm_px=f["fine_x_fwhm_px"],
        y_fwhm_px=f["fine_y_fwhm_px"],
        x_sweeps=_load_sweeps("fine_x_sweeps"),
        y_sweeps=_load_sweeps("fine_y_sweeps"),
    )
    return PipelineState(baseline=baseline, coarse=coarse, fine=fine)


def get_or_measure_pipeline(
    display: Display,
    daq: DAQ,
    *,
    cache_path: str | Path | None = None,
    force: bool = False,
    channels: tuple[str, ...] | list[str] | None = None,
) -> PipelineState:
    """Return a PipelineState, loading from cache if available else measuring.

    If `cache_path` is given and the file exists and `force` is False,
    the cached state is loaded. Otherwise the full baselines -> coarse
    -> fine pipeline runs, and if `cache_path` is set the result is
    saved to disk for next time.
    """
    path = Path(cache_path) if cache_path else None
    if path and path.exists() and not force:
        print(f"loading cached pipeline state from {path}")
        return load_pipeline_state(path)

    print("measuring pipeline state (baselines -> coarse -> fine)...")
    baseline = characterize_baselines(display, daq, channels=channels)
    live = tuple(c for c, l in zip(baseline.channels, baseline.liveness()) if l)
    if not live:
        raise RuntimeError("characterize_baselines found no live channels")
    coarse = localize_coarse(display, daq, baseline, channels=live)
    fine = refine_locations(display, daq, baseline, coarse)
    state = PipelineState(baseline=baseline, coarse=coarse, fine=fine)

    if path:
        save_pipeline_state(path, state)
        print(f"saved pipeline state to {path}")
    return state
