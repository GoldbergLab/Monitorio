"""Tests for add_video_sync_tags._resolve_parameters.

The resolver picks per-field winners between explicit caller args and
calibration JSON, errors out when something is missing, and reads the
screen size from the JSON's monitor.{width,height} block.
"""

from __future__ import annotations

import numpy as np
import pytest

from add_video_sync_tags import _resolve_parameters
from conftest import make_calibration_json


def test_all_from_json(tmp_path):
    cal = make_calibration_json(
        tmp_path / "cal.json",
        screen_w=1920, screen_h=1080,
        pd_xs=(100, 200, 300), pd_y=50,
        bit_radius=15, background_radius=25,
    )
    xs, ys, br, bgr, screen = _resolve_parameters(
        cal, None, None, None, None, None,
    )
    assert list(xs) == [100, 200, 300]
    assert list(ys) == [50, 50, 50]
    assert br == 15
    assert bgr == 25
    assert screen == (1920, 1080)


def test_explicit_overrides_json(tmp_path):
    cal = make_calibration_json(tmp_path / "cal.json")
    xs, ys, br, bgr, screen = _resolve_parameters(
        cal,
        bit_xs=[10, 20], bit_ys=[5],
        bit_radius=7, background_radius=12,
        screen_size=(800, 600),
    )
    assert list(xs) == [10, 20]
    assert list(ys) == [5]
    assert br == 7
    assert bgr == 12
    assert screen == (800, 600)


def test_partial_override(tmp_path):
    # Override only bit_radius; the rest comes from JSON.
    cal = make_calibration_json(
        tmp_path / "cal.json",
        screen_w=1920, screen_h=1080,
        pd_xs=(100, 200), pd_y=50,
        bit_radius=15, background_radius=25,
    )
    xs, ys, br, bgr, screen = _resolve_parameters(
        cal, None, None, bit_radius=99, background_radius=None, screen_size=None,
    )
    assert list(xs) == [100, 200]
    assert br == 99
    assert bgr == 25
    assert screen == (1920, 1080)


def test_no_json_no_args_errors():
    with pytest.raises(ValueError) as excinfo:
        _resolve_parameters(None, None, None, None, None, None)
    msg = str(excinfo.value)
    for required in ("bit_xs", "bit_ys", "bit_radius", "background_radius", "screen_size"):
        assert required in msg


def test_no_json_partial_args_errors_on_missing():
    with pytest.raises(ValueError) as excinfo:
        _resolve_parameters(
            None,
            bit_xs=[10], bit_ys=[20],
            bit_radius=5, background_radius=10,
            screen_size=None,  # only this is missing
        )
    msg = str(excinfo.value)
    assert "screen_size" in msg
    assert "bit_xs" not in msg


def test_json_without_monitor_block_requires_explicit_screen_size(tmp_path):
    import json
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({
        "version": 1,
        # no "monitor" key
        "photodiodes": [
            {"channel": "Dev1/ai0", "x_px": 100.0, "y_px": 50.0,
             "bit_radius_px": 10, "background_radius_px": 20},
        ],
    }))
    with pytest.raises(ValueError) as excinfo:
        _resolve_parameters(p, None, None, None, None, None)
    assert "screen_size" in str(excinfo.value)


def test_empty_photodiode_list_errors(tmp_path):
    import json
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"monitor": {"width": 100, "height": 100},
                             "photodiodes": []}))
    with pytest.raises(ValueError, match="no photodiodes"):
        _resolve_parameters(p, None, None, None, None, None)


def test_screen_size_explicit_overrides_json(tmp_path):
    cal = make_calibration_json(tmp_path / "cal.json", screen_w=1920, screen_h=1080)
    _, _, _, _, screen = _resolve_parameters(
        cal, None, None, None, None, screen_size=(2400, 1600),
    )
    assert screen == (2400, 1600)
