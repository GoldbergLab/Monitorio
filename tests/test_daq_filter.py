"""Tests for terminal-config-aware AI channel filtering.

calibration.daq.list_ai_channels filters physical AI pins by their
self-reported `ai_term_cfgs`, so DIFF/PSEUDO_DIFF only enumerate the
positive-input pins (the negative-input partners aren't separately
addressable). The filter is independent of any specific NI device.

These tests mock nidaqmx.system.System so they run without DAQ hardware
or the NI-DAQmx driver. If the nidaqmx module isn't importable at all
(e.g. on a stripped-down CI box), the entire module is skipped.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

nidaqmx = pytest.importorskip("nidaqmx")
from nidaqmx.constants import TerminalConfiguration  # noqa: E402

from calibration.daq import list_ai_channels  # noqa: E402


def _make_chan(name: str, term_cfgs: list[TerminalConfiguration]):
    chan = MagicMock()
    chan.name = name
    chan.ai_term_cfgs = list(term_cfgs)
    return chan


def _make_device(channels):
    dev = MagicMock()
    dev.ai_physical_chans = channels
    sys_mock = MagicMock()
    sys_mock.local.return_value.devices = {"Dev1": dev}
    return sys_mock


# Mimic the PCIe-6343 layout: AI0-7 and AI16-23 support DIFF;
# AI8-15 and AI24-31 are RSE/NRSE-only (the negative-input partners).
RSE = TerminalConfiguration.RSE
NRSE = TerminalConfiguration.NRSE
DIFF = TerminalConfiguration.DIFF


def _x_series_chans():
    chans = []
    for i in range(32):
        cfgs = [RSE, NRSE]
        if i < 8 or 16 <= i < 24:
            cfgs.append(DIFF)
        chans.append(_make_chan(f"Dev1/ai{i}", cfgs))
    return chans


def test_no_filter_returns_all():
    sys_mock = _make_device(_x_series_chans())
    with patch("calibration.daq.System", sys_mock):
        names = list_ai_channels("Dev1")
    assert len(names) == 32
    assert names[0] == "Dev1/ai0"
    assert names[-1] == "Dev1/ai31"


def test_rse_filter_returns_all_pins():
    sys_mock = _make_device(_x_series_chans())
    with patch("calibration.daq.System", sys_mock):
        names = list_ai_channels("Dev1", terminal_config=RSE)
    assert len(names) == 32


def test_diff_filter_returns_only_positive_inputs():
    sys_mock = _make_device(_x_series_chans())
    with patch("calibration.daq.System", sys_mock):
        names = list_ai_channels("Dev1", terminal_config=DIFF)
    # 8 from AI0-7 and 8 from AI16-23 = 16 differential pairs
    assert len(names) == 16
    suffixes = [n.split("/")[-1] for n in names]
    assert suffixes == [f"ai{i}" for i in list(range(8)) + list(range(16, 24))]


def test_filter_returns_empty_if_unsupported():
    # PSEUDO_DIFF isn't in any pin's term_cfgs in our mock layout.
    sys_mock = _make_device(_x_series_chans())
    with patch("calibration.daq.System", sys_mock):
        names = list_ai_channels(
            "Dev1", terminal_config=TerminalConfiguration.PSEUDO_DIFF,
        )
    assert names == []


def test_filter_preserves_physical_order():
    # If a card had a non-contiguous DIFF layout, filtered order should
    # still match physical pin order.
    chans = [
        _make_chan("Dev1/ai0", [RSE, DIFF]),
        _make_chan("Dev1/ai1", [RSE]),
        _make_chan("Dev1/ai2", [RSE, DIFF]),
        _make_chan("Dev1/ai3", [RSE]),
    ]
    sys_mock = _make_device(chans)
    with patch("calibration.daq.System", sys_mock):
        names = list_ai_channels("Dev1", terminal_config=DIFF)
    assert names == ["Dev1/ai0", "Dev1/ai2"]
