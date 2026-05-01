"""Unit tests for the random-playback module's pure-Python pieces.

The session driver itself opens a fullscreen pygame window and isn't
suited to automated testing -- those parts are exercised manually by
running play_random.py against a real config. These tests cover the
testable bits: IVI sampling, config loading + path resolution.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from playback.play_random import _load_config, _resolve_paths, _sample_ivi


def test_sample_ivi_respects_truncation():
    rng = random.Random(7)
    n = 5000
    samples = [_sample_ivi(rng, 20.0, 2.0, 80.0) for _ in range(n)]
    assert all(2.0 <= s <= 80.0 for s in samples)
    # Mean of an exponential truncated this way should still be roughly
    # close to (but generally less than) the untruncated mean. Loose
    # check: no value out of bounds, mean is non-trivially in range.
    mean = sum(samples) / n
    assert 5.0 < mean < 50.0


def test_sample_ivi_rejects_invalid_params():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        _sample_ivi(rng, mean_s=10.0, lo_s=10.0, hi_s=5.0)
    with pytest.raises(ValueError):
        _sample_ivi(rng, mean_s=-1.0, lo_s=1.0, hi_s=10.0)


def test_sample_ivi_clips_when_truncation_unreachable():
    # mean=1, range=[100,200] -- essentially impossible to sample
    # within. Should clip-and-warn rather than loop forever.
    rng = random.Random(0)
    val = _sample_ivi(rng, mean_s=1.0, lo_s=100.0, hi_s=200.0, max_attempts=10)
    assert 100.0 <= val <= 200.0


def test_resolve_paths_relative_to_config_dir(tmp_path):
    # videos and log_path in config are relative; should resolve to
    # paths relative to the config file, not the CWD.
    (tmp_path / "v1.mp4").touch()
    (tmp_path / "v2.mp4").touch()
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        'videos = ["v1.mp4", "v2.mp4"]\n'
        '[timing]\n'
        'mean_ivi_seconds = 10.0\n'
        'min_ivi_seconds = 1.0\n'
        'max_ivi_seconds = 60.0\n'
        'n_plays = 5\n'
        '[output]\n'
        'log_path = "subdir/log.csv"\n'
    )
    cfg = _load_config(cfg_path)
    videos, log_path = _resolve_paths(cfg_path, cfg)
    assert len(videos) == 2
    for v in videos:
        assert v.is_absolute()
        assert v.parent == tmp_path.resolve()
    assert log_path.is_absolute()
    assert log_path.parent == (tmp_path / "subdir").resolve()


def test_resolve_paths_absolute_kept_as_is(tmp_path):
    abs_video = tmp_path / "abs.mp4"
    abs_video.touch()
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        f'videos = [{json.dumps(str(abs_video))}]\n'
        '[timing]\n'
        'mean_ivi_seconds = 10.0\n'
        'min_ivi_seconds = 1.0\n'
        'max_ivi_seconds = 60.0\n'
        'n_plays = 5\n'
    )
    cfg = _load_config(cfg_path)
    videos, _ = _resolve_paths(cfg_path, cfg)
    assert videos == [abs_video.resolve()]


def test_resolve_paths_missing_video_raises(tmp_path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        'videos = ["does_not_exist.mp4"]\n'
        '[timing]\n'
        'mean_ivi_seconds = 10.0\n'
        'min_ivi_seconds = 1.0\n'
        'max_ivi_seconds = 60.0\n'
        'n_plays = 5\n'
    )
    cfg = _load_config(cfg_path)
    with pytest.raises(FileNotFoundError):
        _resolve_paths(cfg_path, cfg)


def test_resolve_paths_empty_video_list_raises(tmp_path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        'videos = []\n'
        '[timing]\n'
        'mean_ivi_seconds = 10.0\n'
        'min_ivi_seconds = 1.0\n'
        'max_ivi_seconds = 60.0\n'
        'n_plays = 5\n'
    )
    cfg = _load_config(cfg_path)
    with pytest.raises(ValueError, match="at least one"):
        _resolve_paths(cfg_path, cfg)


def test_session_header_includes_config_snapshot_and_metadata(tmp_path):
    from playback.play_random import (
        _write_session_header, _count_existing_sessions,
    )

    cfg_path = tmp_path / "cfg.toml"
    cfg_text = (
        '# A friendly comment\n'
        'videos = ["v1.mp4"]\n'
        '[timing]\n'
        'mean_ivi_seconds = 30.0\n'
        'n_plays = 5\n'
    )
    cfg_path.write_text(cfg_text)
    log = tmp_path / "log.csv"

    with log.open("a", encoding="utf-8") as f:
        _write_session_header(
            f, cfg_path, cfg_text,
            started_iso="2026-05-01T12:00:00+00:00",
            n_sessions_so_far=0,
        )
    body = log.read_text()
    # Banner + config snapshot are present.
    assert "Monitorio playback session #1 on this log file" in body
    assert "config_sha256_12:" in body
    assert "session_started_utc: 2026-05-01T12:00:00+00:00" in body
    # Every config-file line is mirrored as a comment.
    for line in cfg_text.splitlines():
        assert f"#   {line}" in body, f"missing config line: {line!r}"
    # Counts as one session.
    assert _count_existing_sessions(log) == 1


def test_session_header_numbers_subsequent_sessions(tmp_path):
    from playback.play_random import (
        _write_session_header, _count_existing_sessions,
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_text = 'videos = ["v.mp4"]\n[timing]\nn_plays=1\nmean_ivi_seconds=1\n'
    cfg_path.write_text(cfg_text)
    log = tmp_path / "log.csv"

    for i in range(3):
        with log.open("a", encoding="utf-8") as f:
            n = _count_existing_sessions(log)
            _write_session_header(
                f, cfg_path, cfg_text,
                started_iso=f"2026-05-01T12:0{i}:00+00:00",
                n_sessions_so_far=n,
            )
    body = log.read_text()
    assert "session #1 on this log file" in body
    assert "session #2 on this log file" in body
    assert "session #3 on this log file" in body
    assert _count_existing_sessions(log) == 3


def test_config_snapshot_detects_drift_via_hash(tmp_path):
    """Two snapshots of the same config should produce identical hash
    bytes; any edit to the config (even just whitespace) should change
    the hash."""
    from playback.play_random import _write_session_header
    cfg_path = tmp_path / "cfg.toml"
    cfg_text_a = 'videos = ["v.mp4"]\n[timing]\nn_plays=5\nmean_ivi_seconds=30.0\n'
    cfg_text_b = 'videos = ["v.mp4"]\n[timing]\nn_plays=5\nmean_ivi_seconds=31.0\n'
    cfg_path.write_text(cfg_text_a)

    def header_with(text):
        log = tmp_path / "tmp_log.csv"
        log.write_text("")
        with log.open("a", encoding="utf-8") as f:
            _write_session_header(
                f, cfg_path, text,
                started_iso="2026-05-01T00:00:00+00:00",
                n_sessions_so_far=0,
            )
        return log.read_text()

    h_a = header_with(cfg_text_a)
    h_a2 = header_with(cfg_text_a)
    h_b = header_with(cfg_text_b)
    # Same text -> same config_sha256
    assert _extract_hash(h_a) == _extract_hash(h_a2)
    # Different text -> different config_sha256
    assert _extract_hash(h_a) != _extract_hash(h_b)


def _extract_hash(banner_text: str) -> str:
    for line in banner_text.splitlines():
        if line.startswith("# config_sha256_12:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no config_sha256_12 line in:\n{banner_text}")
