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
    from playback.play_random import _write_session_header

    cfg_path = tmp_path / "cfg.toml"
    cfg_text = (
        '# A friendly comment that should be stripped from the snapshot\n'
        '\n'
        '   # An indented comment also stripped\n'
        'videos = ["v1.mp4"]\n'
        '\n'
        '[timing]   # trailing comments are kept (they are part of the line)\n'
        'mean_ivi_seconds = 30.0\n'
        'n_plays = 5\n'
    )
    cfg_path.write_text(cfg_text)
    log = tmp_path / "log.csv"

    with log.open("w", encoding="utf-8") as f:
        _write_session_header(
            f, cfg_path, cfg_text,
            started_iso="2026-05-01T12:00:00+00:00",
        )
    body = log.read_text()
    # Banner + metadata.
    assert "Monitorio playback session" in body
    assert "config_sha256_12:" in body
    assert "session_started_utc: 2026-05-01T12:00:00+00:00" in body
    # Real data lines made it through, with the CSV-comment "# " prefix.
    assert '# videos = ["v1.mp4"]' in body
    assert "# [timing]   # trailing comments are kept (they are part of the line)" in body
    assert "# mean_ivi_seconds = 30.0" in body
    assert "# n_plays = 5" in body
    # Pure-comment lines and blank lines were stripped.
    assert "A friendly comment" not in body
    assert "An indented comment" not in body


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
        with log.open("w", encoding="utf-8") as f:
            _write_session_header(
                f, cfg_path, text,
                started_iso="2026-05-01T00:00:00+00:00",
            )
        return log.read_text()

    h_a = header_with(cfg_text_a)
    h_a2 = header_with(cfg_text_a)
    h_b = header_with(cfg_text_b)
    assert _extract_hash(h_a) == _extract_hash(h_a2)
    assert _extract_hash(h_a) != _extract_hash(h_b)


def _extract_hash(banner_text: str) -> str:
    for line in banner_text.splitlines():
        if line.startswith("# config_sha256_12:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no config_sha256_12 line in:\n{banner_text}")


def test_timestamped_log_path_inserts_timestamp(tmp_path):
    from playback.play_random import _timestamped_log_path
    import datetime
    when = datetime.datetime(2026, 5, 1, 23, 4, 5, tzinfo=datetime.timezone.utc)
    out = _timestamped_log_path(tmp_path / "playback_log.csv", when)
    assert out.name == "playback_log_20260501T230405.csv"
    assert out.parent == tmp_path


def test_timestamped_log_path_handles_missing_extension(tmp_path):
    from playback.play_random import _timestamped_log_path
    import datetime
    when = datetime.datetime(2026, 5, 1, 1, 2, 3, tzinfo=datetime.timezone.utc)
    out = _timestamped_log_path(tmp_path / "session", when)
    # No extension on the base -> default to .csv
    assert out.name == "session_20260501T010203.csv"


def test_timestamped_log_path_avoids_windows_illegal_chars(tmp_path):
    from playback.play_random import _timestamped_log_path
    import datetime
    when = datetime.datetime(2026, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)
    out = _timestamped_log_path(tmp_path / "log.csv", when)
    # No colons, slashes, backslashes etc. in the inserted timestamp.
    inserted = out.stem.split("_", 1)[1]
    for bad in ":/\\<>|?*\"":
        assert bad not in inserted, f"timestamp {inserted!r} contains illegal Windows char {bad!r}"
