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
