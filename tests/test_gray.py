"""Tests for calibration.gray (reflected binary Gray code).

The encode/decode pair has to round-trip and the encode has to satisfy the
'consecutive integers differ in exactly one bit' property -- including
across the wrap edge of any [0, 2**n_bits) cycle, since add_video_sync_tags
relies on that for its frame-number wrap behavior.
"""

from __future__ import annotations

import numpy as np

from calibration import gray


def _popcount(x: int) -> int:
    return bin(int(x)).count("1")


def test_encode_decode_roundtrip_small():
    n = np.arange(2**12)
    encoded = gray.encode(n)
    decoded = gray.decode(encoded, n_bits=12)
    np.testing.assert_array_equal(decoded, n)


def test_encode_decode_roundtrip_scalar():
    for v in (0, 1, 2, 7, 16, 1023, 65535):
        g = int(gray.encode(np.int64(v)))
        n = int(gray.decode(np.int64(g), n_bits=32))
        assert n == v


def test_encode_consecutive_integers_differ_by_one_bit():
    seq = gray.encode(np.arange(2**14))
    diffs = seq[1:] ^ seq[:-1]
    pops = np.array([_popcount(int(d)) for d in diffs])
    assert (pops == 1).all(), f"non-1 hamming distances at indices {np.where(pops != 1)[0][:5]}"


def test_encode_zero_is_zero():
    assert int(gray.encode(np.int64(0))) == 0


def test_encode_stays_within_n_bits():
    # Encoding values [0, 2^n) produces values in [0, 2^n).
    for n_bits in [3, 4, 5, 8, 12]:
        cycle = 1 << n_bits
        seq = gray.encode(np.arange(cycle))
        assert seq.min() >= 0
        assert seq.max() < cycle


def test_encode_wrap_edge_single_bit_change():
    # add_video_sync_tags wraps frame numbers modulo 2^n_bits; the wrap
    # transition (frame 2^n_bits-1 -> frame 2^n_bits == 0 mod cycle) must
    # also differ in only one bit, i.e. gray(cycle-1) XOR gray(0) == one bit.
    for n_bits in [3, 4, 5, 8, 12]:
        cycle = 1 << n_bits
        last = int(gray.encode(np.int64(cycle - 1)))
        first = int(gray.encode(np.int64(0)))
        assert _popcount(last ^ first) == 1, (
            f"n_bits={n_bits}: gray({cycle - 1})={last} XOR gray(0)={first} "
            f"has popcount {_popcount(last ^ first)}, expected 1"
        )


def test_encode_negative_raises():
    import pytest
    with pytest.raises(ValueError):
        gray.encode(np.array([-1]))


def test_decode_negative_raises():
    import pytest
    with pytest.raises(ValueError):
        gray.decode(np.array([-1]))
