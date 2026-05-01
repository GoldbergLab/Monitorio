"""Loader for Intan RHD2000 .rhd recordings (Traditional Intan File Format).

Two public functions, picking out different analog input ports on Intan
hardware:

  load_rhd_board_adc(paths)
    Reads the controller box's analog input ports (BNC connectors on the
    Intan Recording Controller, screw terminals on the older USB
    Interface Board). This is where Monitorio recordings normally route
    the photodiode signals.

  load_rhd_aux(paths)
    Reads the headstage's auxiliary inputs (3 per RHD chip: AUX1, AUX2,
    AUX3), used by most labs for accelerometer or temperature data. NOT
    the same as the controller's analog inputs, despite the name --
    different sample rate, different unit conversion, different physical
    location.

Both return a (n_channels, n_samples) numpy array in volts plus the
sample rate of those channels. The output plugs straight into
decode_sync_tags with scale=1.0 since unit conversion is already
applied here.

Channel order in the returned array matches the order the channels
appear in the .rhd file's signal group enumeration, which mirrors
physical pin order. The decoder expects this same physical order in
its samples argument, so the array passes straight through provided
the recording wired up the PDs in calibration order.

The parser is written from scratch against the published RHD file-
format spec (intantech.com) rather than wrapping Intan's own
importrhdutilities.py module, which is GPL-3.0 and would carry over
to anything that imports it. Scope is intentionally narrow: only the
header fields we need + the bytes for the requested signal type from
each data block. Amp, supply, temp, digital data, and the
non-requested-of-{aux, board ADC} are all skipped efficiently without
unpacking.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# Per the RHD file-format spec.
RHD_MAGIC = 0xC6912702

# Per-step voltage conversion for the headstage AUX inputs (always this,
# regardless of board mode).
AUX_VOLTS_PER_STEP = 0.0000374

# Per-step voltage conversion for the controller-box analog inputs is
# board-mode-dependent. Some board modes also need an offset subtraction
# before the multiplier (the ADC is centered around 32768 in those
# modes, so raw - 32768 gives a signed value).
#   board_mode 0:  USB Interface Board, 0..3.3 V  -- no offset, x * step
#   board_mode 1:  USB Interface Board, +/- 5.0 V -- (x - 32768) * step
#   board_mode 13: Recording Controller, +/- 10.24 V -- (x - 32768) * step
BOARD_ADC_PARAMS = {
    0:  (0,     0.000050354),
    1:  (32768, 0.00015259),
    13: (32768, 0.0003125),
}

# Signal-type codes (per-channel `signal_type` field in the header).
SIGNAL_TYPE_AMP = 0
SIGNAL_TYPE_AUX = 1
SIGNAL_TYPE_SUPPLY = 2
SIGNAL_TYPE_BOARD_ADC = 3
SIGNAL_TYPE_BOARD_DIN = 4
SIGNAL_TYPE_BOARD_DOUT = 5


def load_rhd_board_adc(paths) -> tuple[np.ndarray, float, list[str]]:
    """Load controller-box analog input channels from one or more .rhd files.

    "Board ADC" in the spec; called "Analog In" on the Intan Recording
    Controller's BNC ports and "Analog Inputs" on the older USB
    Interface Board's screw terminals. This is where Monitorio
    recordings normally route the photodiode signals.

    paths, return value, errors: see the load_rhd_aux docstring -- same
    semantics, different physical inputs.

    Sample rate is the full amplifier rate (these channels are sampled
    at the amp rate, not 1/4 of it like the headstage aux). Voltage
    conversion is board-mode dependent; values are returned in volts.
    """
    return _load_signal_type(paths, SIGNAL_TYPE_BOARD_ADC)


def load_rhd_aux(paths) -> tuple[np.ndarray, float, list[str]]:
    """Load headstage auxiliary input channels from one or more .rhd files.

    These are the AUX1/AUX2/AUX3 inputs on each RHD chip's headstage --
    typically used for accelerometer or temperature sensor signals.
    NOT the BNC ports on the recording controller box (use
    load_rhd_board_adc for those).

    paths: a single path (str or Path) or an iterable of paths. When a
           list is given, the files are loaded in order and samples
           concatenated along the time axis. All files must share the
           same enabled-channel layout and sample rate.

    Returns:
        samples: (n_channels, n_total_samples) float64 in volts
        sample_rate: float, sample rate (Hz). For aux inputs that's
                     amp_rate / 4 (e.g. 20 kHz amps -> 5 kHz aux).
        channel_names: list[str] of native channel names, in the same
                       order as the rows of `samples`.

    Raises:
        ValueError: bad magic number, no enabled channels of the
                    requested type, inconsistent layouts across files,
                    or a file size that doesn't divide cleanly into
                    data blocks.
    """
    return _load_signal_type(paths, SIGNAL_TYPE_AUX)


# ----- shared multi-file driver --------------------------------------

def _load_signal_type(paths, signal_type: int):
    if isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(p) for p in paths]
    if not path_list:
        raise ValueError("paths is empty")

    arrays = []
    sample_rate = None
    channel_names = None
    for p in path_list:
        samples_v, rate, names = _load_one(p, signal_type=signal_type)
        if sample_rate is None:
            sample_rate = rate
            channel_names = names
        else:
            if rate != sample_rate:
                raise ValueError(
                    f"file {p} has sample rate {rate}, but earlier "
                    f"files had {sample_rate}; refusing to concatenate "
                    f"recordings with mismatched rates."
                )
            if names != channel_names:
                raise ValueError(
                    f"file {p} has channels {names}, but earlier "
                    f"files had {channel_names}; refusing to concatenate "
                    f"recordings with different channel layouts."
                )
        arrays.append(samples_v)

    if len(arrays) == 1:
        samples = arrays[0]
    else:
        samples = np.concatenate(arrays, axis=1)
    return samples, float(sample_rate), list(channel_names)


# ----- per-file loading -----------------------------------------------

def _load_one(path: Path, *, signal_type: int) -> tuple[np.ndarray, float, list[str]]:
    """Read one .rhd file and return samples for the requested signal type.

    signal_type: SIGNAL_TYPE_AUX or SIGNAL_TYPE_BOARD_ADC. Other types
                 (amp, supply, etc.) aren't supported -- the data block
                 layout differs and we don't have a reason to need them.
    """
    if signal_type not in (SIGNAL_TYPE_AUX, SIGNAL_TYPE_BOARD_ADC):
        raise ValueError(
            f"_load_one only supports AUX ({SIGNAL_TYPE_AUX}) or "
            f"BOARD_ADC ({SIGNAL_TYPE_BOARD_ADC}); got {signal_type}"
        )

    with open(path, "rb") as fh:
        header = _parse_header(fh)
        # Slurp the data section into memory. RHD files are typically
        # tens to a few hundred MB, well within reasonable RAM.
        data_section = fh.read()

    n_amp = header["counts"][SIGNAL_TYPE_AMP]
    n_aux = header["counts"][SIGNAL_TYPE_AUX]
    n_supply = header["counts"][SIGNAL_TYPE_SUPPLY]
    n_board_adc = header["counts"][SIGNAL_TYPE_BOARD_ADC]
    n_temp = header["num_temp_sensors"]
    has_digital = (header["counts"][SIGNAL_TYPE_BOARD_DIN] > 0)

    n_target = n_aux if signal_type == SIGNAL_TYPE_AUX else n_board_adc
    if n_target == 0:
        what = "auxiliary input" if signal_type == SIGNAL_TYPE_AUX else "controller-box analog input"
        raise ValueError(
            f"{path}: no {what} channels are enabled in this recording."
        )

    N, block_size, n_blocks = _detect_block_size(
        data_section_bytes=len(data_section),
        n_amp=n_amp, n_aux=n_aux, n_supply=n_supply,
        n_temp=n_temp, n_board_adc=n_board_adc,
        has_digital=has_digital,
    )
    if N % 4:
        raise ValueError(f"{path}: N={N} is not divisible by 4 (corrupt file?)")

    # Within each data block, in order:
    #   N x int32 timestamps                   = 4*N bytes
    #   n_amp channels x N x uint16            = 2*N*n_amp bytes
    #   n_aux channels x (N/4) x uint16        = 2*(N/4)*n_aux bytes
    #   n_supply x 1 x uint16                  = 2*n_supply bytes
    #   n_temp x 1 x int16                     = 2*n_temp bytes
    #   n_board_adc x N x uint16               = 2*N*n_board_adc bytes
    #   has_digital ? N x uint16 : 0           = 2*N*has_digital bytes
    if signal_type == SIGNAL_TYPE_AUX:
        per_chan = N // 4
        block_offset = 4 * N + 2 * N * n_amp
        out_rate = header["sample_rate"] / 4.0
    else:  # BOARD_ADC
        per_chan = N
        block_offset = (
            4 * N + 2 * N * n_amp + 2 * (N // 4) * n_aux
            + 2 * n_supply + 2 * n_temp
        )
        out_rate = header["sample_rate"]
    block_bytes = 2 * per_chan * n_target

    flat = np.frombuffer(data_section, dtype=np.uint8)
    if flat.size != n_blocks * block_size:
        raise ValueError(
            f"{path}: data section ({flat.size} bytes) doesn't divide "
            f"into {n_blocks} blocks of {block_size} bytes each"
        )
    blocks = flat.reshape(n_blocks, block_size)
    target_bytes = blocks[:, block_offset:block_offset + block_bytes]
    target_u16 = target_bytes.reshape(-1).view(np.uint16).reshape(
        n_blocks, n_target, per_chan,
    )
    samples_u16 = target_u16.transpose(1, 0, 2).reshape(n_target, n_blocks * per_chan)

    if signal_type == SIGNAL_TYPE_AUX:
        samples_v = samples_u16.astype(np.float64) * AUX_VOLTS_PER_STEP
    else:
        bm = header["board_mode"]
        if bm not in BOARD_ADC_PARAMS:
            raise ValueError(
                f"{path}: unsupported board_mode {bm} for board ADC scaling. "
                f"Known modes: {sorted(BOARD_ADC_PARAMS)}."
            )
        offset, step = BOARD_ADC_PARAMS[bm]
        samples_v = (samples_u16.astype(np.int32) - offset).astype(np.float64) * step

    chan_names = [
        ch["native_name"] for ch in header["channels"]
        if ch["signal_type"] == signal_type and ch["enabled"]
    ]
    return samples_v, out_rate, chan_names


def _detect_block_size(
    *,
    data_section_bytes: int,
    n_amp: int, n_aux: int, n_supply: int, n_temp: int,
    n_board_adc: int, has_digital: bool,
) -> tuple[int, int, int]:
    """Find the value of N (60 or 128) that makes blocks divide evenly.

    N is the number of amplifier samples per data block. The spec says
    N=60 for the 256-channel RHD USB interface board with the older
    Intan software, N=128 for the 512/1024-channel Intan Recording
    Controller and the newer RHX software with either board. We don't
    know which from the header alone, so we try both.
    """
    candidates = []
    for N in (128, 60):
        if N % 4:
            continue
        per_aux = N // 4
        block_size = (
            4 * N
            + 2 * N * n_amp
            + 2 * per_aux * n_aux
            + 2 * n_supply
            + 2 * n_temp
            + 2 * N * n_board_adc
            + 2 * N * (1 if has_digital else 0)
        )
        if data_section_bytes % block_size == 0:
            n_blocks = data_section_bytes // block_size
            candidates.append((N, block_size, n_blocks))
    if not candidates:
        raise ValueError(
            f"could not determine N (samples per block): file's data "
            f"section ({data_section_bytes} bytes) doesn't divide "
            f"evenly with N=60 or N=128. The file may be truncated or "
            f"in an unsupported variant of the RHD format."
        )
    # Prefer N=128 when both match (newer / more common); only happens
    # when channel layout is degenerate.
    return candidates[0]


# ----- header parsing -------------------------------------------------

def _parse_header(fh) -> dict:
    """Parse the Standard Intan RHD Header from `fh` and return a dict.

    Leaves the file pointer immediately past the header so the caller
    can read the data section. Raises ValueError on bad magic.
    """
    magic = _read_uint32(fh)
    if magic != RHD_MAGIC:
        raise ValueError(
            f"not an RHD file: magic number 0x{magic:08x} (expected "
            f"0x{RHD_MAGIC:08x})"
        )
    version_major = _read_int16(fh)
    version_minor = _read_int16(fh)
    sample_rate = _read_single(fh)

    # Bandwidth + filter parameters -- not used downstream but we have
    # to step past them.
    _read_int16(fh)        # dsp_enabled
    _read_single(fh)       # dsp_cutoff actual
    _read_single(fh)       # lower_bandwidth actual
    _read_single(fh)       # upper_bandwidth actual
    _read_single(fh)       # dsp_cutoff desired
    _read_single(fh)       # lower_bandwidth desired
    _read_single(fh)       # upper_bandwidth desired
    _read_int16(fh)        # notch filter mode
    _read_single(fh)       # impedance test freq desired
    _read_single(fh)       # impedance test freq actual

    notes_1 = _read_qstring(fh)
    notes_2 = _read_qstring(fh)
    notes_3 = _read_qstring(fh)

    num_temp_sensors = 0
    if (version_major, version_minor) >= (1, 1):
        num_temp_sensors = _read_int16(fh)
    board_mode = 0
    if (version_major, version_minor) >= (1, 3):
        board_mode = _read_int16(fh)
    if (version_major, version_minor) >= (2, 0):
        _read_qstring(fh)  # reference_channel name

    num_signal_groups = _read_int16(fh)
    channels: list[dict] = []
    for _ in range(num_signal_groups):
        _read_qstring(fh)  # group name
        _read_qstring(fh)  # group prefix
        group_enabled = _read_int16(fh)
        group_n_channels = _read_int16(fh)
        _read_int16(fh)    # num amplifier channels (in this group)
        for _ in range(group_n_channels):
            ch = _read_channel(fh)
            # The spec says we save channel descriptors for every
            # channel in the group regardless of whether its parent
            # group is enabled, but we trust `enabled` per channel.
            if not group_enabled:
                ch["enabled"] = 0
            channels.append(ch)

    # Tally enabled channels per signal type for downstream block-size
    # calculations.
    counts = {st: 0 for st in (
        SIGNAL_TYPE_AMP, SIGNAL_TYPE_AUX, SIGNAL_TYPE_SUPPLY,
        SIGNAL_TYPE_BOARD_ADC, SIGNAL_TYPE_BOARD_DIN, SIGNAL_TYPE_BOARD_DOUT,
    )}
    for ch in channels:
        if ch["enabled"]:
            counts[ch["signal_type"]] = counts.get(ch["signal_type"], 0) + 1

    return {
        "version": (version_major, version_minor),
        "sample_rate": float(sample_rate),
        "board_mode": board_mode,
        "num_temp_sensors": num_temp_sensors,
        "channels": channels,
        "counts": counts,
        "notes": (notes_1, notes_2, notes_3),
    }


def _read_channel(fh) -> dict:
    native_name = _read_qstring(fh)
    custom_name = _read_qstring(fh)
    native_order = _read_int16(fh)
    custom_order = _read_int16(fh)
    signal_type = _read_int16(fh)
    enabled = _read_int16(fh)
    chip_channel = _read_int16(fh)
    board_stream = _read_int16(fh)
    _read_int16(fh)   # spike voltage trigger mode
    _read_int16(fh)   # spike voltage threshold
    _read_int16(fh)   # spike digital trigger channel
    _read_int16(fh)   # spike digital edge polarity
    _read_single(fh)  # electrode impedance magnitude
    _read_single(fh)  # electrode impedance phase
    return {
        "native_name": native_name,
        "custom_name": custom_name,
        "native_order": native_order,
        "custom_order": custom_order,
        "signal_type": signal_type,
        "enabled": enabled,
        "chip_channel": chip_channel,
        "board_stream": board_stream,
    }


# ----- low-level primitives -------------------------------------------

def _read_uint32(fh) -> int:
    return struct.unpack("<I", fh.read(4))[0]


def _read_int16(fh) -> int:
    return struct.unpack("<h", fh.read(2))[0]


def _read_single(fh) -> float:
    return struct.unpack("<f", fh.read(4))[0]


def _read_qstring(fh) -> str:
    """Read a Qt-style length-prefixed UTF-16 string.

    Format per RHD spec: uint32 byte_length (little endian), then
    `byte_length` bytes interpreted as UTF-16 little-endian
    (`byte_length` is bytes, not characters). A byte_length of
    0xFFFFFFFF marks a null string -- we return ''.
    """
    n_bytes = _read_uint32(fh)
    if n_bytes == 0xFFFFFFFF or n_bytes == 0:
        return ""
    return fh.read(n_bytes).decode("utf-16-le", errors="replace")
