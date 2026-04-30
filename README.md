# Monitorio

Hardware + software for sub-frame-accurate synchronization between a
displayed video and an external recording, using on-screen photodiode
sync tags.

The video tagging script overlays a small array of black/white "bit
circles" onto each frame of a video; the bits encode the frame number
in reflected-binary [Gray code](https://en.wikipedia.org/wiki/Gray_code).
A small PCB of photodiodes (Monitorio) is mounted on the display so
each photodiode covers one bit position. The photodiodes' analog
outputs go into an NI-DAQmx data acquisition card, which records them
alongside whatever else the experiment is sampling. Decoding that DAQ
trace recovers, for each DAQ sample, which video frame was on screen
at that moment.

You only need the Monitorio PCB if you want a turnkey assembly.
**The software side works with any photodiode + amplifier hardware**
that produces an analog voltage proportional to local screen
illumination, fast enough to settle within one video frame. A breadboard
build with discrete photodiodes and a single quad op-amp is sufficient;
the calibration tool will locate them automatically wherever you stick
them on the screen.


## Repository layout

```
Hardware/
  Monitorio v1.0/          KiCad project + Gerbers for the v1.0 PCB
  Monitorio_v1.1/          v1.1 (4 photodiodes, OPA4323 quad TIA, RJ45 out)
Source/
  add_video_sync_tags.py   CLI: tag a video with sync circles
  decode_sync_tags.py      Python module: recover video frame # per DAQ sample
  decodeSyncTags.m         MATLAB wrapper around decode_sync_tags.py
  setupMonitorioPython.m   MATLAB helper: pyenv setup against the repo venv
  calibration/             Python package for the calibration pipeline
    daq.py                 NI-DAQmx wrapper
    display.py             pygame-ce fullscreen drawing primitives
    procedure.py           baseline / localize / refine / rise-time / crosstalk
    plot.py                matplotlib visualization of calibration results
    gray.py                reflected-binary Gray code encode/decode
    io.py                  pipeline cache (.npz) so reruns skip the slow steps
    scripts/
      calibrate.py         CLI: full calibration end-to-end -> JSON output
      smoke_test_*.py      individual stage tests for development
requirements.txt           Python dependencies (driver + ffmpeg are external)
```


## Hardware

The Monitorio PCB (see `Hardware/Monitorio_v1.1/`) carries four surface-
mount photodiodes (`D1`-`D4`), an OPA4323 quad op-amp configured as four
transimpedance amplifiers, and an RJ45 output jack. Each PD's amplified
voltage is one channel; the cable carries all four plus ground/shield to
a breakout into the DAQ. Run on a single 5 V supply, so signals are
0-5 V (typically ~50% dynamic range with the default photodiode
sensitivity and gain).

Pieces of the PCB design files:

- `*.kicad_sch`, `*.kicad_pcb` -- KiCad schematic and board.
- `CAM/` -- Gerbers and drill file for fab.
- `Photodiode Amplifier Reference Circuit.pdf` -- the TI app note the
  TIA topology is taken from.

The software does not know or care about the PCB. Channel layout is
detected from photodiode response during calibration, and any other
analog source that meets the requirements below can be substituted.

### What other photodiode hardware needs to do

If you're not using the Monitorio PCB, your replacement must:

1. Output an analog voltage that increases monotonically with light
   level on each photodiode, with at least a few hundred mV of dynamic
   range between black and white screen.
2. Settle within one video frame interval (a TIA bandwidth of ~1 kHz is
   plenty for 60 Hz video; the calibration procedure measures actual
   rise time and reports it).
3. Plug into an NI-DAQmx-compatible analog input device. Channels can be
   single-ended (RSE/NRSE) or differential (DIFF/PSEUDO_DIFF); the
   calibration script accepts a `--terminal-config` flag.


## Software requirements

Driver and OS-level tools (install separately, **not** via pip):

- **NI-DAQmx driver** -- vendor download from National Instruments,
  required for `nidaqmx` to talk to any DAQ device. Windows builds are
  the most-tested target; Linux builds are also available from NI.
- **ffmpeg** (and `ffprobe`) -- must be on `PATH`. The video tagging
  script shells out to ffmpeg for decoding/encoding; the decoder uses
  ffprobe to read fps and frame count from the tagged video.
- **Python 3.12+** -- developed against 3.14. Any 3.12+ should work.

Python packages (install with `pip install -r requirements.txt` inside
a virtualenv):

- `pygame-ce>=2.5` -- fullscreen rendering for the calibration patterns.
- `numpy>=1.26`
- `nidaqmx>=1.0` -- requires the NI driver above.
- `matplotlib>=3.8` -- optional; used only for the inspection plots that
  pop up at the end of calibration.


## Pipeline overview

```
   +-----------+       +---------------+       +---------------+
   | calibrate |  -->  | tag the video |  -->  | record video  |
   |  (one-off)|       |   (per video) |       | + DAQ trace   |
   +-----------+       +---------------+       +---------------+
        |                      ^                       |
        v                      |                       v
   calibration JSON  ----------+              decode DAQ trace
                                              into frame numbers
                                              (downstream script)
```

1. **Calibrate** the rig once after physically mounting the PD board.
   This locates each photodiode on the screen, picks bit-circle and
   background-circle radii, measures monitor rise/fall time and channel
   crosstalk, and writes everything to a JSON file.
2. **Tag** each video you'll display: this writes a new copy with
   Gray-coded bit circles overlaid at the calibrated positions.
3. **Display** the tagged video on the calibrated rig and **record**
   the photodiode channels on the DAQ alongside whatever else the
   experiment is sampling.
4. **Decode** the DAQ trace with `Source/decode_sync_tags.py` (see
   "`decode_sync_tags.py` reference" below): pass in the recorded `(n_channels, n_samples)`
   array, the tagged video path, and the calibration JSON; get back a
   table of `(frame_number, sample_index)` rows. The decoder reads the
   tagger's `<output_video>.tags.json` sidecar to get the sync-bit
   setting and channel-to-bit assignment automatically.


## Quick start

```bash
# One-off, after wiring up the PD board and pointing it at the display.
# Runs the full calibration interactively (instruction screens + plots,
# asks before saving). Output: calibration_<timestamp>.json
venv/Scripts/python Source/calibration/scripts/calibrate.py

# Tag a video against an existing calibration JSON.
venv/Scripts/python Source/add_video_sync_tags.py \
    in.mp4 out.mp4 \
    --calibration-file calibration_20260417-162539.json

# Or do both in one step (calibrates first, then tags using its JSON):
venv/Scripts/python Source/add_video_sync_tags.py \
    in.mp4 out.mp4 --calibrate

# Manual override -- skip calibration entirely if you have known
# good positions for the rig:
venv/Scripts/python Source/add_video_sync_tags.py \
    in.mp4 out.mp4 \
    --bit-xs 75,140,205,270 --bit-ys 1850 \
    --bit-radius 15 --background-radius 25
```


## `calibrate.py` reference

Runs all of the following, in order, and writes the result to a single
JSON file:

1. Baseline characterization -- per-channel noise floor (black screen)
   and full-scale response (white screen). Channels whose dynamic
   range doesn't exceed the noise floor by 10x are flagged as "dead"
   and skipped for the rest of the calibration.
2. Coarse structured-light localization -- displays Gray-coded stripe
   patterns at successively finer bit widths along each axis. Each PD
   reads its own position one bit per pattern; the result locates each
   PD to within 32 px on each axis. ~1 minute on a typical rig.
3. Fine bar-sweep refinement -- a thin bar marches across each PD's
   coarse window; a noise-rejected weighted centroid pinpoints the PD's
   center to sub-pixel precision and records its FWHM (used as a
   diameter estimate).
4. Bit-circle radius selection per PD -- sized from FWHM to saturate
   the PD without encroaching on neighbors.
5. Background-circle radius selection per PD -- sized from the bar-
   sweep tails so the PD doesn't see past the edge of its black
   background when the bit is off.
6. Rise/fall-time measurement -- single-channel high-rate capture of
   black-to-white and white-to-black transitions on each PD. Useful as
   a per-rig sanity check (LCDs are slow, OLEDs are fast).
7. Crosstalk verification -- light each PD's chosen bit circle in turn
   and read every channel; reports the worst off-diagonal leak.

After all measurements complete, three matplotlib figures pop up
(spatial response, temporal response, crosstalk heatmap) for visual
sanity check. The script then asks whether to save the JSON.

The JSON contains: monitor index/resolution, DAQ device + product type
+ terminal config + sample rate, and per-PD `(x_px, y_px, bit_radius_px,
background_radius_px, baseline_dark_v, baseline_bright_v,
dynamic_range_v, rise_duration_s, fall_duration_s, ...)`. Plus the full
crosstalk matrix.

Common flags:

- `--display N` -- pick which monitor (default 0).
- `--device NAME` -- pick which DAQ device, e.g. `Dev1`.
- `--terminal-config {RSE,DIFF,NRSE,PSEUDO_DIFF}` -- AI terminal
  configuration. RSE is the default. In DIFF/PSEUDO_DIFF the script
  automatically restricts itself to the half of physical channels that
  the device exposes as differential positive inputs.
- `--cache PATH` -- cache the slow baselines + coarse + fine
  measurements (~1 min) so a re-run picks up where the last left off.
  Default is `calibration_cache.npz` in CWD. Pass `--force` to ignore
  any existing cache.
- `--no-plot`, `--no-confirm` -- non-interactive batch mode.
- `--crosstalk-threshold PCT` -- pass/fail threshold (default 5%).


## `add_video_sync_tags.py` reference

Reads `video_in`, writes `video_out` with a Gray-coded sync-tag overlay
at every frame. Audio (if present) is copied through unchanged.

Parameter resolution order (per parameter, top wins):

1. Value passed explicitly on the CLI (`--bit-xs`, `--bit-radius`, etc.)
2. Calibration JSON (`--calibration-file` or `--calibrate`).
3. Error -- there is no "default" because sensible values depend on
   the specific rig.

Frame numbers are 1-indexed. With *N* sync tags, the encoder cycles
through `2**N` distinct codes. **Videos longer than `2**N` frames wrap**
through repeated cycles -- e.g. with 4 photodiodes (N=4, cycle=16), a
1000-frame video produces ~62 cycles. The reflected-binary Gray code
preserves the single-bit-change-per-step property at the wrap edge, so
mid-transition robustness still holds. The decoder is responsible for
disambiguating which cycle each sample belongs to (typically using its
own timestamp and the known frame rate). One quirk: frame `2**N`
(`16, 32, 48, ...`) Gray-encodes to all bits off, which is visually
indistinguishable from no tag -- the decoder must accept that case.

Selected flags:

- `--calibration-file PATH` -- use a saved calibration JSON.
- `--calibrate` -- run `calibrate.py` first and use its output, all in
  one command. Forwards `--display`, `--device`, `--cache`, `--force`,
  `--terminal-config` to the subprocess.
- `--bit-xs A,B,C`, `--bit-ys A,B,C`, `--bit-radius N`,
  `--background-radius N` -- manual overrides for any of the calibrated
  values. All are in **screen pixel coords** (the same coord system the
  calibration JSON uses), not video-frame coords.
- `--sync-bit` / `--no-sync-bit` -- whether to reserve the first PD as
  an always-on "video active" indicator. **Enabled by default.** When
  on, PD index 0 in the calibration JSON (= the lowest-numbered live
  AI channel, by physical pin order) is lit on every video frame, and
  the remaining n-1 PDs Gray-encode the frame number. This lets the
  decoder cleanly distinguish "video off" (sync dark) from any frame
  state (sync lit) -- without it, frame numbers at every multiple of
  2**n_bits encode to all-dark and look identical to "off." Cost: the
  cycle drops by 2x (16 -> 8 with 4 PDs), so the decoder has to
  disambiguate cycles from timing more often. The script prints the
  sync/frame-bit channel assignment at startup so a decoder author can
  confirm it.
- `--screen-size WxH` -- screen pixel dimensions of the display the PD
  board is mounted on, e.g. `2400x1600`. Required unless
  `--calibration-file` is given (then read from the JSON's
  `monitor.width/height`). The output video is rendered at whatever size
  preserves the screen aspect ratio with **minimal padding** around the
  input (no upscaling, no cropping); tag positions and radii are scaled
  by `output_w / screen_w` so they land at the correct screen pixels
  when the output is shown full-screen on this display. The script
  errors out if the input is larger than the screen in either dimension
  (cropping would lose content), and warns if the scaled tag radius
  drops below ~3 px (likely too small to read reliably -- use a larger
  input video).
- `--codec NAME` -- ffmpeg encoder. Defaults to `libx264` (CPU; always
  works). On a machine with an NVIDIA GPU and an ffmpeg build linked
  against the NVIDIA SDK, `--codec h264_nvenc` (or `hevc_nvenc`) is
  typically 5-10x faster. The script does a build-level check at
  startup and errors out if the encoder isn't compiled in; runtime
  failures (e.g. ffmpeg has the encoder but the GPU isn't supported)
  surface as an ffmpeg error during the encode -- if that happens,
  fall back to the default.
- `--preset NAME` -- encoder preset. Defaults are codec-dependent:
  `veryfast` for libx264, `p4` (medium) for NVENC. Pass any preset
  string the chosen encoder accepts.
- `--quality N` -- visual-quality knob; lower = higher quality, larger
  file. Mapped to `-crf` for libx264 and `-cq` for NVENC. Default 18
  is perceptually lossless on both.
- `--progress` -- per-frame progress.


## `decode_sync_tags.py` reference

After recording the photodiode signals on a DAQ alongside whatever else
the experiment is sampling, `Source/decode_sync_tags.py` recovers the mapping
between video frame number and DAQ sample index.

The repo deliberately doesn't include code for parsing your specific
DAQ's file format -- load the samples into a numpy array however you
like, then call:

```python
from decode_sync_tags import decode_sync_tags

result = decode_sync_tags(
    samples,                           # (n_channels, n_samples) array
    sample_rate=50_000,                # Hz
    video_path="exp01_tagged.mp4",     # the tagger's output
    calibration_path="cal.json",       # the calibration JSON
    scale="intan_aux",                 # samples are Intan ADC steps
                                       #   ('volts', 'intan_aux',
                                       #   'intan_supply', etc., or a
                                       #   numeric multiplier)
    output_path="exp01_frames.csv",    # CSV table to write
    metadata="2026-04-30 rig A trial 7",
)
# result.frame_table is an (n_frames, 2) int64 array of
#   (frame_number, sample_index) rows.
```

The tagger writes a sidecar `<output_video>.tags.json` next to its
output that records the sync-bit setting and channel-to-bit assignment;
the decoder reads it automatically. Pass `sync_bit_override=True/False`
to force a value (useful if the sidecar is missing or wrong).

The decoder operates in **strict mode**: it assumes the recording
contains exactly one complete video playback bracketed by "video off"
padding on both sides, and raises `RuntimeError` otherwise. Specifically
it errors out if:

- No bimodal signal is detected on the sync-bit channel (or any channel
  in `--no-sync-bit` mode) -- recording probably has no video, the
  channel order doesn't match calibration, or the scale factor produced
  near-constant voltages.
- More than one video segment is detected (chunk the recording yourself
  before passing it in).
- The first decoded frame doesn't match Gray-code value 1 (segment
  doesn't start at frame 1, recording is partial, or threshold/debounce
  picked the wrong first transition).
- The decoded total frame count doesn't match the source video's frame
  count to within ±1 (recording truncated, or unwrap miscounted).

Softer issues land in `result.warnings_` (and the CSV's `# warning:`
header lines): per-channel threshold drift vs. calibration, individual
dropped frames recovered via timing, fps mismatch between the recording
and the video file's metadata, etc.

Specific case worth flagging in **--no-sync-bit mode**: if the video's
total frame count is an exact multiple of `2**n_pds` (i.e. the cycle
length), the very last frame Gray-encodes to all zeros and is
indistinguishable from the post-video "off" period. The tagger handles
this by default by appending a single black-content frame so the output
isn't an exact cycle multiple (`--no-pad-for-unambiguous-end` opts
out); the decoder also has a safety-net that detects the case if it
slips through, synthesizes the missing last frame at
`last_real_sample + sample_rate/fps`, and emits a warning suggesting
`--sync-bit` to remove the ambiguity. With `--sync-bit` on (the
default) the issue can't arise -- the sync PD is lit on every video
frame.

Pass `verbose=1` for a one-line summary of decoder decisions
(input shape, sync-bit source, segment bounds, decoded count) printed
to stderr; `verbose=2` adds per-channel thresholds, SNR values,
detected segments, and per-transition unwrap accounting. Useful when a
real-rig recording isn't decoding the way you expect.

The internal bimodal threshold is found per channel via Otsu's method
on the recording itself, with the calibration JSON's
`baseline_dark_v` / `baseline_bright_v` used only as a sanity check.
This makes the decoder unit-agnostic given the right `scale`: NI DAQ
records in volts (`scale="volts"` or `1.0`); Intan auxiliary input
channels record raw ADC steps that convert to volts at
`0.0000374 V/step` (`scale="intan_aux"`). The `SCALE_PRESETS` dict in
`decode_sync_tags.py` lists the available shorthand names. Also be aware that
the Intan aux input ranges only ~0 to 2.45 V -- verify the PD signal
doesn't clip at the top of that range.

The CSV output has a `#`-prefixed header carrying the source video
path, calibration path, sample rate, fps, sync-bit setting, cycle,
per-channel detected thresholds, segment bounds, the `metadata` string
the user passed in, and any warnings. Two data columns:
`frame_number,sample_index`. `frame_number` is 1-indexed; `sample_index`
is into the original `samples` array.


## MATLAB wrapper

`Source/decodeSyncTags.m` calls the Python decoder from MATLAB via
MATLAB's [Python Interface](https://www.mathworks.com/help/matlab/call-python-libraries.html).
`Source/setupMonitorioPython.m` configures `pyenv` to point at the
repo's `venv` (auto-discovered alongside the script) in
`OutOfProcess` mode.

Setup -- once per MATLAB session, BEFORE any other Python calls:

```matlab
addpath('path/to/Monitorio/Source');
setupMonitorioPython();
```

Then call `decodeSyncTags` like the Python function:

```matlab
% NI-DAQmx recording (already in volts):
result = decodeSyncTags(samples, 50000, 'tagged.mp4', 'cal.json');

% Intan recording (raw aux-input ADC steps):
result = decodeSyncTags(samples, 30000, 'tagged.mp4', 'cal.json', ...
    'Scale', 'intan_aux', ...
    'OutputPath', 'frames.csv', ...
    'Metadata', 'rig A trial 7', ...
    'Verbose', 1);

% frameTable is N-by-2 double; columns are 1-indexed frame_number and
% 0-indexed sample_index (matches Python). Add 1 if you want a
% MATLAB-style 1-based sample index.
sampleIndex1Based = result.frameTable(:, 2) + 1;
```

`samples` must be `n_channels`-by-`n_samples` with channels in the
same order as the calibration JSON's photodiode list (= physical AI
pin order). Channel order, sync-bit setting, and per-bit channel
assignment are all read from the tagger's `<video>.tags.json` sidecar
automatically.

If you need to use a venv at a different path:

```matlab
setupMonitorioPython('VenvPath', 'C:\path\to\some\other\venv');
```

`pyenv` cannot be reconfigured after Python has been loaded in a
session, so if you previously imported a different Python from MATLAB,
restart MATLAB before re-running setup.


## Smoke tests

`Source/calibration/scripts/smoke_test_*.py` exercise individual
calibration stages (baselines, DAQ, display, localize, refine,
rise-time, crosstalk). Useful for development and for diagnosing a
rig where the full `calibrate.py` is misbehaving. They all default to
RSE; modify them if you need a different terminal config.


## Automated test suite

`tests/` holds pytest-based regression tests for the parts of the
codebase that are testable without rig hardware: Gray-code encode/
decode and wrap invariants, parameter resolution priorities, terminal-
config-aware AI channel filtering (with nidaqmx mocked), and
end-to-end aspect-ratio scaling of the tagging pipeline against
synthetic ffmpeg videos.

```bash
pip install -r requirements-dev.txt
pytest                # from repo root; auto-discovers tests/
pytest tests/test_gray.py -v        # one file
pytest -k "aspect_scaling"          # one keyword
```

Tests that need ffmpeg auto-skip if it isn't on `PATH`; the nidaqmx
filter tests skip if the `nidaqmx` package isn't importable.


## Caveats and limitations

- LCDs have slow pixels and may show PWM in their backlight. The
  `--rise-time` measurement reports actual transition times; if the
  monitor's response is so slow that the per-frame transition isn't
  complete before the next refresh, the bit pattern is unreliable.
  OLED is recommended for timing-critical work.
- The calibration assumes the PDs don't move between calibration and
  measurement. If the PCB is jostled, re-calibrate.
- Multiplexer crosstalk on NI cards is real but tiny at the
  calibration's default 5 kHz DC sample rate. Higher sample rates
  (>= 50 kHz, e.g. for the rise-time measurement) are run single-channel
  to sidestep the mux entirely.
- The decoder is not included in this repository -- different consumers
  want different downstream behaviors (frame index per DAQ sample,
  per-frame onset timestamp, etc.). The calibration JSON contains
  everything needed; see the per-PD `baseline_dark_v` /
  `baseline_bright_v` fields for thresholding.
