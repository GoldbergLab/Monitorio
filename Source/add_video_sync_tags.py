"""Overlay Gray-coded frame-number sync tags onto a video.

Python port of addVideoSyncTags.m. Uses ffmpeg (must be on PATH) for
video I/O so no heavy Python video library is required; numpy does the
per-frame pixel math.

Each output frame has, at every screen-coord (bitXs[k], bitYs[k]):
  - a black filled disk of `background_radius` pixels (in screen px)
  - a white filled disk of `bit_radius` pixels INSIDE the black disk,
    drawn only when bit k of grayEncode(frame_number) is 1.

Coordinate system
-----------------
All tag positions and radii are in SCREEN pixel coordinates -- i.e. they
match what the calibration tool measured on the physical display the PD
board is mounted on. The output video is rendered at whatever resolution
preserves the screen aspect ratio with minimal padding around the input
(no upscaling, no cropping). Tag coords are then scaled by
`output_w / screen_w` to land at the correct screen pixels when the
output is shown full-screen on the calibrated display.

Concretely: if you calibrated on a 2400x1600 screen and feed in a
1920x1080 video, the output is padded to 2400x1620 (16:9 -> 3:2),
scale=1.0, tags drawn at the JSON's screen pixel coords. If you instead
feed a 1200x800 video (already 3:2), no pad is needed; output is
1200x800, scale=0.5, tags shrink and shift correspondingly. Either way
the output displays correctly when shown full-screen on the calibrated
display.

The screen size MUST be supplied (via the calibration JSON's
`monitor.width/height`, or via the `screen_size` argument); without it
the scale factor is undefined. The input video must not be larger than
the screen in either dimension (no cropping).

Frame numbers are 1-indexed (frame 1 is the first output frame) so the
encoding round-trips cleanly with any decoder expecting MATLAB-style
indexing. Sync tags are Gray-coded: at most one bit changes between
consecutive frames, so a photodiode decoder that samples mid-transition
can only ever be off by +-1, never misread a wildly different value.

If the video has more frames than n_bits sync tags can uniquely encode
(2**n_bits, e.g. 16 for n_bits=4), the encoded frame number wraps:
frame 1, 2, ..., 2**n_bits, 1, 2, ... The reflected-binary Gray code
preserves the "exactly one bit changes per step" property across the
wrap, including the wrap edge itself (gray(2**n_bits - 1) and gray(0)
also differ in only one bit). The decoder is responsible for resolving
which cycle each frame belongs to (typically using its sample
timestamp). Note that frame 2**n_bits, 2*2**n_bits, ... map to gray(0),
i.e. all bits off, which is visually indistinguishable from an
unillumated tag area; downstream code should tolerate that case.

Sync-bit mode (sync_bit=True / --sync-bit)
------------------------------------------
Reserves the FIRST photodiode (PD index 0 in the calibration JSON, which
corresponds to the first live AI channel in physical-pin order) as a
"video active" indicator: it's drawn lit on every video frame, never
dark. The remaining n-1 PDs encode the frame number in (n-1)-bit Gray
code, cycling every 2**(n-1) frames. The decoder can then unambiguously
tell "video off" (sync bit dark) from any frame state (sync bit lit),
which the default mode cannot guarantee because frame numbers at every
multiple of 2**n_bits Gray-encode to all bits off.

Cost: one bit, so cycle drops by 2x (e.g. 16 -> 8 with 4 PDs); frame
numbers wrap more often, and the decoder has to disambiguate cycles
more frequently from timing.

Channel-order convention: photodiodes appear in the calibration JSON in
the same order as their AI channels in physical-pin order (which is how
list_ai_channels enumerates them). The tagger reads them in that order:
when sync_bit is False, PD index k carries Gray-code bit k; when
sync_bit is True, PD index 0 is the sync bit and PD index k+1 carries
Gray-code bit k for k in [0, n-1). The decoder must apply the same
convention with the same calibration JSON.

Parameter resolution for each of bit_xs / bit_ys / bit_radius /
background_radius / screen_size:
    1. Value passed explicitly to add_video_sync_tags() wins.
    2. Otherwise, if calibration_file is given, its value is used
       (screen_size from the JSON's monitor.width/height).
    3. Otherwise, this function errors -- there's no safe default
       because sensible values depend on the specific Monitorio board +
       monitor combination.

CLI usage:
    python add_video_sync_tags.py IN.mp4 OUT.mp4 --calibration-file cal.json
    python add_video_sync_tags.py IN.mp4 OUT.mp4 \\
        --bit-xs 31,88,145,202 --bit-ys 40 \\
        --bit-radius 20 --background-radius 35 \\
        --screen-size 2400x1600
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Share the exact same Gray encode used by the calibration display.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration import gray


def add_video_sync_tags(
    video_in: str | Path,
    video_out: str | Path,
    *,
    calibration_file: str | Path | None = None,
    bit_xs=None,
    bit_ys=None,
    bit_radius: int | None = None,
    background_radius: int | None = None,
    screen_size: tuple[int, int] | None = None,
    sync_bit: bool = True,
    pad_for_unambiguous_end: bool = True,
    codec: str = "libx264",
    preset: str | None = None,
    quality: int = 18,
    show_progress: bool = False,
) -> int:
    """Write `video_out` = `video_in` with Gray-coded sync-tag overlays.

    Returns the number of frames written.

    screen_size: (width, height) in pixels of the physical display the PD
             board is mounted on. Tag positions and radii are interpreted
             as screen pixel coords and scaled to the output frame so
             that the output displays correctly when shown full-screen
             on this display. Read from the calibration JSON's
             monitor.width/height when not given explicitly.
    codec:   ffmpeg encoder name. 'libx264' (default, CPU) is always
             available. 'h264_nvenc' uses NVIDIA NVENC; ~5-10x faster
             on supported GPUs but requires both an NVENC-capable card
             and an ffmpeg build linked against the NVIDIA SDK. Other
             codec names (hevc_nvenc, etc.) are passed through to
             ffmpeg unchanged.
    preset:  encoder preset. Defaults are codec-dependent: 'veryfast'
             for libx264, 'p4' (medium) for *_nvenc. Pass any preset
             string ffmpeg's encoder accepts.
    quality: visual-quality knob. For libx264 it's mapped to -crf;
             for *_nvenc it's mapped to -cq. 18 is "perceptually
             indistinguishable from source" for both encoders.
    sync_bit: if True, dedicate the first PD as an always-on
             "video active" indicator and Gray-encode the frame number
             across the remaining n-1 PDs. Lets the decoder cleanly
             distinguish "video off" from "frame at a multiple of
             2**n_bits" (which would otherwise both look all-dark).
             Cycle length is halved (2**(n-1) instead of 2**n).
    pad_for_unambiguous_end: only relevant in --no-sync-bit mode. If the
             input video's frame count is an exact multiple of the cycle
             length, its last frame Gray-encodes to all-zeros, which is
             indistinguishable from the post-video off period in the
             recording. With this flag on (the default), the tagger
             appends a single extra black-content frame so the output's
             length isn't a cycle multiple and no frame falls on the
             ambiguous all-zeros codeword at the boundary. Has no effect
             in sync-bit mode (the sync PD is lit on every video frame
             and removes the ambiguity by construction).
    """
    video_in = Path(video_in)
    video_out = Path(video_out)
    if not video_in.exists():
        raise FileNotFoundError(video_in)

    xs, ys, bit_r, bg_r, screen = _resolve_parameters(
        calibration_file, bit_xs, bit_ys, bit_radius, background_radius,
        screen_size,
    )
    screen_w, screen_h = screen
    n_pds = max(len(xs), len(ys))
    if len(xs) == 1:
        xs = np.tile(xs, n_pds)
    if len(ys) == 1:
        ys = np.tile(ys, n_pds)
    if len(xs) != n_pds or len(ys) != n_pds:
        raise ValueError(
            f"bit_xs and bit_ys must be broadcast-compatible: "
            f"got len(bit_xs)={len(xs)}, len(bit_ys)={len(ys)}"
        )
    if sync_bit and n_pds < 2:
        raise ValueError(
            "sync_bit=True needs at least 2 photodiodes (one for the "
            "sync indicator + at least one for the frame number); only "
            f"got {n_pds}. Either add a PD or set sync_bit=False."
        )
    # Convention: PDs are listed in physical AI-channel order. With
    # sync_bit=True, PD index 0 is the always-on indicator and PDs
    # [1..n_pds) carry the n_pds-1 Gray-code frame bits. Without it, all
    # n_pds PDs are frame bits.
    sync_idx = 0 if sync_bit else None
    frame_bit_idx = list(range(1, n_pds)) if sync_bit else list(range(n_pds))
    n_frame_bits = len(frame_bit_idx)
    cycle = 1 << n_frame_bits  # grayEncode maps [0, 2**n) -> [0, 2**n) bijectively

    # Announce the bit assignment so a decoder author can confirm they're
    # reading the right channel for each role. Uses calibration JSON
    # channel names if available, else generic indices.
    cal_channels = _read_channel_names(calibration_file) if calibration_file else None
    def _label(i: int) -> str:
        return cal_channels[i] if cal_channels and i < len(cal_channels) else f"PD#{i}"
    if sync_bit:
        print(f"  sync bit: {_label(sync_idx)}", file=sys.stderr)
    print(
        "  frame bits: "
        + ", ".join(f"bit {k}={_label(idx)}" for k, idx in enumerate(frame_bit_idx)),
        file=sys.stderr,
    )

    info = _probe_video(video_in)
    w, h, fps, n_frames = info["width"], info["height"], info["fps"], info["n_frames"]

    if w > screen_w or h > screen_h:
        raise ValueError(
            f"input video ({w}x{h}) is larger than the calibrated screen "
            f"({screen_w}x{screen_h}) in at least one dimension. Cropping "
            f"would lose content; resize the video down first."
        )

    # Pad input minimally to match screen aspect ratio; output is then any
    # resolution with screen_w/screen_h aspect, no upscale, no crop. The
    # axis with the larger ratio of (screen / input) is the one we leave
    # untouched -- the other axis gets black-bar padding.
    if w * screen_h >= h * screen_w:
        # Input is wider than (or equal to) screen in aspect: pad height.
        out_w = w
        out_h = int(round(w * screen_h / screen_w))
    else:
        # Input is taller than screen in aspect: pad width.
        out_h = h
        out_w = int(round(h * screen_w / screen_h))
    pad_left = (out_w - w) // 2
    pad_top = (out_h - h) // 2
    needs_pad = (out_w != w) or (out_h != h)

    # Map screen-pixel coords -> output-frame coords. Both axes scale by
    # the same factor since out has matching aspect ratio.
    scale = out_w / screen_w
    scaled_xs = [int(round(float(x) * scale)) for x in xs]
    scaled_ys = [int(round(float(y) * scale)) for y in ys]
    scaled_bit_r = max(1, int(round(bit_r * scale)))
    scaled_bg_r = max(1, int(round(bg_r * scale)))

    if scaled_bit_r < 3:
        # The white bit circle has to be readable by the photodiode after
        # the screen scales the output back up. <3 px in the output frame
        # is so small it's likely to alias / disappear into pixel grid.
        print(
            f"  warning: scaled bit radius is {scaled_bit_r} px (screen "
            f"radius {bit_r}, scale {scale:.3f}); may be too small to "
            f"read reliably. Consider feeding a larger input video.",
            file=sys.stderr,
        )

    # Precompute per-bit disk masks (bounding box + boolean mask); they
    # never change frame to frame, so doing it once saves a meshgrid per
    # frame per bit.
    bg_masks = [
        _disk_mask(x, y, scaled_bg_r, out_w, out_h)
        for x, y in zip(scaled_xs, scaled_ys)
    ]
    bit_masks = [
        _disk_mask(x, y, scaled_bit_r, out_w, out_h)
        for x, y in zip(scaled_xs, scaled_ys)
    ]
    if any(m is None for m in bg_masks) or any(m is None for m in bit_masks):
        offscreen = [
            (i, scaled_xs[i], scaled_ys[i])
            for i, m in enumerate(bg_masks) if m is None
        ]
        print(
            f"  warning: tag(s) at output coords {offscreen} fall entirely "
            f"outside the {out_w}x{out_h} output frame and won't be drawn. "
            f"Check that the calibration screen size matches the rig.",
            file=sys.stderr,
        )

    if n_frames is not None and n_frames >= cycle:
        # Wrapping is supported by design (with limited PDs you usually have
        # to), but warn so a user who didn't realize they were wrapping isn't
        # surprised when their decoder sees repeated codes.
        n_cycles = n_frames / cycle
        sync_note = " (1 PD reserved as sync)" if sync_bit else ""
        print(
            f"  note: {n_frames} frames > 2**{n_frame_bits} = {cycle}"
            f"{sync_note}; frame number will wrap {n_cycles:.1f} times. "
            f"Decoder must disambiguate cycles via timestamp.",
            file=sys.stderr,
        )

    # Validate the chosen encoder up front so the user gets a clear error
    # instead of cryptic ffmpeg output if their build doesn't include it.
    if not _ffmpeg_has_encoder(codec):
        raise RuntimeError(
            f"ffmpeg on PATH does not list encoder {codec!r}. Available "
            f"encoders can be inspected with `ffmpeg -encoders`. NVENC "
            f"requires an NVIDIA GPU with an NVENC engine and an ffmpeg "
            f"build linked against the NVIDIA SDK; fall back to "
            f"--codec libx264 if NVENC isn't available."
        )

    is_nvenc = codec.endswith("_nvenc")
    eff_preset = preset if preset is not None else ("p4" if is_nvenc else "veryfast")
    quality_args = (
        ["-cq", str(int(quality))] if is_nvenc else ["-crf", str(int(quality))]
    )

    # ffmpeg read: decode video-only rgb24 stream to stdout.
    read_cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(video_in),
        "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    # ffmpeg write: take raw rgb24 from stdin (video), re-open input for
    # audio if present, mux video + original audio.
    write_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "-",
        "-i", str(video_in),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", codec, "-preset", eff_preset, "-pix_fmt", "yuv420p",
        *quality_args,
        "-c:a", "copy",
        str(video_out),
    ]

    read_proc = subprocess.Popen(
        read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    write_proc = subprocess.Popen(
        write_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    frame_bytes = w * h * 3
    # Reusable output buffer when padding; otherwise we modify a fresh
    # copy of each input frame.
    out_frame = (
        np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if needs_pad
        else None
    )

    frames_written = 0
    try:
        while True:
            raw = read_proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

            if needs_pad:
                out_frame[:] = 0
                out_frame[pad_top:pad_top + h, pad_left:pad_left + w] = frame
                draw_target = out_frame
            else:
                # np.frombuffer gives a read-only view; copy once so we
                # can draw into it.
                draw_target = frame.copy()

            frames_written += 1
            # Wrap modulo cycle so videos longer than 2**n_frame_bits frames
            # keep encoding cleanly; gray() over a full [0, 2**n) cycle has
            # the single-bit-change property at every step including the
            # wrap edge.
            g = int(gray.encode(np.int64(frames_written % cycle)))

            # Always-on black backgrounds for every PD.
            for m in bg_masks:
                _apply_mask(draw_target, m, 0)
            # Sync-bit PD (if enabled): always lit on every video frame.
            if sync_idx is not None:
                _apply_mask(draw_target, bit_masks[sync_idx], 255)
            # Frame-bit PDs: light bit k of g via PD at frame_bit_idx[k].
            for k in range(n_frame_bits):
                if (g >> k) & 1:
                    _apply_mask(draw_target, bit_masks[frame_bit_idx[k]], 255)

            write_proc.stdin.write(draw_target.tobytes())

            if show_progress and frames_written % 30 == 0:
                if n_frames:
                    pct = 100.0 * frames_written / n_frames
                    print(
                        f"\r  frame {frames_written}/{n_frames} ({pct:.1f}%)",
                        end="", flush=True,
                    )
                else:
                    print(f"\r  frame {frames_written}", end="", flush=True)

        # If we're in --no-sync-bit mode and the input ended on a frame
        # whose number is a multiple of the cycle, that frame Gray-encoded
        # to all zeros and is indistinguishable from "video off" in the
        # recording. Append one extra black-content frame (tagged with
        # gray((N+1) % cycle), which is gray(1) = bit 0 lit) so the
        # output's last frame is unambiguously visible. Skipped in
        # sync-bit mode (the sync PD already disambiguates) and skippable
        # via pad_for_unambiguous_end=False.
        padded_frames = 0
        if (
            pad_for_unambiguous_end
            and not sync_bit
            and frames_written > 0
            and frames_written % cycle == 0
        ):
            pad_target = (
                out_frame
                if out_frame is not None
                else np.zeros((out_h, out_w, 3), dtype=np.uint8)
            )
            pad_target[:] = 0
            for m in bg_masks:
                _apply_mask(pad_target, m, 0)
            g = int(gray.encode(np.int64((frames_written + 1) % cycle)))
            for k in range(n_frame_bits):
                if (g >> k) & 1:
                    _apply_mask(pad_target, bit_masks[frame_bit_idx[k]], 255)
            write_proc.stdin.write(pad_target.tobytes())
            frames_written += 1
            padded_frames = 1
            print(
                f"  note: appended 1 black padding frame to avoid the "
                f"all-zeros-last-frame ambiguity (input had {frames_written - 1} "
                f"frames, a multiple of cycle {cycle}). Disable with "
                f"pad_for_unambiguous_end=False / --no-pad-for-unambiguous-end.",
                file=sys.stderr,
            )

        if show_progress:
            print()
    finally:
        if write_proc.stdin:
            write_proc.stdin.close()
        read_rc = read_proc.wait()
        write_rc = write_proc.wait()

    if read_rc not in (0, None):
        err = read_proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg (read) exited with {read_rc}:\n{err}")
    if write_rc not in (0, None):
        err = write_proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg (write) exited with {write_rc}:\n{err}")

    # Write the sidecar manifest <video_out>.tags.json so the decoder can
    # recover sync-bit setting + per-bit channel assignment without the
    # operator having to hand-pass them. Best-effort: a write failure is
    # logged but doesn't kill the tagging run that already succeeded.
    sidecar_path = video_out.with_suffix(video_out.suffix + ".tags.json")
    sidecar = {
        "schema_version": 1,
        "tagged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_video": str(video_in),
        "output_video": str(video_out),
        "calibration_file": str(calibration_file) if calibration_file else None,
        "fps": float(fps),
        "n_frames_written": int(frames_written),
        "padded_frames": int(padded_frames),
        "screen_size": [int(screen_w), int(screen_h)],
        "output_size": [int(out_w), int(out_h)],
        "sync_bit": bool(sync_bit),
        "n_pds": int(n_pds),
        "n_frame_bits": int(n_frame_bits),
        "cycle": int(cycle),
        "channel_assignment": {
            "sync": (cal_channels[sync_idx] if cal_channels and sync_idx is not None else None),
            "frame_bits": [
                (cal_channels[i] if cal_channels else f"PD#{i}")
                for i in frame_bit_idx
            ],
        },
    }
    try:
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
    except OSError as e:
        print(f"  warning: could not write sidecar {sidecar_path}: {e}",
              file=sys.stderr)
    return frames_written


# ----- internals ------------------------------------------------------

def _resolve_parameters(
    calibration_file, bit_xs, bit_ys, bit_radius, background_radius, screen_size,
):
    cal_xs = cal_ys = cal_br = cal_bgr = cal_screen = None
    if calibration_file is not None:
        with open(calibration_file) as f:
            cal = json.load(f)
        pds = cal.get("photodiodes") or []
        if not pds:
            raise ValueError(f"{calibration_file} has no photodiodes")
        cal_xs = np.array([int(round(p["x_px"])) for p in pds])
        cal_ys = np.array([int(round(p["y_px"])) for p in pds])
        cal_br = int(max(p["bit_radius_px"] for p in pds))
        cal_bgr = int(max(p["background_radius_px"] for p in pds))
        monitor = cal.get("monitor") or {}
        if "width" in monitor and "height" in monitor:
            cal_screen = (int(monitor["width"]), int(monitor["height"]))

    xs = np.asarray(bit_xs, dtype=np.int64) if bit_xs is not None else cal_xs
    ys = np.asarray(bit_ys, dtype=np.int64) if bit_ys is not None else cal_ys
    br = int(bit_radius) if bit_radius is not None else cal_br
    bgr = int(background_radius) if background_radius is not None else cal_bgr
    screen = (
        (int(screen_size[0]), int(screen_size[1]))
        if screen_size is not None else cal_screen
    )

    missing = []
    if xs is None: missing.append("bit_xs")
    if ys is None: missing.append("bit_ys")
    if br is None: missing.append("bit_radius")
    if bgr is None: missing.append("background_radius")
    if screen is None: missing.append("screen_size")
    if missing:
        raise ValueError(
            f"Missing required value(s): {', '.join(missing)}. "
            f"Pass them explicitly, or supply calibration_file."
        )
    return xs, ys, br, bgr, screen


def _read_channel_names(calibration_file) -> list[str] | None:
    """Return the per-PD AI channel names from the calibration JSON, in
    list order. Returns None if `calibration_file` is None or the JSON
    has no 'photodiodes' / per-PD 'channel' fields.
    """
    if calibration_file is None:
        return None
    try:
        with open(calibration_file) as f:
            cal = json.load(f)
    except (OSError, ValueError):
        return None
    pds = cal.get("photodiodes") or []
    return [p.get("channel", "?") for p in pds] or None


_ENCODER_CACHE: set[str] | None = None


def _ffmpeg_has_encoder(name: str) -> bool:
    """Return True iff `ffmpeg -encoders` lists an encoder of the given name.

    Caches the parsed encoder set on first call; subsequent checks are
    free. Does NOT verify that the encoder will actually run on this
    machine (NVENC may be listed but the GPU/driver may reject it at
    runtime). Catches the common "ffmpeg build missing the encoder
    entirely" case up front; remaining failures will surface as ffmpeg
    errors during the real encode.
    """
    global _ENCODER_CACHE
    if _ENCODER_CACHE is None:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, check=False,
        )
        names = set()
        for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
            # Lines after the header look like:
            #   ' V....D libx264              libx264 H.264 ...'
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("V"):
                names.add(parts[1])
        _ENCODER_CACHE = names
    return name in _ENCODER_CACHE


def _probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on {path}:\n{proc.stderr.decode(errors='replace')}"
        )
    data = json.loads(proc.stdout)["streams"][0]
    w, h = int(data["width"]), int(data["height"])

    r_fr = data["r_frame_rate"]
    if "/" in r_fr:
        num, den = r_fr.split("/")
        fps = float(num) / float(den) if float(den) else float(num)
    else:
        fps = float(r_fr)

    n_frames = None
    if data.get("nb_frames") not in (None, "N/A"):
        try:
            n_frames = int(data["nb_frames"])
        except (TypeError, ValueError):
            pass
    if n_frames is None and data.get("duration") not in (None, "N/A"):
        try:
            n_frames = int(round(float(data["duration"]) * fps))
        except (TypeError, ValueError):
            pass
    return {"width": w, "height": h, "fps": fps, "n_frames": n_frames}


def _disk_mask(cx: int, cy: int, r: int, w: int, h: int):
    """Precompute a disk as (row_slice, col_slice, boolean mask), clipped to image."""
    col_min = max(0, cx - r)
    col_max = min(w - 1, cx + r)
    row_min = max(0, cy - r)
    row_max = min(h - 1, cy + r)
    if col_min > col_max or row_min > row_max:
        return None
    ys_local, xs_local = np.meshgrid(
        np.arange(row_min, row_max + 1),
        np.arange(col_min, col_max + 1),
        indexing="ij",
    )
    mask = (xs_local - cx) ** 2 + (ys_local - cy) ** 2 <= r * r
    return (slice(row_min, row_max + 1), slice(col_min, col_max + 1), mask)


def _apply_mask(img: np.ndarray, mask_info, value: int) -> None:
    if mask_info is None:
        return
    row_slice, col_slice, mask = mask_info
    img[row_slice, col_slice][mask] = value


# ----- CLI ------------------------------------------------------------

def _csv_ints(s: str):
    s = s.strip().lower().replace("x", ",")
    return [int(x) for x in s.split(",") if x]


def _run_calibrate_subprocess(
    *,
    out_path: Path,
    display: int | None,
    device: str | None,
    cache: str | None,
    force: bool,
    terminal_config: str | None,
) -> None:
    """Invoke calibration/scripts/calibrate.py, writing JSON to out_path.

    Forwards only the calibration-relevant options. calibrate.py still
    runs its own interactive UI (intro prompt, plots, save confirmation),
    so the operator has a chance to eyeball the result before the JSON
    is written and tagging begins.
    """
    script = Path(__file__).resolve().parent / "calibration" / "scripts" / "calibrate.py"
    cmd = [sys.executable, str(script), "--output", str(out_path)]
    if display is not None:
        cmd += ["--display", str(display)]
    if device is not None:
        cmd += ["--device", device]
    if cache is not None:
        cmd += ["--cache", cache]
    if force:
        cmd += ["--force"]
    if terminal_config is not None:
        cmd += ["--terminal-config", terminal_config]

    print(f"Running calibration: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"calibrate.py exited with code {proc.returncode}")
    if not out_path.exists():
        raise SystemExit(
            "calibrate.py finished but no JSON was written -- "
            "did you decline the save prompt?"
        )


def _cli():
    p = argparse.ArgumentParser(
        description="Overlay Gray-coded frame-number sync tags onto a video.",
    )
    p.add_argument("video_in", help="path to input video")
    p.add_argument("video_out", help="path to output video")

    src = p.add_argument_group("parameter source (one of these is required)")
    src.add_argument(
        "--calibration-file", dest="calibration_file", default=None,
        help="path to an existing Monitorio calibration JSON",
    )
    src.add_argument(
        "--calibrate", action="store_true",
        help="run calibrate.py first and use its JSON output (mutually "
             "exclusive with --calibration-file). Calibration runs "
             "interactively -- you'll see the instruction screens and "
             "plots and be asked to confirm before tagging begins.",
    )
    src.add_argument(
        "--bit-xs", dest="bit_xs", type=_csv_ints, default=None,
        help="comma-separated list of X pixel positions",
    )
    src.add_argument(
        "--bit-ys", dest="bit_ys", type=_csv_ints, default=None,
        help="comma-separated list of Y pixel positions (single value broadcasts)",
    )
    src.add_argument("--bit-radius", dest="bit_radius", type=int, default=None)
    src.add_argument(
        "--background-radius", dest="background_radius", type=int, default=None,
    )

    cal_fwd = p.add_argument_group("forwarded to calibrate.py (used with --calibrate)")
    cal_fwd.add_argument("--display", type=int, default=None)
    cal_fwd.add_argument("--device", default=None, help="NI DAQ device name (e.g. Dev1)")
    cal_fwd.add_argument(
        "--cache", default=None,
        help="path to pipeline cache .npz. Loaded if it exists, written if not.",
    )
    cal_fwd.add_argument(
        "--force", action="store_true",
        help="ignore any existing --cache and re-measure baselines+coarse+fine",
    )
    cal_fwd.add_argument(
        "--terminal-config", type=str.upper, default=None,
        dest="terminal_config",
        choices=("RSE", "DIFF", "NRSE", "PSEUDO_DIFF"),
        help="AI terminal configuration to pass through to calibrate.py "
             "(default: the script's own default, currently RSE)",
    )

    tag = p.add_argument_group("tagging options")
    tag.add_argument(
        "--screen-size", dest="screen_size", type=_csv_ints, default=None,
        help="screen pixel dimensions WxH (e.g. 2400x1600) of the display "
             "the PD board is mounted on. Tag positions and radii are in "
             "screen pixels and get scaled to the output frame so the "
             "result displays correctly when shown full-screen on this "
             "screen. Required unless --calibration-file is given (then "
             "read from the JSON's monitor.width/height).",
    )
    tag.add_argument(
        "--sync-bit", dest="sync_bit", action=argparse.BooleanOptionalAction,
        default=True,
        help="reserve the first PD (PD index 0 in the calibration JSON, "
             "i.e. the lowest-numbered live AI channel) as an always-on "
             "'video active' indicator and Gray-encode the frame number "
             "across the remaining n-1 PDs. Lets the decoder distinguish "
             "'video off' from 'frame at a multiple of 2**n_bits' (which "
             "would otherwise both look all-dark). Cycle length is halved. "
             "Default: enabled. Pass --no-sync-bit to spend all PDs on "
             "frame bits instead.",
    )
    tag.add_argument(
        "--pad-for-unambiguous-end", dest="pad_for_unambiguous_end",
        action=argparse.BooleanOptionalAction, default=True,
        help="only relevant in --no-sync-bit mode. If the input's frame "
             "count is an exact multiple of the cycle length, append one "
             "extra black-content frame to the output so the last frame "
             "isn't on the all-zeros codeword (which would be ambiguous "
             "with 'video off' in the recording). Default: enabled.",
    )
    tag.add_argument(
        "--codec", default="libx264",
        help="ffmpeg encoder name. Default 'libx264' (CPU, always works). "
             "Use 'h264_nvenc' or 'hevc_nvenc' for NVIDIA GPU acceleration "
             "(typically 5-10x faster on supported GPUs); requires both an "
             "NVENC-capable card and an ffmpeg build with NVIDIA SDK support.",
    )
    tag.add_argument(
        "--preset", default=None,
        help="encoder preset string. Default depends on codec: 'veryfast' "
             "for libx264, 'p4' for NVENC. Pass any preset the chosen "
             "encoder accepts.",
    )
    tag.add_argument(
        "--quality", type=int, default=18,
        help="visual quality knob (lower = higher quality). Mapped to "
             "-crf for libx264, -cq for NVENC. Default 18 is perceptually "
             "lossless on both.",
    )
    tag.add_argument(
        "--crf", type=int, default=None,
        help=argparse.SUPPRESS,  # legacy alias for --quality
    )
    tag.add_argument("--progress", action="store_true", help="print per-frame progress")
    args = p.parse_args()

    if args.calibrate and args.calibration_file:
        p.error("--calibrate and --calibration-file are mutually exclusive")

    quality = args.crf if args.crf is not None else args.quality

    def _tag(calibration_file: str | None) -> None:
        n = add_video_sync_tags(
            video_in=args.video_in, video_out=args.video_out,
            calibration_file=calibration_file,
            bit_xs=args.bit_xs, bit_ys=args.bit_ys,
            bit_radius=args.bit_radius, background_radius=args.background_radius,
            screen_size=tuple(args.screen_size) if args.screen_size else None,
            sync_bit=args.sync_bit,
            pad_for_unambiguous_end=args.pad_for_unambiguous_end,
            codec=args.codec, preset=args.preset, quality=quality,
            show_progress=args.progress,
        )
        print(f"wrote {n} frames to {args.video_out}")

    if args.calibrate:
        # Temp dir stays alive until after tagging completes so the JSON
        # is still readable when add_video_sync_tags opens it.
        with tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True,
        ) as tmpdir:
            cal_path = Path(tmpdir) / "calibration.json"
            _run_calibrate_subprocess(
                out_path=cal_path,
                display=args.display, device=args.device,
                cache=args.cache, force=args.force,
                terminal_config=args.terminal_config,
            )
            _tag(str(cal_path))
    else:
        _tag(args.calibration_file)


if __name__ == "__main__":
    _cli()
