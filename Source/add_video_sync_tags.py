"""Overlay Gray-coded frame-number sync tags onto a video.

Python port of addVideoSyncTags.m. Uses ffmpeg (must be on PATH) for
video I/O so no heavy Python video library is required; numpy does the
per-frame pixel math.

Each output frame has, at every (bitXs[k], bitYs[k]) position:
  - a black filled disk of `background_radius` pixels
  - a white filled disk of `bit_radius` pixels INSIDE the black disk,
    drawn only when bit k of grayEncode(frame_number) is 1.

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

Parameter resolution for each of bit_xs / bit_ys / bit_radius /
background_radius:
    1. Value passed explicitly to add_video_sync_tags() wins.
    2. Otherwise, if calibration_file is given, its value is used.
    3. Otherwise, this function errors -- there's no safe default
       because sensible values depend on the specific Monitorio board +
       monitor combination.

CLI usage:
    python add_video_sync_tags.py IN.mp4 OUT.mp4 --calibration-file cal.json
    python add_video_sync_tags.py IN.mp4 OUT.mp4 \\
        --bit-xs 31,88,145,202 --bit-ys 40 \\
        --bit-radius 20 --background-radius 35
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
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
    enlarged_size: tuple[int, int] | None = None,
    crf: int = 18,
    show_progress: bool = False,
) -> int:
    """Write `video_out` = `video_in` with Gray-coded sync-tag overlays.

    Returns the number of frames written.
    """
    video_in = Path(video_in)
    video_out = Path(video_out)
    if not video_in.exists():
        raise FileNotFoundError(video_in)

    xs, ys, bit_r, bg_r = _resolve_parameters(
        calibration_file, bit_xs, bit_ys, bit_radius, background_radius,
    )
    n_bits = max(len(xs), len(ys))
    if len(xs) == 1:
        xs = np.tile(xs, n_bits)
    if len(ys) == 1:
        ys = np.tile(ys, n_bits)
    if len(xs) != n_bits or len(ys) != n_bits:
        raise ValueError(
            f"bit_xs and bit_ys must be broadcast-compatible: "
            f"got len(bit_xs)={len(xs)}, len(bit_ys)={len(ys)}"
        )
    cycle = 1 << n_bits  # grayEncode maps [0, 2**n) -> [0, 2**n) bijectively

    info = _probe_video(video_in)
    w, h, fps, n_frames = info["width"], info["height"], info["fps"], info["n_frames"]

    if enlarged_size is not None:
        out_w, out_h = int(enlarged_size[0]), int(enlarged_size[1])
        if out_w < w or out_h < h:
            raise ValueError(
                f"enlarged_size ({out_w}x{out_h}) is smaller than input ({w}x{h})"
            )
        pad_left = (out_w - w) // 2
        pad_top = (out_h - h) // 2
    else:
        out_w, out_h = w, h
        pad_left = 0
        pad_top = 0

    # Precompute per-bit disk masks (bounding box + boolean mask); they
    # never change frame to frame, so doing it once saves a meshgrid per
    # frame per bit.
    bg_masks = [
        _disk_mask(int(x) + pad_left, int(y) + pad_top, bg_r, out_w, out_h)
        for x, y in zip(xs, ys)
    ]
    bit_masks = [
        _disk_mask(int(x) + pad_left, int(y) + pad_top, bit_r, out_w, out_h)
        for x, y in zip(xs, ys)
    ]

    if n_frames is not None and n_frames >= cycle:
        # Wrapping is supported by design (with limited PDs you usually have
        # to), but warn so a user who didn't realize they were wrapping isn't
        # surprised when their decoder sees repeated codes.
        n_cycles = n_frames / cycle
        print(
            f"  note: {n_frames} frames > 2**{n_bits} = {cycle}; "
            f"frame number will wrap {n_cycles:.1f} times. Decoder must "
            f"disambiguate cycles via timestamp.",
            file=sys.stderr,
        )

    # ffmpeg read: decode video-only rgb24 stream to stdout.
    read_cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(video_in),
        "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    # ffmpeg write: take raw rgb24 from stdin (video), re-open input for
    # audio if present, mux into H.264 + original audio.
    write_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "-",
        "-i", str(video_in),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(int(crf)),
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
    # Reusable output buffer when enlarging; otherwise we modify a fresh
    # copy of each input frame.
    out_frame = (
        np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if enlarged_size is not None
        else None
    )

    frames_written = 0
    try:
        while True:
            raw = read_proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

            if enlarged_size is not None:
                out_frame[:] = 0
                out_frame[pad_top:pad_top + h, pad_left:pad_left + w] = frame
                draw_target = out_frame
            else:
                # np.frombuffer gives a read-only view; copy once so we
                # can draw into it.
                draw_target = frame.copy()

            frames_written += 1
            # Wrap modulo cycle so videos longer than 2**n_bits frames keep
            # encoding cleanly; gray() over a full [0, 2**n_bits) cycle has
            # the single-bit-change property at every step including the
            # wrap edge.
            g = int(gray.encode(np.int64(frames_written % cycle)))

            # Always-on black backgrounds.
            for m in bg_masks:
                _apply_mask(draw_target, m, 0)
            # White-on-black for every bit of g that's set.
            for k in range(n_bits):
                if (g >> k) & 1:
                    _apply_mask(draw_target, bit_masks[k], 255)

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
    return frames_written


# ----- internals ------------------------------------------------------

def _resolve_parameters(calibration_file, bit_xs, bit_ys, bit_radius, background_radius):
    cal_xs = cal_ys = cal_br = cal_bgr = None
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

    xs = np.asarray(bit_xs, dtype=np.int64) if bit_xs is not None else cal_xs
    ys = np.asarray(bit_ys, dtype=np.int64) if bit_ys is not None else cal_ys
    br = int(bit_radius) if bit_radius is not None else cal_br
    bgr = int(background_radius) if background_radius is not None else cal_bgr

    missing = []
    if xs is None: missing.append("bit_xs")
    if ys is None: missing.append("bit_ys")
    if br is None: missing.append("bit_radius")
    if bgr is None: missing.append("background_radius")
    if missing:
        raise ValueError(
            f"Missing required value(s): {', '.join(missing)}. "
            f"Pass them explicitly, or supply calibration_file."
        )
    return xs, ys, br, bgr


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
        "--enlarged-size", dest="enlarged_size", type=_csv_ints, default=None,
        help="output dimensions WxH (e.g. 1920x1080) if input should be padded",
    )
    tag.add_argument(
        "--crf", type=int, default=18,
        help="libx264 CRF (lower = higher quality, larger file). Default 18.",
    )
    tag.add_argument("--progress", action="store_true", help="print per-frame progress")
    args = p.parse_args()

    if args.calibrate and args.calibration_file:
        p.error("--calibrate and --calibration-file are mutually exclusive")

    def _tag(calibration_file: str | None) -> None:
        n = add_video_sync_tags(
            video_in=args.video_in, video_out=args.video_out,
            calibration_file=calibration_file,
            bit_xs=args.bit_xs, bit_ys=args.bit_ys,
            bit_radius=args.bit_radius, background_radius=args.background_radius,
            enlarged_size=tuple(args.enlarged_size) if args.enlarged_size else None,
            crf=args.crf, show_progress=args.progress,
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
