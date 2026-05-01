"""Play randomly-chosen tagged videos with exponential inter-video gaps.

Reads a TOML config naming the videos and the timing parameters, opens
a fullscreen pygame window on the chosen monitor (black throughout the
session except while a video is playing), and loops:

    1. Sample an inter-video interval (IVI) from
       Exp(1/mean_ivi_s), truncated to [min_ivi_s, max_ivi_s] via
       rejection sampling.
    2. Sleep that long, with the screen kept black.
    3. Pick a random video from the configured list.
    4. Play it: decode raw RGB frames with ffmpeg, blit each frame
       into the fullscreen pygame window at the right time.
    5. Append a row to the playback log with wall-clock timestamps.
    6. Repeat until n_plays plays have happened or
       total_session_seconds have elapsed (whichever the config sets).

The log + the continuous DAQ recording together let the analyst chunk
the recording into per-playback windows for the decoder. Wall-clock
timestamps in the log are approximate (subject to OS scheduling and
player startup latency); the decoder establishes exact frame timing
from the PD signals.

Press ESC at any time during playback to abort. Pygame events are
drained during the IVI sleeps so ESC during a gap also aborts.

Why pygame for the window: it gives us native fullscreen on a chosen
display index (already used by the calibration tool), no title bar /
no chrome, no window-open/close animations between plays (the window
persists; only its contents change), and no extra dependencies. Video
frames are decoded by a short-lived ffmpeg subprocess pipe (same
mechanism add_video_sync_tags uses) and blitted into the window via
pygame.surfarray.

Config (TOML):

    videos = ["C:/path/to/v1.mp4", "C:/path/to/v2.mp4"]   # required, >=1

    [timing]
    mean_ivi_seconds = 30.0           # exponential distribution mean
    min_ivi_seconds = 5.0             # rejection-sample lower bound
    max_ivi_seconds = 120.0           # rejection-sample upper bound
    # Pick exactly one termination condition:
    n_plays = 50                      # ...or omit and use:
    # total_session_seconds = 1800

    [display]
    monitor_index = 1                 # 0 = primary; pygame's display list

    [output]
    log_path = "playback_log.csv"     # appended to; created if missing

    [random]
    seed = 42                         # optional; deterministic when set

CLI:
    venv/Scripts/python Source/playback/play_random.py CONFIG.toml
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pygame


# ----- config loading -------------------------------------------------

def _load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _resolve_paths(config_path: Path, cfg: dict) -> tuple[list[Path], Path]:
    """Resolve relative paths in cfg to absolute, anchored at config_dir.

    Lets users write configs with paths relative to the config file
    rather than the CWD they happen to run from.
    """
    config_dir = config_path.parent.resolve()

    raw_videos = cfg.get("videos") or []
    if not raw_videos:
        raise ValueError(f"{config_path}: 'videos' must list at least one file")
    videos = []
    for v in raw_videos:
        p = Path(v)
        if not p.is_absolute():
            p = config_dir / p
        if not p.exists():
            raise FileNotFoundError(p)
        videos.append(p.resolve())

    log_path_s = cfg.get("output", {}).get("log_path", "playback_log.csv")
    log_path = Path(log_path_s)
    if not log_path.is_absolute():
        log_path = config_dir / log_path
    return videos, log_path


# ----- IVI sampling ---------------------------------------------------

def _sample_ivi(
    rng: random.Random, mean_s: float, lo_s: float, hi_s: float,
    max_attempts: int = 1000,
) -> float:
    """Truncated exponential via rejection sampling.

    Returns a draw from Exp(1/mean_s) that lies in [lo_s, hi_s]. Falls
    back to clipping after max_attempts (unreachable for sane parameter
    choices; sane = the truncation interval contains a non-trivial
    chunk of the distribution's mass).
    """
    if not (0 <= lo_s < hi_s):
        raise ValueError(f"need 0 <= min_ivi_seconds < max_ivi_seconds; got {lo_s}, {hi_s}")
    if mean_s <= 0:
        raise ValueError(f"mean_ivi_seconds must be positive; got {mean_s}")
    for _ in range(max_attempts):
        x = rng.expovariate(1.0 / mean_s)
        if lo_s <= x <= hi_s:
            return x
    # Pathological config: the [lo, hi] interval misses essentially all
    # the distribution mass. Clip and warn.
    print(
        f"  warning: rejection sampling for IVI failed after "
        f"{max_attempts} draws (mean={mean_s}, range=[{lo_s}, {hi_s}]); "
        f"clipping a draw. Consider widening the truncation bounds.",
        file=sys.stderr,
    )
    return min(hi_s, max(lo_s, rng.expovariate(1.0 / mean_s)))


# ----- video probing --------------------------------------------------

def _probe_video(path: Path) -> dict:
    """ffprobe a video file, return {width, height, fps, n_frames}."""
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
    return {"width": w, "height": h, "fps": fps, "n_frames": n_frames}


# ----- playback (one video) -------------------------------------------

def _play_one(
    video_path: Path, screen, screen_w: int, screen_h: int,
    *, info: dict | None = None,
) -> dict:
    """Decode `video_path` and blit its frames into `screen`.

    Returns a dict with playback diagnostics:
      aborted: True iff user pressed ESC / closed the window.
      duration: actual wall-clock duration of playback (seconds).
      frames_shown: how many frames were blitted.
      expected_frames: ffprobed frame count (None if ffprobe couldn't
                       determine).
      expected_duration: expected_frames / fps when known, else None.
      max_late_ms: worst slip past a frame's nominal display time.
      n_late_frames: number of frames whose blit was >= 5 ms past
                     their target time (a rough "the playback fell
                     behind here" indicator).
      ffmpeg_returncode: exit status of the decoding ffmpeg process.
      ffmpeg_stderr: any stderr output from ffmpeg (truncated).

    Frames are scheduled at their nominal interval (1 / fps) measured
    against a single monotonic anchor at playback start, so any
    per-frame jitter in the Python loop doesn't accumulate over a long
    video. If a frame falls behind (rendering can't keep up), we don't
    sleep negative -- we move on and account for it in max_late_ms /
    n_late_frames.
    """
    if info is None:
        info = _probe_video(video_path)
    vw, vh, fps = info["width"], info["height"], info["fps"]
    expected_frames = info.get("n_frames")

    if vw > screen_w or vh > screen_h:
        raise ValueError(
            f"video {video_path} is {vw}x{vh} but screen is "
            f"{screen_w}x{screen_h}. Resize the video first; this "
            f"playback tool only centers, never scales."
        )
    px = (screen_w - vw) // 2
    py = (screen_h - vh) // 2

    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = vw * vh * 3
    frame_period = 1.0 / fps
    surf = pygame.Surface((vw, vh))
    aborted = False
    frames_shown = 0
    max_late = 0.0
    n_late = 0
    t_start = time.perf_counter()
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(vh, vw, 3)
            # pygame.surfarray uses (W, H, 3) axis order.
            pygame.surfarray.blit_array(surf, frame.transpose(1, 0, 2))
            screen.fill((0, 0, 0))
            screen.blit(surf, (px, py))

            target = t_start + frames_shown * frame_period
            now = time.perf_counter()
            slip = now - target
            if slip > 0:
                # Behind schedule. Track but don't sleep negative.
                if slip > max_late:
                    max_late = slip
                if slip > 0.005:
                    n_late += 1
            else:
                time.sleep(-slip)  # i.e. sleep(target - now)
            pygame.display.flip()

            # Drain events; ESC or window-close aborts.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    aborted = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    aborted = True
                    break
            if aborted:
                break

            frames_shown += 1
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    duration = time.perf_counter() - t_start
    rc = proc.returncode
    stderr_bytes = b""
    if proc.stderr is not None:
        try:
            stderr_bytes = proc.stderr.read() or b""
        except (OSError, ValueError):
            pass
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    if len(stderr_text) > 500:
        stderr_text = stderr_text[:500] + "...[truncated]"

    return {
        "aborted": aborted,
        "duration": duration,
        "frames_shown": frames_shown,
        "expected_frames": expected_frames,
        "expected_duration": (expected_frames / fps) if expected_frames else None,
        "max_late_ms": max_late * 1000.0,
        "n_late_frames": n_late,
        "ffmpeg_returncode": rc,
        "ffmpeg_stderr": stderr_text,
    }


# ----- session driver -------------------------------------------------

def _wait_with_events(seconds: float, *, screen) -> bool:
    """Sleep for `seconds` while pumping pygame events. Returns True on
    user abort (ESC or window-close)."""
    end = time.perf_counter() + seconds
    while True:
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        # Sleep in small chunks so events stay responsive.
        time.sleep(min(0.05, remaining))


def _say(msg: str) -> None:
    """Status line to the operator's console. flush so it appears immediately."""
    print(msg, file=sys.stderr, flush=True)


def run_session(config_path: Path) -> int:
    cfg = _load_config(config_path)
    videos, log_path = _resolve_paths(config_path, cfg)

    timing = cfg.get("timing", {})
    mean_ivi = float(timing.get("mean_ivi_seconds", 30.0))
    min_ivi = float(timing.get("min_ivi_seconds", 5.0))
    max_ivi = float(timing.get("max_ivi_seconds", 120.0))
    n_plays = timing.get("n_plays")
    total_seconds = timing.get("total_session_seconds")
    if n_plays is None and total_seconds is None:
        raise ValueError(
            f"{config_path}: must set either timing.n_plays or "
            f"timing.total_session_seconds"
        )
    if n_plays is not None and total_seconds is not None:
        raise ValueError(
            f"{config_path}: set only one of timing.n_plays or "
            f"timing.total_session_seconds, not both"
        )

    monitor_idx = int(cfg.get("display", {}).get("monitor_index", 0))
    seed = cfg.get("random", {}).get("seed")
    rng = random.Random(seed) if seed is not None else random.Random()

    # Probe each video once up front so we (a) fail fast on corrupt
    # files and (b) have the metadata cached for status messages.
    video_info: dict[Path, dict] = {}
    for v in videos:
        try:
            video_info[v] = _probe_video(v)
        except RuntimeError as e:
            raise RuntimeError(f"failed to probe video {v}: {e}") from e

    # Open the persistent fullscreen window. Stays open for the whole
    # session; we just blit into it as needed.
    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if not (0 <= monitor_idx < len(sizes)):
        pygame.display.quit()
        raise ValueError(
            f"display.monitor_index={monitor_idx} but only {len(sizes)} "
            f"monitor(s) attached: {sizes}"
        )
    sw, sh = sizes[monitor_idx]
    screen = pygame.display.set_mode(
        (sw, sh), flags=pygame.FULLSCREEN, display=monitor_idx,
    )
    pygame.mouse.set_visible(False)
    pygame.display.set_caption("Monitorio random playback")

    def black():
        screen.fill((0, 0, 0))
        pygame.display.flip()

    black()

    # --- session-start banner ----------------------------------------
    _say("=" * 64)
    _say(f"Monitorio random playback session")
    _say(f"  config:      {config_path}")
    _say(f"  videos:      {len(videos)}")
    for v in videos:
        info = video_info[v]
        nf = info["n_frames"]
        dur = nf / info["fps"] if nf else float("nan")
        _say(
            f"    {v.name}  ({info['width']}x{info['height']}, "
            f"{info['fps']:.2f} fps, {nf or '?'} frames, "
            f"{dur:.2f} s)"
        )
    _say(f"  monitor:     #{monitor_idx} ({sw}x{sh})")
    _say(
        f"  IVI:         Exp(mean={mean_ivi:.1f} s) truncated to "
        f"[{min_ivi:.1f}, {max_ivi:.1f}] s"
    )
    if n_plays is not None:
        _say(f"  termination: n_plays = {n_plays}")
    else:
        _say(f"  termination: total_session_seconds = {total_seconds:.0f}")
    _say(f"  log:         {log_path}")
    _say(f"  seed:        {seed if seed is not None else '(random)'}")
    _say(f"  press ESC during playback or IVI to abort cleanly.")
    _say("=" * 64)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", encoding="utf-8") as log:
        if new_log:
            log.write(
                "play_index,start_time_iso,start_time_unix,"
                "video_path,duration_seconds,frames_shown,"
                "expected_frames,ivi_seconds,aborted,n_late_frames,"
                "max_late_ms,ffmpeg_returncode\n"
            )
            log.flush()

        session_start = time.perf_counter()
        plays_done = 0
        aborted = False
        n_failures = 0
        try:
            while not aborted:
                # Termination conditions checked at top of loop.
                if n_plays is not None and plays_done >= n_plays:
                    break
                elapsed = time.perf_counter() - session_start
                if total_seconds is not None and elapsed >= total_seconds:
                    break

                # Pick the next video and IVI now so we can announce
                # both during the gap (operator can see what's coming).
                video = rng.choice(videos)
                info = video_info[video]
                ivi = _sample_ivi(rng, mean_ivi, min_ivi, max_ivi)

                if n_plays is not None:
                    progress = f"{plays_done + 1}/{n_plays}"
                else:
                    pct = 100.0 * elapsed / total_seconds
                    progress = f"#{plays_done + 1}, {pct:.1f}% elapsed"
                _say(
                    f"[{progress}] next: {video.name} "
                    f"({info['n_frames'] or '?'} frames, "
                    f"{(info['n_frames'] or 0) / info['fps']:.2f} s) "
                    f"in {ivi:.1f} s"
                )

                aborted = _wait_with_events(ivi, screen=screen)
                if aborted:
                    _say("[abort] user pressed ESC during IVI")
                    break

                t0_unix = time.time()
                t0_iso = datetime.datetime.fromtimestamp(
                    t0_unix, tz=datetime.timezone.utc,
                ).isoformat(timespec="microseconds")
                _say(f"[{progress}] starting playback at {t0_iso}")

                play_failed = False
                play_failure_msg = ""
                try:
                    res = _play_one(
                        video, screen, sw, sh, info=info,
                    )
                except Exception as e:
                    play_failed = True
                    play_failure_msg = repr(e)
                    res = {
                        "aborted": False, "duration": 0.0, "frames_shown": 0,
                        "expected_frames": info.get("n_frames"),
                        "expected_duration": None,
                        "max_late_ms": 0.0, "n_late_frames": 0,
                        "ffmpeg_returncode": -1,
                        "ffmpeg_stderr": "",
                    }
                black()

                # Failure detection + status. Things we flag:
                # - Python-level exception during playback
                # - ffmpeg exited with a non-zero code
                # - Frame count differs significantly from probed count
                # - Significant slip (lots of late frames)
                problems = []
                if play_failed:
                    problems.append(f"exception: {play_failure_msg}")
                rc = res["ffmpeg_returncode"]
                if rc not in (0, None):
                    problems.append(
                        f"ffmpeg exit {rc}: {res['ffmpeg_stderr'][:200]}"
                    )
                exp_frames = res["expected_frames"]
                if exp_frames is not None and not res["aborted"]:
                    delta = res["frames_shown"] - exp_frames
                    if abs(delta) > max(2, int(0.01 * exp_frames)):
                        problems.append(
                            f"frame count {res['frames_shown']} vs probed "
                            f"{exp_frames} (off by {delta:+d})"
                        )
                if res["n_late_frames"] > max(5, 0.05 * res["frames_shown"]):
                    problems.append(
                        f"{res['n_late_frames']} of {res['frames_shown']} "
                        f"frames slipped past their target time "
                        f"(max {res['max_late_ms']:.1f} ms late)"
                    )

                # Per-play summary line.
                tag = "ABORTED" if res["aborted"] else (
                    "FAILED" if problems else "ok"
                )
                _say(
                    f"[{progress}] {tag}: {res['frames_shown']} frames "
                    f"in {res['duration']:.3f} s "
                    f"(target {(res['expected_duration'] or 0):.3f} s, "
                    f"max slip {res['max_late_ms']:.1f} ms)"
                )
                for p in problems:
                    _say(f"  ! {p}")
                if problems:
                    n_failures += 1

                # Log row -- always written, even on failure, so the
                # CSV is a faithful record of what happened.
                log.write(
                    f"{plays_done + 1},{t0_iso},{t0_unix:.6f},"
                    f"{video},{res['duration']:.6f},{res['frames_shown']},"
                    f"{exp_frames if exp_frames is not None else ''},"
                    f"{ivi:.4f},{str(res['aborted']).lower()},"
                    f"{res['n_late_frames']},{res['max_late_ms']:.3f},"
                    f"{rc if rc is not None else ''}\n"
                )
                log.flush()
                plays_done += 1

                if res["aborted"]:
                    aborted = True
                    _say("[abort] user pressed ESC during playback")
        finally:
            pygame.mouse.set_visible(True)
            pygame.display.quit()

    # --- session-end summary -----------------------------------------
    elapsed = time.perf_counter() - session_start
    _say("=" * 64)
    _say(
        f"Session ended: {plays_done} play(s) over {elapsed:.1f} s"
        f"{'  (aborted)' if aborted else ''}"
        f"{f'  ({n_failures} flagged)' if n_failures else ''}"
    )
    if plays_done:
        _say(f"  log: {log_path}")
    _say("=" * 64)
    return 1 if n_failures else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config", type=Path, help="path to TOML config file")
    args = p.parse_args(argv)
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    return run_session(args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
