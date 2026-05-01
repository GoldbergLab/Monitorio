"""Play randomly-chosen tagged videos with exponential inter-video gaps.

Reads a TOML config naming the videos and the timing parameters, opens
a fullscreen pygame window on the chosen monitor (black throughout the
session except while a video is playing), and loops:

    1. Sample an inter-video interval (IVI) from
       Exp(1/mean_ivi_s), truncated to [min_ivi_s, max_ivi_s] via
       rejection sampling.
    2. Sleep that long, with the screen kept black.
    3. Pick a random video from the configured list.
    4. Hand it to VLC to play (audio + video, frame-accurate sync).
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

Architecture: pygame opens a persistent fullscreen black window on the
chosen monitor. VLC is told to render into pygame's HWND
(set_hwnd / set_xwindow), so the same window does double duty as
"black background between plays" (pygame) and "playback surface"
(VLC). One window for the whole session means no Windows window-
open/close animations, no chrome to disable per-player. VLC handles
audio + video sync natively.

Requires: VLC installed system-wide, plus `pip install python-vlc`.

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

import pygame
try:
    import vlc
except OSError as _vlc_err:
    # python-vlc loads libvlc.dll via ctypes at import time. Two common
    # ways this fails on a fresh box:
    #   - WinError 193 / "%1 is not a valid Win32 application": Python's
    #     and VLC's bitness disagree (most often 64-bit Python paired
    #     with a 32-bit VLC install, or vice versa). Install matching-
    #     bitness VLC; check Python bitness via
    #     python -c "import struct; print(struct.calcsize('P')*8)".
    #   - "Could not find module 'libvlc.dll'": VLC isn't installed at
    #     all, or its install dir isn't on PATH. Install from
    #     https://www.videolan.org/ or set VLC_PLUGIN_PATH.
    _winerr = getattr(_vlc_err, "winerror", None)
    if _winerr == 193:
        raise OSError(
            "Failed to load libvlc.dll: bitness mismatch between Python "
            "and the installed VLC. Run "
            "  python -c \"import struct; print(struct.calcsize('P')*8)\"  "
            "to see Python's bitness, then install matching-bitness VLC "
            "from https://www.videolan.org/ (64-bit VLC for 64-bit "
            "Python, etc.). If both bitnesses are installed, put the "
            "matching VLC dir on PATH ahead of the other."
        ) from _vlc_err
    raise OSError(
        "Failed to import the VLC bindings (python-vlc). Make sure VLC "
        "media player itself is installed system-wide -- python-vlc is "
        "just the bindings; it loads VLC's libvlc.dll at import. "
        "Install VLC from https://www.videolan.org/ if you haven't "
        f"already. Original error: {_vlc_err!r}"
    ) from _vlc_err


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
    video_path: Path, *, vlc_player, vlc_instance, screen,
    info: dict | None = None,
) -> dict:
    """Hand `video_path` to VLC, wait for it to finish, return diagnostics.

    `vlc_player` is a pre-configured vlc.MediaPlayer that's already had
    set_hwnd() (or set_xwindow / set_nsobject) called on the pygame
    fullscreen window. We just hand it new media, call play(), and
    poll for the End/Stopped/Error state.

    Returns a dict mirroring the old (pygame-blit-based) implementation
    so the session-driver status and CSV-log code stays unchanged:
      aborted, duration, frames_shown, expected_frames,
      expected_duration, max_late_ms, n_late_frames, vlc_state,
      vlc_error.

    Note that frames_shown / max_late_ms / n_late_frames don't have the
    same meaning under VLC as they did under pygame's per-frame loop --
    VLC handles vsync internally and we don't see per-frame timing.
    Instead:
      - frames_shown: estimated as round(duration * fps) (VLC doesn't
        expose a frame counter on the python-vlc API surface). This is
        only useful for the CSV log; the decoder's PD-signal-driven
        frame count is the source of truth.
      - max_late_ms / n_late_frames: always 0; left in the dict so the
        CSV columns keep the same meaning across engines (and so a mix
        of pygame-engine and vlc-engine sessions is still parseable).
    """
    if info is None:
        info = _probe_video(video_path)
    fps = info["fps"]
    expected_frames = info.get("n_frames")
    vw, vh = info["width"], info["height"]
    sw, sh = screen.get_size()
    if vw != sw or vh != sh:
        # VLC will scale to fit the window. That's fine for casual
        # testing and produces a watchable playback, but the tagger
        # places the sync circles at SCREEN pixel coordinates: any
        # scaling moves them off the calibrated photodiode positions
        # so a recording with this playback won't decode correctly.
        # Warn (don't raise) so an operator running play_random as a
        # smoke test on a different display can still see videos
        # play.
        print(
            f"  warning: video {video_path.name} is {vw}x{vh} but the "
            f"playback window is {sw}x{sh}; VLC will scale. The tagged "
            f"sync circles won't land at their calibrated screen-pixel "
            f"positions, so a Monitorio recording made with this "
            f"playback won't decode correctly. Resize the video to "
            f"match the screen for actual experiments.",
            file=sys.stderr,
        )

    # VLC parses media internally when play() is called and surfaces
    # parse / decode failures via the Error state in the wait loop
    # below, so we don't need an explicit pre-play parse step. (The
    # python-vlc parse-status enum doesn't expose a stable name for
    # the "in progress" value across versions, which makes any
    # parse-and-poll dance brittle.)
    media = vlc_instance.media_new(str(video_path))
    vlc_player.set_media(media)

    aborted = False
    err_msg = ""
    t_start = time.perf_counter()

    play_rc = vlc_player.play()
    if play_rc != 0:
        err_msg = f"vlc_player.play() returned {play_rc}"
        return {
            "aborted": False, "duration": 0.0, "frames_shown": 0,
            "expected_frames": expected_frames,
            "expected_duration": (expected_frames / fps) if expected_frames else None,
            "max_late_ms": 0.0, "n_late_frames": 0,
            "vlc_state": "play_failed",
            "vlc_error": err_msg,
        }

    # Wait for VLC to leave the Opening / Buffering states and start
    # actually playing, then for it to reach Ended (or Error / Stopped).
    last_state = None
    poll = 0.020  # 20 ms poll cadence
    grace_until = time.perf_counter() + 5.0  # seconds for VLC to start
    while True:
        state = vlc_player.get_state()
        if state == vlc.State.Ended:
            break
        if state == vlc.State.Stopped:
            # Either we (or someone else) stopped it, or it never started.
            break
        if state == vlc.State.Error:
            err_msg = f"vlc reached Error state at t={time.perf_counter() - t_start:.2f}s"
            break

        # Drain pygame events so ESC / window-close still abort.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                aborted = True
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                aborted = True
                break
        if aborted:
            vlc_player.stop()
            break

        # If VLC never actually starts, give up.
        if state in (vlc.State.NothingSpecial, vlc.State.Opening, vlc.State.Buffering):
            if time.perf_counter() > grace_until:
                err_msg = f"vlc stuck in {state} after 5 s"
                vlc_player.stop()
                break
        last_state = state
        time.sleep(poll)

    duration = time.perf_counter() - t_start
    final_state = vlc_player.get_state()
    if not err_msg and final_state == vlc.State.Error:
        err_msg = "vlc final state Error"

    # After the media ends, stop the player and have pygame redraw black
    # so we go back to a black window for the IVI gap.
    vlc_player.stop()
    screen.fill((0, 0, 0))
    pygame.display.flip()

    return {
        "aborted": aborted,
        "duration": duration,
        "frames_shown": int(round(duration * fps)) if duration > 0 else 0,
        "expected_frames": expected_frames,
        "expected_duration": (expected_frames / fps) if expected_frames else None,
        "max_late_ms": 0.0,
        "n_late_frames": 0,
        "vlc_state": str(final_state).rsplit(".", 1)[-1],
        "vlc_error": err_msg,
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

    # Open the persistent fullscreen pygame window. Stays open for the
    # whole session; serves as the persistent "black between plays"
    # surface AND as VLC's render target during plays (we hand its
    # HWND to VLC below).
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

    # Set up VLC and embed its video output into the pygame window.
    vlc_instance = vlc.Instance(
        "--no-video-title-show",     # no overlay with the file name
        "--no-osd",                  # no on-screen display
        "--no-stats",
        "--no-snapshot-preview",
        "--no-keyboard-events",      # we handle ESC via pygame
        "--no-mouse-events",
        "--quiet",
    )
    vlc_player = vlc_instance.media_player_new()
    wm = pygame.display.get_wm_info()
    if sys.platform == "win32":
        # On Windows pygame's get_wm_info()['window'] is the SDL window
        # handle, which is the HWND we want.
        vlc_player.set_hwnd(int(wm["window"]))
    elif sys.platform.startswith("linux"):
        vlc_player.set_xwindow(int(wm["window"]))
    elif sys.platform == "darwin":
        vlc_player.set_nsobject(int(wm["window"]))
    else:
        pygame.display.quit()
        raise RuntimeError(
            f"unsupported platform {sys.platform!r}: "
            f"don't know how to embed VLC video output"
        )

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
                "expected_frames,ivi_seconds,aborted,"
                "vlc_state,vlc_error\n"
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
                        video, vlc_player=vlc_player,
                        vlc_instance=vlc_instance, screen=screen,
                        info=info,
                    )
                except Exception as e:
                    play_failed = True
                    play_failure_msg = repr(e)
                    res = {
                        "aborted": False, "duration": 0.0, "frames_shown": 0,
                        "expected_frames": info.get("n_frames"),
                        "expected_duration": None,
                        "max_late_ms": 0.0, "n_late_frames": 0,
                        "vlc_state": "exception",
                        "vlc_error": play_failure_msg,
                    }
                black()

                # Failure detection + status. Things we flag:
                # - Python-level exception during playback
                # - VLC reached an Error state instead of Ended
                # - Duration differs significantly from probed expectation
                problems = []
                if play_failed:
                    problems.append(f"exception: {play_failure_msg}")
                if res["vlc_error"]:
                    problems.append(f"vlc: {res['vlc_error']}")
                exp_dur = res["expected_duration"]
                if exp_dur is not None and not res["aborted"]:
                    actual = res["duration"]
                    if abs(actual - exp_dur) > max(0.2, 0.05 * exp_dur):
                        problems.append(
                            f"playback duration {actual:.3f}s differs from "
                            f"probed {exp_dur:.3f}s by more than 5% / 200ms; "
                            f"VLC may have failed to play through cleanly"
                        )

                # Per-play summary line.
                tag = "ABORTED" if res["aborted"] else (
                    "FAILED" if problems else "ok"
                )
                _say(
                    f"[{progress}] {tag}: {res['duration']:.3f} s elapsed "
                    f"(target {(exp_dur or 0):.3f} s, vlc state: "
                    f"{res['vlc_state']})"
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
                    f"{res['expected_frames'] if res['expected_frames'] is not None else ''},"
                    f"{ivi:.4f},{str(res['aborted']).lower()},"
                    f"{res['vlc_state']},"
                    f"{json.dumps(res['vlc_error']) if res['vlc_error'] else ''}\n"
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
    example_cfg = Path(__file__).resolve().parent / "example_config.toml"
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            "Example config: " + str(example_cfg) + "\n"
            "  copy and edit it for your rig, then run:\n"
            "    python Source/playback/play_random.py path/to/myconfig.toml\n"
            "Press ESC during a play or an IVI to abort cleanly.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "config", type=Path, nargs="?",
        help="path to TOML config file (see example linked below)",
    )
    args = p.parse_args(argv)
    if args.config is None:
        # No-arg invocation: print help instead of argparse's terse
        # "the following arguments are required" error.
        p.print_help()
        return 0
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    return run_session(args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
