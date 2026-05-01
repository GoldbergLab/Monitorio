# Real-world test fixture: test2 segment 1

Captured 2026-05-01 from the Goldberg Lab Monitorio rig:
- 3 photodiodes in sync-bit mode (cycle = 4) on Intan Recording
  Controller's controller-box analog inputs (ANALOG-IN-4 = sync,
  ANALOG-IN-5 = bit 0, ANALOG-IN-6 = bit 1).
- 2400×1600 monitor displaying a 1200×1920 source video at 45 fps,
  tagged with `--leading-guard-frames 5`.

Files:
- `samples.npz` — 14 s slice (about 2 s of pre-pad + 10 s of video +
  2 s of post-pad) of the relevant 3 board ADC channels, in volts,
  float32 to keep the fixture small (half the size of float64 with
  ample precision over the 0–2.45 V Intan aux input range). Includes
  the sample rate (20 kHz) and the within-extract sample indices that
  bracket the raw sync-on segment.
- `tagged_video.mp4` — the actual tagged video the user displayed
  (1200×1920, 45 fps, 451 source frames + 5 leading guard frames =
  456 output frames).
- `tagged_video.mp4.tags.json` — sidecar from the tagger, used by the
  decoder for sync-bit and channel-assignment info. References
  absolute paths from the original test machine that don't exist in
  this checkout, but the decoder only reads parameters out of it
  (sync_bit, n_source_frames, etc.), not the input-video path.
- `calibration.json` — the calibration the rig was tagged against.

Total size: about 2 MB. The full RHD recording it was extracted from
is ~340 MB across 6 files; not committed.

Used by `tests/test_real_world.py`.
