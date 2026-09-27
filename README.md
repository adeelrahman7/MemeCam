# MemeCam

A desktop app that watches your webcam, recognizes hand gestures with MediaPipe, and pops a
meme reaction into the corner of the live video.

**Status:** Phase 1 (MVP) plus custom gesture recording: record your own hand poses in the
app and map them to memes.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a webcam. uv installs Python 3.12 for you if
needed.

```bash
cd memecam
uv sync                       # creates .venv with runtime + dev dependencies
```

### Download the model

MediaPipe's `.task` model files aren't on PyPI; download them into `models/`:

| File | Needed for | URL |
| --- | --- | --- |
| `gesture_recognizer.task` | **Phase 1** | https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task |
| `face_landmarker.task` | Phase 2 | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |
| `hand_landmarker.task` | Phase 4 (optional) | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task |

macOS / Linux:

```bash
curl -L -o models/gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
```

Windows (PowerShell):

```powershell
Invoke-WebRequest -OutFile models/gesture_recognizer.task `
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
```

`models/*.task` is git-ignored.

## Run

```bash
uv run memecam            # or: uv run python -m memecam
uv run memecam --debug    # start with the debug overlay on
uv run memecam --config path/to/other.json
```

Press **D** (or tick *Debug overlay*) to toggle hand skeletons, `Left/Right: Gesture score`
labels, FPS, inference time, and the debouncer's hold progress / cooldown state.

Hold a gesture steady for `hold_frames` frames (about 0.2 s at 30 fps with the default of 6)
to fire its reaction.

On macOS, the first run triggers a camera permission prompt for your terminal app. Allow it,
then restart.

## Configuration: `config/reactions.json`

```json
{
  "camera":    { "index": 0, "width": 1280, "height": 720, "mirror": true, "swap_handedness": false },
  "inference": { "model": "models/gesture_recognizer.task", "max_width": 640,
                 "num_hands": 2, "min_detection_confidence": 0.5, "min_gesture_score": 0.6 },
  "debounce":  { "hold_frames": 6, "cooldown_seconds": 1.5 },
  "overlay":   { "corner": "top_right", "width_fraction": 0.3, "margin_px": 16, "display_seconds": 2.0,
                 "stay_while_held": true, "linger_seconds": 0.5 },
  "custom_gestures": { "file": "config/custom_gestures.json", "threshold": 0.2,
                       "rotation_invariant": false, "countdown_seconds": 3.0, "capture_frames": 45 },
  "reactions": [
    { "gesture": "Thumb_Up", "image": "assets/memes/thumbs_up.png" },
    { "gesture": "Victory",  "image": "assets/memes/victory_left.png", "hand": "left" }
  ]
}
```

- Every section except `reactions` is optional and falls back to the defaults above.
- `overlay.stay_while_held`: the meme stays up while you hold the gesture, then disappears
  `linger_seconds` after you let go. Set it to `false` for a quick flash of `display_seconds`.
- `gesture` is either one of MediaPipe's built-in labels (`Closed_Fist`, `Open_Palm`,
  `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`) or the name of a gesture
  you recorded (see below). Built-in names are Capitalized, and custom names are lowercase,
  so they can't clash.
- `hand` is `any` (default), `left` or `right`, meaning your real hand. Per gesture, use
  either one `any` reaction or separate `left`/`right` ones.
- `image` paths are relative to the project root (or absolute). Phase 1 supports `.png`;
  transparency is respected.
- Unknown keys, bad values, missing images and duplicate reactions are rejected at startup
  with a message listing every problem, for example:

  ```text
  reactions.json has 2 problem(s):
    • debounce.hold_frames: Input should be greater than or equal to 1
    • reactions[0].gesture: unknown gesture 'Thumbs_Up'; expected one of [...]
  ```

The bundled PNGs in `assets/memes/` are simple placeholders. Swap in real memes whenever
you like.

## Custom gestures (record your own)

1. Click **Record gesture** (or press **R**) and type a name like `rock_on`
   (lowercase, digits and `_`).
2. A 3-second countdown gives you time to pose. Then hold the pose while the bar fills
   (about 1.5 s). Move your hand a little closer/further and tilt it slightly; that
   variety makes matching more reliable. **Esc** cancels.
3. Pick a `.png` for the reaction. It's copied into `assets/memes/<name>.png`. Or press
   Cancel to save just the gesture and map it later.
4. Done. It's live immediately, no restart. It's saved to `config/custom_gestures.json`,
   and the reaction is added to `config/reactions.json`.

Re-recording an existing name replaces it. To delete a gesture, remove it from
`custom_gestures.json` and its reaction from `reactions.json`.

**How matching works**

- Each frame, your hand's 21 landmarks are normalized: moved so the wrist is at (0, 0),
  scaled so the palm is length 1, and left hands mirrored onto right hands. Position,
  distance from the camera and which hand you use stop mattering. Use a reaction's
  `"hand": "left"` to make a gesture one-handed.
- The pose is compared with every recorded sample using RMS landmark distance, in palm
  lengths. The closest gesture wins if its distance is under `threshold` **and** it's
  clearly closer than the next-best gesture (so two similar recordings don't flicker).
- A custom match takes priority over MediaPipe's built-in label for that hand, then goes
  through the same hold-and-cooldown debouncer.

**Tuning:** with debug on (**D**), each hand shows `nearest <name> d=0.xx`. Hold your
gesture and note `d` (usually below 0.1), then do *other* poses and note theirs
(usually 0.3+). Put `threshold` between the two. Two settings keep a match steady:
`smoothing` averages the hand shape over recent frames (0 = off), and `release_factor`
keeps an active match until `d` exceeds `threshold × release_factor`, so it doesn't flicker
at the edge.

**Record with variety.** The recorder keeps the 20 most *different* frames, so while the
bar fills, move your hand a bit closer and farther and tilt it slightly. A perfectly
frozen hold gives a template that only recognizes that exact position. Set `rotation_invariant: true` if tilting
your hand should still match. Leave it `false` if orientation matters (thumb up vs down).

Static poses only; motion gestures (waves, swipes) need a different approach.

## How it works

```
FrameWorker (QThread)                                     UI thread
─────────────────────                                     ─────────
Camera.read() ─► flip (selfie) ─► downscale copy ─► GestureTracker
                     │                                   │ GestureFrame (data only)
                     │                   pick reaction ◄─┘
                     │                   GestureDebouncer (hold N frames, cooldown)
                     ▼                                   │ fired key
             full-size frame ◄── CornerOverlay / DebugRenderer
                     │
                     └── QImage ──── frame_ready signal ────► MainWindow / VideoView
```

- **Threading.** Only `FrameWorker` touches the camera and MediaPipe; both are created inside
  `run()`. The UI receives finished `QImage`s via a queued Qt signal, and sends commands
  (start/cancel recording, apply new settings) through a thread-safe queue.
- **Downscale for inference, draw at full size.** MediaPipe sees a copy no wider than
  `inference.max_width`; its landmarks are normalized (0–1), so they map directly onto the
  full-resolution frame.
- **Mirroring.** The frame is flipped before inference so the preview acts like a mirror.
  MediaPipe's Tasks API labels hands as they'd appear in a non-mirrored photo, so the
  tracker swaps its labels for the flipped frame. `handedness` always means your real hand.
  If the debug overlay still shows "Left" on your right hand (some camera drivers already
  mirror their output), set `"swap_handedness": true` under `camera`.
- **Debouncing.** A reaction fires once a gesture is held for `hold_frames` consecutive
  frames. It won't re-fire until you release or change the gesture, and every fire starts a
  global `cooldown_seconds`.
- **Tracking returns data only.** `tracking/` produces dataclasses from `core/types.py`; all
  drawing lives in `render/`.
- **Resource paths.** `paths.resource_root()` returns `sys._MEIPASS` inside a PyInstaller
  bundle and the project root otherwise, so `models/`, `assets/` and `config/` resolve in both.

### Layout

```
memecam/
├── pyproject.toml
├── config/
│   ├── reactions.json           # settings + gesture → image mapping
│   └── custom_gestures.json     # recorded poses (created on first recording)
├── models/                      # .task files (downloaded, git-ignored)
├── assets/memes/                # reaction PNGs
├── src/memecam/
│   ├── app.py                   # entry point: args, config, window, worker
│   ├── config.py                # pydantic schema, load_settings()/ConfigError
│   ├── storage.py               # saves a recorded gesture + image + reaction
│   ├── paths.py                 # resource_root() with sys._MEIPASS support
│   ├── core/
│   │   ├── types.py             # dataclasses shared between modules
│   │   ├── camera.py            # cv2.VideoCapture wrapper
│   │   ├── pipeline.py          # mirror → downscale → infer → classify → debounce → draw
│   │   ├── frame_worker.py      # QThread that runs the pipeline
│   │   └── fps.py
│   ├── tracking/gesture_tracker.py   # MediaPipe GestureRecognizer (Tasks API, VIDEO mode)
│   ├── gestures/
│   │   ├── debouncer.py
│   │   ├── templates.py         # normalize landmarks, RMS matching
│   │   ├── library.py           # custom_gestures.json schema, load/save
│   │   ├── recorder.py          # countdown → capture state machine
│   │   └── classifier.py        # custom match first, else built-in label
│   ├── render/
│   │   ├── overlay.py           # corner PNG alpha-blend
│   │   ├── debug.py             # landmarks, labels, FPS, match distances
│   │   └── recording.py         # countdown / progress HUD
│   └── ui/main_window.py
└── tests/
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

### Why the dependency override in `pyproject.toml`?

`mediapipe` depends on `opencv-contrib-python`, which bundles its own Qt plugins and conflicts
with PySide6. `[tool.uv] override-dependencies` removes it, so the only OpenCV installed is
`opencv-python-headless`. MediaPipe's Tasks API only needs `import cv2` to succeed, which the
headless build satisfies. If `uv pip list` ever shows `opencv-contrib-python` or
`opencv-python`, remove them.

## Roadmap

1. ~~MVP: gestures → corner meme, debug overlay, validated config, tests~~ ✅
2. FaceLandmarker + face-anchored overlays (eyes, forehead) that scale with face size
3. Animated GIF/WebP reactions + sound effects
4. ~~Recorded custom poses~~ ✅ (pulled forward). Still to do: rule-based gestures in
   `gestures/rules.py` (e.g. "hand raised on the left side")
5. Settings panel to remap gestures, saving back to `reactions.json`
6. Snapshot/clip recording + pyvirtualcam output
7. PyInstaller `.spec` bundling `models/`, `assets/`, `config/` and MediaPipe data. Note:
   recorded gestures are written next to `config/`, which is read-only/temporary inside a
   bundle, so Phase 7 moves user data to a per-user folder.
