# MemeCam

A desktop app that watches your webcam, recognizes hand gestures with MediaPipe, and pops a
meme reaction into the corner of the live video.

**Status:** Phase 1 (MVP): camera → GestureRecognizer → debouncer → corner PNG, with a debug
overlay.

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
  "camera":    { "index": 0, "width": 1280, "height": 720, "mirror": true },
  "inference": { "model": "models/gesture_recognizer.task", "max_width": 640,
                 "num_hands": 2, "min_detection_confidence": 0.5, "min_gesture_score": 0.6 },
  "debounce":  { "hold_frames": 6, "cooldown_seconds": 1.5 },
  "overlay":   { "corner": "top_right", "width_fraction": 0.3, "margin_px": 16, "display_seconds": 2.0 },
  "reactions": [
    { "gesture": "Thumb_Up", "image": "assets/memes/thumbs_up.png" },
    { "gesture": "Victory",  "image": "assets/memes/victory_left.png", "hand": "left" }
  ]
}
```

- Every section except `reactions` is optional and falls back to the defaults above.
- `gesture` must be one of MediaPipe's built-in labels: `Closed_Fist`, `Open_Palm`,
  `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.
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
  `run()`. The UI receives finished `QImage`s via a queued Qt signal.
- **Downscale for inference, draw at full size.** MediaPipe sees a copy no wider than
  `inference.max_width`; its landmarks are normalized (0–1), so they map directly onto the
  full-resolution frame.
- **Mirroring.** The frame is flipped before inference. MediaPipe labels handedness assuming
  a mirrored selfie image, so "Right" is your right hand, and it also appears on the right
  side of the preview. If you set `mirror: false`, the tracker swaps the labels so
  `handedness` still means your real hand.
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
├── config/reactions.json
├── models/                      # .task files (downloaded, git-ignored)
├── assets/memes/                # reaction PNGs
├── src/memecam/
│   ├── app.py                   # entry point: args, config, window, worker
│   ├── config.py                # pydantic schema + load_config()/ConfigError
│   ├── paths.py                 # resource_root() with sys._MEIPASS support
│   ├── core/
│   │   ├── types.py             # dataclasses shared between modules
│   │   ├── camera.py            # cv2.VideoCapture wrapper
│   │   ├── pipeline.py          # mirror → downscale → infer → debounce → draw (no Qt)
│   │   ├── frame_worker.py      # QThread that runs the pipeline
│   │   └── fps.py
│   ├── tracking/gesture_tracker.py   # MediaPipe GestureRecognizer (Tasks API, VIDEO mode)
│   ├── gestures/debouncer.py
│   ├── render/
│   │   ├── overlay.py           # corner PNG alpha-blend
│   │   └── debug.py             # landmarks, labels, FPS
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
4. Custom landmark-based gestures in `gestures/rules.py`
5. Settings panel to remap gestures, saving back to `reactions.json`
6. Snapshot/clip recording + pyvirtualcam output
7. PyInstaller `.spec` bundling `models/`, `assets/`, `config/` and MediaPipe data
