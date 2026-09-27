# MemeCam

A desktop app that watches your webcam, recognizes hand gestures with MediaPipe, and pops a
meme reaction into the corner of the live video, or puts things on your face (sunglasses on
your eyes, a crown on your head).

**Status:** Phases 1–2 done, plus extras: gesture memes, recording your own hand poses
and face expressions, choosing where each image appears (a corner or a spot on your
face), managing/deleting gestures in the app, and face-anchored overlays. Developed and
tested on Windows; macOS/Linux should work but are untested.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a webcam. uv installs Python 3.12 for you if
needed.

```bash
uv sync                       # run in the project folder; creates .venv with all dependencies
```

Just installed uv and PowerShell says `uv` isn't recognized? Open a new terminal (or
restart VS Code) so it picks up the updated PATH.

### Download the model

MediaPipe's `.task` model files aren't on PyPI; download them into `models/`:

| File | Needed for | URL |
| --- | --- | --- |
| `gesture_recognizer.task` | **Required** | https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task |
| `face_landmarker.task` | Face overlays and face expressions | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |
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
Invoke-WebRequest -OutFile models/face_landmarker.task `
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

(The same `curl -L -o models/face_landmarker.task <url>` works on macOS/Linux.) Without the
face model the app still runs; it shows a warning and skips face overlays.

`models/*.task` is git-ignored.

## Run

```bash
uv run memecam            # or: uv run python -m memecam
uv run memecam --debug    # start with the debug overlay on
uv run memecam --config path/to/other.json
```

The top bar has **● Record**, **Gestures** and **Debug** buttons. Press **D** (or click
*Debug*) to toggle hand skeletons, `Left/Right: Gesture score`
labels, FPS, inference time, and the debouncer's hold progress / cooldown state.

Keys: **D** debug overlay · **R** record a gesture · **G** manage/delete gestures ·
**Esc** cancel recording.

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
  "custom_gestures": { "file": "config/custom_gestures.json", "threshold": 0.2, "face_threshold": 0.4,
                       "rotation_invariant": false, "countdown_seconds": 3.0, "capture_frames": 45 },
  "reactions": [
    { "gesture": "Thumb_Up", "image": "assets/memes/thumbs_up.png" },
    { "gesture": "Victory",  "image": "assets/memes/victory_left.png", "hand": "left" }
  ]
}
```

- Every section is optional and falls back to the defaults above. `reactions` and
  `face_overlays` can be empty (see [Face overlays](#face-overlays)).
- `overlay.stay_while_held`: the meme stays up while you hold the gesture, then disappears
  `linger_seconds` after you let go. Set it to `false` for a quick flash of `display_seconds`.
- `gesture` is either one of MediaPipe's built-in labels (`Closed_Fist`, `Open_Palm`,
  `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`) or the name of a gesture
  you recorded (see below). Built-in names are Capitalized, and custom names are lowercase,
  so they can't clash.
- `corner` (optional) puts that one meme in its own corner (`top_left`, `top_right`,
  `bottom_left`, `bottom_right`); otherwise it uses `overlay.corner`.
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

You can record two kinds of gesture:

- **Hand pose**: a shape you make with either hand (🤘, "erm actually" finger, ...).
- **Face expression**: a face you pull (mouth wide open, big smile, eyebrows up, kissy
  face, puffed cheeks, ...). Needs the face model (see Setup).

1. Click **● Record** (or press **R**). Choose **Hand pose** or **Face
   expression** and type a name like `rock_on` or `shocked_face` (lowercase, digits and
   `_`). Pick where its image should appear under **Show image**: a screen corner, or a
   spot on your face (eyes, forehead/head, nose, mouth, chin, whole face) with a size. The
   dialog remembers your last choices.
2. A 3-second countdown gives you time to get ready. Then hold the pose or expression
   while the bar fills (about 1.5 s). Add a little variety (move your hand closer and
   farther, or tilt your head slightly); that makes matching more reliable. **Esc** cancels.
3. Pick a `.png` for the reaction. It's copied into `assets/memes/<name>.png` and shown
   where you chose. Or press Cancel to save just the gesture and map it later.
4. Done. It's live immediately, no restart. It's saved to `config/custom_gestures.json`,
   and the reaction is added to `config/reactions.json`.

Re-recording an existing name replaces it. To delete a gesture, use **Manage gestures**
(below).

**How hand matching works**

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

**How face matching works**

- MediaPipe's face model reports 52 expression scores each frame (0–1: `jawOpen`,
  `mouthSmileLeft`, `browInnerUp`, `cheekPuff`, ...). A face expression is recorded as a
  set of those scores; where you're looking (`eyeLook*`) is ignored.
- Live scores are compared with the recording (distance between score sets). A match
  needs a distance under `custom_gestures.face_threshold` (default `0.4`; roughly, noise
  is ~0.1 and a big smile vs a resting face is ~1.0). The same `smoothing` and
  `release_factor` settings keep it steady.
- With debug on, each face shows `face: <name>`, `nearest <name> d=0.xx`, and its three
  strongest expression scores, handy for seeing what the model thinks you're doing.
- **Exaggerate.** Subtle expressions look a lot like your resting face, and blinking or
  talking can set them off. If a recording is too subtle, the app warns you after saving.
- A face expression triggers memes and face overlays just like a hand gesture. The
  `"hand"` filter on reactions doesn't apply to faces.

Static poses and expressions only; motion gestures (waves, nods, swipes) need a different
approach.

## Managing and deleting gestures

Click **Gestures** (or press **G**) to see every gesture that's set up: its type
and what it shows where. Changes apply immediately, with no restart.

- **Change placement…** moves a gesture's image to another corner or face spot, keeping
  the same image. If a gesture shows several images, pick which one to move. A gesture can
  have one corner meme (per hand) but any number of face images.
- **Delete…** (or the **Delete** key) removes a gesture. A confirmation lists exactly
  what will be removed.

- **Custom gesture** (a hand pose or face expression you recorded): removes the recording,
  its corner meme and the face overlays it triggers. The **Type** column shows which kind
  it is.
- **Built-in gesture** (e.g. `Thumb_Up`): MediaPipe's model will still recognize it, but
  its meme and face overlays are removed, so it no longer does anything. Add it back by
  editing `config/reactions.json`.
- Image files are never deleted; they stay in `assets/`.

## Face overlays

Images attached to your face that move, scale and tilt with it. Each entry in
`face_overlays` in `config/reactions.json`:

```json
"face": { "model": "models/face_landmarker.task", "num_faces": 1, "min_confidence": 0.5, "smoothing": 0.5 },
"face_overlays": [
  { "image": "assets/face/sunglasses.png", "anchor": "eyes", "width": 1.4,
    "trigger": "Closed_Fist", "linger_seconds": 1.0 },
  { "image": "assets/face/crown.png", "anchor": "forehead", "width": 1.3, "offset_y": -0.45,
    "trigger": "Thumb_Up", "linger_seconds": 1.0 }
]
```

| Key | Meaning |
| --- | --- |
| `anchor` | Where the image's center goes: `eyes`, `forehead`, `nose`, `mouth`, `chin`, `face` |
| `width` | Image width as a multiple of the distance between your outer eye corners, so it scales with your face |
| `offset_x`, `offset_y` | Shift in units of the image's own width/height, along your head's axes (tilts with you). `offset_y: -0.5` sits the image's bottom edge on the anchor, which suits hats |
| `trigger` | `null` = always on. A gesture name (built-in or custom) = shown while you hold it |
| `linger_seconds` | With a trigger, how long it stays after you drop the gesture |

The defaults: make a **fist** for sunglasses, give a **thumbs-up** for a crown (plus the
NICE. meme). One gesture can drive both a corner meme and a face overlay. Every detected face
gets the overlays (`face.num_faces` sets how many are tracked).

**Making your own images:** use a transparent PNG and design it so the part that belongs on
the anchor is at the image's center. For sunglasses, the lens centers should sit on the
horizontal center line. With debug on (**D**) you'll see the face mesh, each anchor's
position, the head tilt, and the eye distance in pixels.

`face.smoothing` averages the landmarks over recent frames to stop overlays jittering
(0 = off; higher = steadier but laggier). The face model only runs when `face_overlays`
isn't empty, so hand-only setups keep their full frame rate.

## Troubleshooting

**Low FPS.** Turn on Debug (**D**) and compare `FPS` with `infer … ms`:

- **`infer` is small but FPS is low (e.g. 10–15):** the webcam itself is slow, usually
  because it's dim. Webcams lengthen exposure in low light, which lowers their frame rate.
  - Add light in front of you.
  - Turn off the driver's low-light slowdown: in `src/memecam/core/camera.py`, uncomment
    `self._cap.set(cv2.CAP_PROP_SETTINGS, 1)`, run once, and in the window that opens
    untick *Low Light Compensation* or set *Exposure* to manual. Then comment the line
    out again; the driver remembers the setting.
  - Some webcams only reach full speed at 720p in MJPG mode: uncomment the
    `CAP_PROP_FOURCC … "MJPG"` line in the same file.
- **`infer` is high (30 ms+):** lower `inference.max_width` (e.g. `480`), set
  `num_hands` to `1`, or remove face overlays / face gestures you don't use (the face
  model only runs when something needs it).

**Camera won't open.** Close other apps using it (Teams, Zoom, the Camera app) and check
*Settings → Privacy & security → Camera → Let desktop apps access your camera*. With
several cameras, change `camera.index` (0, 1, …).

**Left and right hands are swapped** (debug shows "Left" on your right hand): set
`"swap_handedness": true` under `camera`.

**"Face overlays and face expressions are disabled".** `models/face_landmarker.task` is
missing; download it (see Setup). Hand gestures keep working without it.

**A custom gesture misfires or won't trigger.** In debug, watch its `d` value. Re-record
with a bit of movement for variety, or adjust `custom_gestures.threshold` (hands) /
`face_threshold` (faces). Exaggerate face expressions.

**Stopping the app.** Close the window. Ctrl+C in the terminal may not stop a Qt app; if
it hangs, use Ctrl+Break or close the terminal.

**Your data.** Recorded gestures live in `config/custom_gestures.json`, and what they
trigger lives in `config/reactions.json`. Back them up (or commit them) to keep your
gestures. Meme images you add go in `assets/memes/`; be careful about committing
copyrighted images to a public repo.

## How it works

```
FrameWorker (QThread)                                     UI thread
─────────────────────                                     ─────────
Camera.read() ─► flip (selfie) ─► downscale copy ─► GestureTracker (+ FaceTracker)
                     │                                   │ GestureFrame, FaceFrame (data only)
                     │                   pick reaction ◄─┘
                     │                   GestureDebouncer (hold N frames, cooldown)
                     ▼                                   │ fired key
             full-size frame ◄── FaceOverlayLayer / CornerOverlay / DebugRenderer
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
├── assets/
│   ├── memes/                   # corner reaction PNGs
│   └── face/                    # face overlay PNGs (sunglasses, crown)
├── src/memecam/
│   ├── app.py                   # entry point: args, config, window, worker
│   ├── config.py                # pydantic schema, load_settings()/ConfigError
│   ├── storage.py               # save / list / delete / re-place gestures
│   ├── placement.py             # corner vs face-spot placement + per-spot defaults
│   ├── paths.py                 # resource_root() with sys._MEIPASS support
│   ├── core/
│   │   ├── types.py             # dataclasses shared between modules
│   │   ├── camera.py            # cv2.VideoCapture wrapper
│   │   ├── pipeline.py          # mirror → downscale → infer → classify → debounce → draw
│   │   ├── frame_worker.py      # QThread that runs the pipeline
│   │   └── fps.py
│   ├── tracking/
│   │   ├── gesture_tracker.py   # MediaPipe GestureRecognizer (Tasks API, VIDEO mode)
│   │   └── face_tracker.py      # MediaPipe FaceLandmarker (Tasks API, VIDEO mode)
│   ├── face/
│   │   ├── anchors.py           # anchor points, eye-distance scale, head tilt, smoothing
│   │   └── layer.py             # which face overlays are on, and where they go
│   ├── gestures/
│   │   ├── debouncer.py
│   │   ├── templates.py         # normalize landmarks, RMS matching
│   │   ├── library.py           # custom_gestures.json schema, load/save
│   │   ├── recorder.py          # countdown → capture state machine
│   │   ├── classifier.py        # custom match first, else built-in label
│   │   ├── expressions.py       # face-expression matching (blendshape scores)
│   │   └── activator.py         # hold-to-show state for gesture-triggered overlays
│   ├── render/
│   │   ├── overlay.py           # corner PNG alpha-blend
│   │   ├── face_overlay.py      # scaled + rotated face images (premultiplied alpha)
│   │   ├── debug.py             # landmarks, labels, FPS, match distances, face anchors
│   │   └── recording.py         # countdown / progress HUD
│   └── ui/
│       ├── main_window.py       # top bar + video + toasts
│       ├── theme.py             # dark theme (palette + stylesheet)
│       ├── widgets.py           # smooth video view, fading toast messages
│       ├── record_dialog.py     # Record a gesture: name + hand/face + placement
│       ├── placement_picker.py  # placement dropdown + Change placement window
│       └── gesture_manager.py   # Manage gestures window (list + delete)
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
2. ~~FaceLandmarker + face-anchored overlays that scale with face size~~ ✅
3. Animated GIF/WebP reactions + sound effects
4. ~~Recorded custom poses~~ ✅ (pulled forward). Still to do: rule-based gestures in
   `gestures/rules.py` (e.g. "hand raised on the left side")
5. Settings panel to remap gestures, saving back to `reactions.json`
6. Snapshot/clip recording + pyvirtualcam output
7. PyInstaller `.spec` bundling `models/`, `assets/`, `config/` and MediaPipe data. Note:
   recorded gestures are written next to `config/`, which is read-only/temporary inside a
   bundle, so Phase 7 moves user data to a per-user folder.
