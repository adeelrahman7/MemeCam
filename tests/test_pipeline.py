import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hands import detection
from memecam.config import load_settings
from memecam.core.pipeline import FramePipeline
from memecam.core.types import GestureFrame, HandDetection
from memecam.gestures.classifier import GestureSource
from memecam.gestures.recorder import RecordPhase
from memecam.storage import save_recorded_gesture

DT = 1 / 30


class FakeTracker:
    """Returns whatever hands the test puts in ``current``."""

    def __init__(self) -> None:
        self.current: tuple[HandDetection, ...] = ()
        self.input_shapes: list[tuple[int, ...]] = []

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> GestureFrame:
        self.input_shapes.append(frame_rgb.shape)
        return GestureFrame(self.current)


@pytest.fixture
def settings(tmp_path: Path):
    memes = tmp_path / "assets" / "memes"
    memes.mkdir(parents=True)
    for name, color in [("up.png", (0, 255, 0, 255)), ("rock.png", (255, 0, 0, 255))]:
        Image.new("RGBA", (40, 20), color).save(memes / name)
    (tmp_path / "config").mkdir()
    cfg = {
        "debounce": {"hold_frames": 3, "cooldown_seconds": 0},
        "custom_gestures": {"countdown_seconds": 0.5, "capture_frames": 10},
        "reactions": [{"gesture": "Thumb_Up", "image": "assets/memes/up.png"}],
    }
    (tmp_path / "config" / "reactions.json").write_text(json.dumps(cfg))
    return load_settings(tmp_path / "config" / "reactions.json", tmp_path)


def frame() -> np.ndarray:
    return np.full((720, 1280, 3), 50, np.uint8)


def run(pipe, n, start):
    return [pipe.process(frame(), start + i * DT) for i in range(n)]


def test_inference_runs_on_downscaled_copy(settings):
    tracker = FakeTracker()
    FramePipeline(settings, tracker).process(frame(), 0.0)
    assert tracker.input_shapes[0] == (360, 640, 3)


def test_builtin_gesture_fires_after_hold(settings):
    tracker = FakeTracker()
    pipe = FramePipeline(settings, tracker)
    tracker.current = (detection("fist", gesture="Thumb_Up"),)
    fired = [r.fired_reaction for r in run(pipe, 4, 0.0)]
    assert fired == [None, None, "Thumb_Up", None]


def test_record_then_trigger_custom_gesture(settings):
    tracker = FakeTracker()
    pipe = FramePipeline(settings, tracker)

    # 1. Record "rock_on". The model thinks it's a thumbs-up, but no meme may fire
    #    while recording.
    tracker.current = (detection("rock", gesture="Thumb_Up", jitter=0.02),)
    pipe.start_recording("rock_on", 0.0)
    results = run(pipe, 70, 0.0)  # 0.5 s countdown + 10 capture frames ≈ frame 25
    done_at = [
        i for i, r in enumerate(results) if r.recording and r.recording.phase is RecordPhase.DONE
    ]
    assert len(done_at) == 1
    # Nothing fires while recording, nor during the quiet period right after it.
    assert all(r.fired_reaction is None for r in results[: done_at[0] + 40])
    template = results[done_at[0]].recording.template
    assert template is not None

    # 2. Save it with an image and reload, like the UI does.
    new_settings = save_recorded_gesture(
        settings, template, settings.root / "assets/memes/rock.png"
    )
    pipe = FramePipeline(new_settings, tracker)

    # 3. The same pose from the *left* hand, elsewhere and bigger: the custom gesture
    #    beats MediaPipe's built-in label and fires its reaction.
    tracker.current = (
        detection("rock", left=True, gesture="Thumb_Up", center=(0.3, 0.7), size=0.2),
    )
    results = run(pipe, 4, 10.0)
    hand = results[0].hands[0]
    assert hand.name == "rock_on"
    assert hand.source is GestureSource.CUSTOM
    assert [r.fired_reaction for r in results] == [None, None, "rock_on", None]

    # The reaction image (pure red) is drawn in the top-right corner.
    img = results[-1].image_bgr
    assert tuple(img[40, 1200]) == (0, 0, 255)

    # 4. A different pose falls back to the built-in label.
    tracker.current = (detection("open", gesture="Open_Palm"),)
    hand = pipe.process(frame(), 20.0).hands[0]
    assert hand.name == "Open_Palm"
    assert hand.source is GestureSource.BUILTIN
    assert hand.custom is not None and hand.custom.nearest == "rock_on"


def _meme_visible(img: np.ndarray) -> bool:
    return tuple(img[40, 1200]) == (0, 255, 0)  # up.png is pure green, top-right corner


def test_meme_stays_while_gesture_is_held_then_lingers(settings):
    tracker = FakeTracker()
    pipe = FramePipeline(settings, tracker)
    tracker.current = (detection("fist", gesture="Thumb_Up"),)
    held = run(pipe, 150, 0.0)  # 5 seconds of holding; display_seconds is only 2
    assert [r.fired_reaction for r in held].count("Thumb_Up") == 1
    assert all(_meme_visible(r.image_bgr) for r in held[2:])

    tracker.current = ()  # let go
    after = run(pipe, 30, 5.0)  # linger_seconds defaults to 0.5
    assert _meme_visible(after[5].image_bgr)
    assert not _meme_visible(after[-1].image_bgr)


def test_flash_mode_hides_meme_after_display_seconds(tmp_path, settings):
    raw = json.loads(settings.config_path.read_text())
    raw["overlay"] = {"stay_while_held": False}
    settings.config_path.write_text(json.dumps(raw))
    flash = load_settings(settings.config_path, settings.root)

    tracker = FakeTracker()
    pipe = FramePipeline(flash, tracker)
    tracker.current = (detection("fist", gesture="Thumb_Up"),)
    held = run(pipe, 150, 0.0)
    assert _meme_visible(held[30].image_bgr)
    assert not _meme_visible(held[-1].image_bgr)  # gone after 2 s even though still held
