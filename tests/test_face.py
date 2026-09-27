import math

import numpy as np
import pytest

from faces import face_px
from memecam.face.anchors import Anchor, FaceSmoother, Placement, head_roll, place
from memecam.gestures.activator import HoldActivator
from memecam.render.face_overlay import FaceOverlayRenderer

# --- anchors ------------------------------------------------------------------------


def test_eyes_anchor_center_scale_and_level_angle():
    p = place(face_px(center=(640, 300), eye_dist=100), Anchor.EYES)
    assert p is not None
    assert p.center == pytest.approx((640, 300))
    assert p.scale == pytest.approx(100)
    assert p.angle == pytest.approx(0)


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (Anchor.FOREHEAD, (640, 200)),
        (Anchor.NOSE, (640, 355)),
        (Anchor.MOUTH, (640, 400)),
        (Anchor.CHIN, (640, 460)),
        (Anchor.FACE, (640, 330)),
    ],
)
def test_anchor_positions(anchor, expected):
    p = place(face_px(center=(640, 300), eye_dist=100), anchor)
    assert p is not None and p.center == pytest.approx(expected)


@pytest.mark.parametrize("deg", [-40, -10, 0, 15, 45])
@pytest.mark.parametrize("mirrored", [False, True])
def test_head_tilt_is_measured_and_never_upside_down(deg, mirrored):
    angle = head_roll(face_px(angle_deg=deg, mirrored=mirrored))
    assert math.degrees(angle) == pytest.approx(deg)


def test_scale_follows_face_size():
    near = place(face_px(eye_dist=180), Anchor.EYES)
    far = place(face_px(eye_dist=60), Anchor.EYES)
    assert near is not None and far is not None
    assert near.scale / far.scale == pytest.approx(3)


def test_degenerate_face_is_skipped():
    assert place(np.zeros((100, 2)), Anchor.EYES) is None  # too few landmarks
    assert place(np.zeros((478, 2)), Anchor.EYES) is None  # collapsed


def test_smoother_averages_small_moves_and_snaps_on_big_jumps():
    sm = FaceSmoother(0.5)
    a = face_px(center=(600, 300))
    sm.update([a])
    (moved,) = sm.update([face_px(center=(610, 300))])
    assert moved[1][0] == pytest.approx(605)  # halfway: smoothed
    (jumped,) = sm.update([face_px(center=(1000, 300))])
    assert jumped[1][0] == pytest.approx(1000)  # far jump: snapped
    two = sm.update([a, a])  # face count changed: no smoothing
    assert len(two) == 2 and np.allclose(two[0], a)


# --- activator ----------------------------------------------------------------------


def test_hold_activator():
    act = HoldActivator(hold_frames=3, linger_seconds=0.5)
    dt = 1 / 30
    states = [act.update(True, i * dt) for i in range(5)]
    assert states == [False, False, True, True, True]
    assert act.update(False, 5 * dt)  # lingering
    assert not act.update(False, 5 * dt + 0.6)  # gone
    assert not act.update(True, 1.0)  # needs a fresh hold


def test_hold_activator_interrupted_hold_restarts():
    act = HoldActivator(hold_frames=3, linger_seconds=0)
    assert [act.update(p, i) for i, p in enumerate([True, True, False, True, True, True])] == [
        False, False, False, False, False, True,
    ]  # fmt: skip


# --- renderer -----------------------------------------------------------------------


@pytest.fixture
def red_bar(tmp_path):
    """A 100x20 image: left half red, right half blue, fully opaque."""
    from PIL import Image

    img = Image.new("RGBA", (100, 20), (255, 0, 0, 255))
    img.paste((0, 0, 255, 255), (50, 0, 100, 20))
    path = tmp_path / "bar.png"
    img.save(path)
    return FaceOverlayRenderer({"bar": path})


def blank():
    return np.zeros((400, 600, 3), np.uint8)


RED, BLUE = (0, 0, 255), (255, 0, 0)  # BGR


def test_renders_scaled_at_center(red_bar):
    frame = blank()
    red_bar.draw(frame, "bar", Placement((300, 200), 100, 0.0), width=2.0)  # 200 px wide
    assert tuple(frame[200, 220]) == RED  # left half
    assert tuple(frame[200, 380]) == BLUE  # right half
    assert tuple(frame[200, 190]) == (0, 0, 0)  # just outside (starts at x=200)
    assert tuple(frame[200, 410]) == (0, 0, 0)


def test_rotates_with_head(red_bar):
    frame = blank()
    red_bar.draw(frame, "bar", Placement((300, 200), 100, math.radians(90)), width=2.0)
    assert tuple(frame[120, 300]) == RED  # the bar now points down: left end is on top
    assert tuple(frame[280, 300]) == BLUE
    assert tuple(frame[200, 220]) == (0, 0, 0)


def test_offset_moves_along_head_axes(red_bar):
    frame = blank()
    # 200x40 bar, offset_y=-0.5 → bottom edge on the anchor (a "hat").
    red_bar.draw(frame, "bar", Placement((300, 200), 100, 0.0), width=2.0, offset_y=-0.5)
    assert tuple(frame[190, 250]) == RED
    assert tuple(frame[205, 250]) == (0, 0, 0)


def test_partially_and_fully_offscreen_is_safe(red_bar):
    frame = blank()
    red_bar.draw(frame, "bar", Placement((10, 10), 100, 0.3), width=2.0)
    assert frame.any()
    before = frame.copy()
    red_bar.draw(frame, "bar", Placement((5000, 5000), 100, 0.0), width=2.0)
    assert np.array_equal(frame, before)


def test_small_overlay_uses_prescaled_copy(red_bar):
    frame = blank()
    red_bar.draw(frame, "bar", Placement((300, 200), 10, 0.0), width=2.0)  # 20 px wide
    assert tuple(frame[200, 294]) == RED
    assert tuple(frame[200, 306]) == BLUE
