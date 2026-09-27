import math

import numpy as np
import pytest

from hands import POSES, landmarks
from memecam.core.types import Handedness, Landmark
from memecam.gestures.templates import (
    GestureTemplate,
    TemplateMatcher,
    align_rotation,
    normalize,
    pose_distance,
)

R, L = Handedness.RIGHT, Handedness.LEFT
ASPECT = 16 / 9
DEFAULT_THRESHOLD = 0.2


def pose(name: str, hand: Handedness = R, **kwargs):
    kwargs.setdefault("aspect", ASPECT)
    result = normalize(landmarks(name, left=hand is L, **kwargs), hand, kwargs["aspect"])
    assert result is not None
    return result


def template(name: str, *poses) -> GestureTemplate:
    return GestureTemplate(name=name, samples=np.stack(poses))


# --- normalize --------------------------------------------------------------------------
def test_normalized_pose_has_wrist_at_origin_and_unit_palm():
    p = pose("open")
    assert p.shape == (21, 2)
    assert np.allclose(p[0], 0)
    assert np.linalg.norm(p[9]) == pytest.approx(1.0)


def test_position_and_size_do_not_matter():
    a = pose("peace", center=(0.3, 0.7), size=0.10)
    b = pose("peace", center=(0.7, 0.4), size=0.25)
    assert pose_distance(a, b) < 1e-9


def test_aspect_ratio_is_undone():
    a = pose("rock", aspect=16 / 9)
    b = pose("rock", aspect=4 / 3)
    assert pose_distance(a, b) < 1e-9


def test_left_hand_matches_right_hand_recording():
    assert pose_distance(pose("rock", R), pose("rock", L)) < 1e-9


def test_degenerate_input_returns_none():
    assert normalize((), R, ASPECT) is None
    same = tuple(Landmark(0.5, 0.5, 0.0) for _ in range(21))
    assert normalize(same, R, ASPECT) is None


def test_align_rotation_undoes_tilt():
    upright = pose("peace")
    tilted = pose("peace", angle_deg=35)
    assert pose_distance(upright, tilted) > 0.3
    assert pose_distance(align_rotation(upright), align_rotation(tilted)) < 1e-9


# --- the metric separates real poses ------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b"), [(a, b) for a in POSES for b in POSES if a < b], ids=lambda x: x
)
def test_different_poses_are_further_apart_than_the_default_threshold(a, b):
    assert pose_distance(pose(a), pose(b)) > DEFAULT_THRESHOLD


def test_jittered_pose_stays_within_threshold():
    clean = pose("rock")
    noisy = pose("rock", jitter=0.05, seed=3)
    assert pose_distance(clean, noisy) < DEFAULT_THRESHOLD


# --- matcher ------------------------------------------------------------------------
def make_matcher(*templates, rotation_invariant=False, threshold=DEFAULT_THRESHOLD):
    return TemplateMatcher(templates, threshold=threshold, rotation_invariant=rotation_invariant)


def test_matches_the_right_template():
    m = make_matcher(template("rock_on", pose("rock")), template("peace_out", pose("peace")))
    result = m.match(pose("rock", jitter=0.03, seed=1))
    assert result.name == "rock_on"
    assert result.nearest == "rock_on"
    assert result.distance < 0.1


def test_unrelated_pose_is_rejected_but_nearest_is_reported():
    m = make_matcher(template("rock_on", pose("rock")))
    result = m.match(pose("fist"))
    assert result.name is None
    assert result.nearest == "rock_on"
    assert result.distance > DEFAULT_THRESHOLD


def test_uses_the_closest_of_several_samples():
    m = make_matcher(template("wave", pose("fist"), pose("open")))
    assert m.match(pose("open")).distance < 1e-9


def test_ambiguous_match_is_rejected():
    # Two templates recorded from nearly the same pose: neither should win.
    a = pose("peace")
    b = pose("peace", jitter=0.03, seed=7)
    m = make_matcher(template("one", a), template("two", b), threshold=0.5)
    halfway = (a + b) / 2  # equally close to both
    result = m.match(halfway)
    assert result.distance < 0.5  # it's within the threshold...
    assert result.name is None  # ...but rejected as ambiguous
    assert m.match(a).name == "one"  # an unambiguous pose still matches


def test_rotation_invariance_is_optional():
    tilted = pose("point", angle_deg=40)
    strict = make_matcher(template("pointy", pose("point")))
    loose = make_matcher(template("pointy", pose("point")), rotation_invariant=True)
    assert strict.match(tilted).name is None
    assert loose.match(tilted).name == "pointy"


def test_empty_matcher():
    result = make_matcher().match(pose("open"))
    assert result.name is None and result.nearest is None and math.isinf(result.distance)


def test_template_shape_is_checked():
    with pytest.raises(ValueError, match="shape"):
        GestureTemplate("bad", np.zeros((3, 20, 2)))
    with pytest.raises(ValueError, match="at least one"):
        GestureTemplate("bad", np.zeros((0, 21, 2)))
