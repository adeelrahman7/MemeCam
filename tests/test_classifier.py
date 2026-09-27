import numpy as np
import pytest

from hands import canonical, detection
from memecam.gestures.classifier import GestureClassifier, GestureSource
from memecam.gestures.templates import (
    GestureTemplate,
    TemplateMatcher,
    most_diverse,
    normalize,
    pose_distance,
)

ASPECT = 16 / 9
THRESHOLD = 0.2


def point_template() -> GestureTemplate:
    hand = detection("point")
    pose = normalize(hand.landmarks, hand.handedness, ASPECT)
    return GestureTemplate("erm_actually", np.stack([pose]))


def classifier(smoothing=0.5, release_factor=1.3) -> GestureClassifier:
    matcher = TemplateMatcher([point_template()], threshold=THRESHOLD, rotation_invariant=False)
    return GestureClassifier(matcher, THRESHOLD, smoothing=smoothing, release_factor=release_factor)


def between(a: str, b: str, t: float) -> np.ndarray:
    """A canonical hand shape t of the way from pose a to pose b."""
    return (1 - t) * canonical(a) + t * canonical(b)


def blend_at_distance(target: float) -> np.ndarray:
    """A point→peace blend whose distance from the 'point' template is ~target."""
    ref = normalize(detection("point").landmarks, detection("point").handedness, ASPECT)
    for t in np.linspace(0, 1, 2001):
        h = detection(between("point", "peace", t))  # type: ignore[arg-type]
        if pose_distance(normalize(h.landmarks, h.handedness, ASPECT), ref) >= target:
            return between("point", "peace", t)
    raise AssertionError("target distance not reachable")


def names(clf, shapes, gesture="Pointing_Up"):
    out = []
    for shape in shapes:
        (hg,) = clf.classify_frame([detection(shape, gesture=gesture)], ASPECT)
        out.append(hg.name)
    return out


def test_pose_hovering_around_threshold_does_not_flicker():
    inside, edge = blend_at_distance(0.15), blend_at_distance(0.23)
    shapes = [inside, inside, edge, inside, edge, edge, inside]

    # Without smoothing/hysteresis, the label flips to the built-in one at every edge frame.
    raw = names(classifier(smoothing=0.0, release_factor=1.0), shapes)
    assert "Pointing_Up" in raw

    # With them (the defaults), it holds steady.
    assert set(names(classifier(), shapes)) == {"erm_actually"}


def test_clearly_different_pose_still_releases():
    shapes = ["point"] * 3 + ["open"] * 4
    out = names(classifier(), shapes, gesture="Open_Palm")
    assert out[:3] == ["erm_actually"] * 3
    assert out[-1] == "Open_Palm"


def test_does_not_start_matching_above_threshold():
    edge = blend_at_distance(0.23)  # inside the release zone, but never matched first
    assert set(names(classifier(), [edge] * 5)) == {"Pointing_Up"}


def test_state_resets_when_hand_leaves():
    clf = classifier()
    names(clf, ["point"] * 3)
    clf.classify_frame([], ASPECT)  # hand gone
    (hg,) = clf.classify_frame([detection(blend_at_distance(0.23), gesture="Pointing_Up")], ASPECT)
    assert hg.name == "Pointing_Up"  # no leftover stickiness
    assert hg.source is GestureSource.BUILTIN


def test_rejects_bad_parameters():
    matcher = TemplateMatcher([], threshold=THRESHOLD, rotation_invariant=False)
    with pytest.raises(ValueError):
        GestureClassifier(matcher, THRESHOLD, smoothing=1.0)
    with pytest.raises(ValueError):
        GestureClassifier(matcher, THRESHOLD, release_factor=0.9)


def test_most_diverse_keeps_the_outliers():
    base = normalize(detection("point").landmarks, detection("point").handedness, ASPECT)
    far = normalize(detection("peace").landmarks, detection("peace").handedness, ASPECT)
    poses = [base] * 30 + [far] + [base] * 10  # one unusual frame in a steady hold
    idx = most_diverse(poses, 4)
    assert len(idx) == 4
    assert 30 in idx  # evenly spaced picking would most likely have missed it


def test_most_diverse_with_fewer_poses_than_requested():
    base = normalize(detection("point").landmarks, detection("point").handedness, ASPECT)
    assert most_diverse([base, base], 5) == [0, 1]
