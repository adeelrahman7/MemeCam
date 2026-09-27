import numpy as np
import pytest

from faces import EXPRESSIONS, expression, face_detection
from memecam.core.types import FaceDetection
from memecam.gestures.expressions import (
    BLENDSHAPE_NAMES,
    ExpressionClassifier,
    ExpressionMatcher,
    expression_distances,
    expression_strength,
    top_blendshapes,
)
from memecam.gestures.templates import GestureKind, GestureTemplate

THRESHOLD = 0.4


def tpl(name: str, *exprs: str, jitter=0.0) -> GestureTemplate:
    samples = np.stack([expression(e, jitter=jitter, seed=i) for i, e in enumerate(exprs)])
    return GestureTemplate(name, samples, kind=GestureKind.FACE)


def dist(a: str, b: str, **kw) -> float:
    return float(expression_distances(expression(a)[None], expression(b, **kw))[0])


@pytest.mark.parametrize(("a", "b"), [(a, b) for a in EXPRESSIONS for b in EXPRESSIONS if a < b
                                      and "slight_smile" not in (a, b)])  # fmt: skip
def test_distinct_expressions_are_beyond_the_default_threshold(a, b):
    assert dist(a, b) > THRESHOLD


def test_noise_stays_within_threshold():
    assert dist("shocked", "shocked", jitter=0.03, seed=1) < THRESHOLD


def test_gaze_is_ignored():
    looking_away = expression("smile", eyeLookOutLeft=1.0, eyeLookInRight=1.0, eyeLookUpLeft=0.8)
    assert float(expression_distances(expression("smile")[None], looking_away)[0]) == 0.0


def test_strength_and_top_blendshapes():
    assert expression_strength(np.stack([expression("shocked")])) == pytest.approx(0.8)
    assert expression_strength(np.stack([expression("slight_smile")])) < 0.3
    assert top_blendshapes(expression("shocked"), 2) == [("jawOpen", 0.8), ("browInnerUp", 0.6)]
    assert len(BLENDSHAPE_NAMES) == 52


def test_matcher_picks_the_right_expression_and_ignores_hand_templates():
    hand = GestureTemplate("a_hand", np.zeros((1, 21, 2)))
    m = ExpressionMatcher([tpl("shocked_face", "shocked"), tpl("happy", "smile"), hand],
                          threshold=THRESHOLD)  # fmt: skip
    assert m.names == ["shocked_face", "happy"]
    assert m.match(expression("shocked", jitter=0.03, seed=5)).name == "shocked_face"
    assert m.match(expression("smile")).name == "happy"
    neutral = m.match(expression("neutral"))
    assert neutral.name is None and neutral.nearest is not None


def faces_of(*exprs: np.ndarray) -> list[FaceDetection]:
    return [face_detection(blendshapes=e) for e in exprs]


def classifier(**kw) -> ExpressionClassifier:
    m = ExpressionMatcher([tpl("shocked_face", "shocked")], threshold=THRESHOLD)
    return ExpressionClassifier(m, THRESHOLD, **kw)


def blend(t: float) -> np.ndarray:
    return (1 - t) * expression("shocked") + t * expression("neutral")


def at_distance(target: float) -> np.ndarray:
    ref = expression("shocked")
    for t in np.linspace(0, 1, 2001):
        if float(expression_distances(ref[None], blend(t))[0]) >= target:
            return blend(t)
    raise AssertionError


def test_classifier_holds_steady_near_the_threshold():
    inside, edge = at_distance(0.3), at_distance(0.45)
    seq = [inside, inside, edge, inside, edge, edge]
    raw = classifier(smoothing=0.0, release_factor=1.0)
    assert None in [raw.classify_frame(faces_of(s))[0].name for s in seq]
    steady = classifier()
    assert {steady.classify_frame(faces_of(s))[0].name for s in seq} == {"shocked_face"}


def test_classifier_resets_when_faces_change_and_handles_missing_scores():
    clf = classifier(smoothing=0.0)
    assert clf.classify_frame(faces_of(expression("shocked")))[0].name == "shocked_face"
    two = clf.classify_frame(faces_of(expression("neutral"), expression("shocked")))
    assert [g.name for g in two] == [None, "shocked_face"]
    (no_scores,) = clf.classify_frame([face_detection()])  # blendshapes unavailable
    assert no_scores.name is None and no_scores.scores is None
    assert clf.classify_frame([]) == ()


def test_face_template_shape_is_checked():
    with pytest.raises(ValueError, match=r"face samples must have shape \(n, 52\)"):
        GestureTemplate("bad", np.zeros((2, 21, 2)), kind=GestureKind.FACE)
    with pytest.raises(ValueError, match="hand samples"):
        GestureTemplate("bad", np.zeros((2, 52)))
