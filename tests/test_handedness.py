import pytest

from memecam.core.types import Handedness
from memecam.tracking.gesture_tracker import resolve_handedness

L, R = Handedness.LEFT, Handedness.RIGHT


@pytest.mark.parametrize(
    ("label", "mirrored", "swap", "expected"),
    [
        # Mirrored selfie frame: MediaPipe's label is backwards, so it's swapped.
        ("Left", True, False, R),
        ("Right", True, False, L),
        # Raw (non-mirrored) frame: label is used as-is.
        ("Left", False, False, L),
        ("Right", False, False, R),
        # Manual override inverts whatever the automatic rule decided.
        ("Left", True, True, L),
        ("Right", False, True, L),
    ],
)
def test_resolve_handedness(label, mirrored, swap, expected):
    assert resolve_handedness(label, input_is_mirrored=mirrored, swap=swap) is expected
