import pytest

from hands import detection
from memecam.core.types import Handedness
from memecam.gestures.recorder import GestureRecorder, RecordPhase

ASPECT = 16 / 9
DT = 1 / 30


def make(countdown=1.0, frames=10, max_samples=4, timeout=2.0) -> GestureRecorder:
    return GestureRecorder(
        countdown_seconds=countdown,
        capture_frames=frames,
        max_samples=max_samples,
        timeout_seconds=timeout,
    )


def run(rec, hands_per_frame, start=0.0):
    """Feed frames at 30 fps; return all statuses."""
    return [rec.update(h, ASPECT, start + i * DT) for i, h in enumerate(hands_per_frame)]


def test_idle_recorder_returns_none():
    rec = make()
    assert not rec.active
    assert rec.update([detection("rock")], ASPECT, 0.0) is None


def test_full_recording_flow():
    rec = make(countdown=1.0, frames=10, max_samples=4)
    rec.start("rock_on", 0.0)
    hand = [detection("rock")]

    first = rec.update(hand, ASPECT, 0.0)
    assert first is not None and first.phase is RecordPhase.COUNTDOWN
    assert first.seconds_left == pytest.approx(1.0)

    statuses = run(rec, [hand] * 60, start=0.5)
    phases = [s.phase for s in statuses if s is not None]
    assert phases[0] is RecordPhase.COUNTDOWN
    assert RecordPhase.CAPTURING in phases
    done = [s for s in statuses if s is not None and s.phase is RecordPhase.DONE]
    assert len(done) == 1  # reported exactly once
    template = done[0].template
    assert template is not None
    assert template.name == "rock_on"
    assert template.samples.shape == (4, 21, 2)  # thinned to max_samples
    assert template.recorded_with is Handedness.RIGHT
    assert not rec.active
    assert statuses[-1] is None  # idle again afterwards


def test_no_countdown_captures_immediately():
    rec = make(countdown=0.0, frames=3)
    rec.start("x_pose", 0.0)
    statuses = run(rec, [[detection("peace")]] * 3)
    assert [s.phase for s in statuses] == [
        RecordPhase.CAPTURING,
        RecordPhase.CAPTURING,
        RecordPhase.DONE,
    ]


def test_frames_without_a_hand_are_skipped():
    rec = make(countdown=0.0, frames=3)
    rec.start("x_pose", 0.0)
    hand = [detection("peace")]
    statuses = run(rec, [hand, [], [], hand, hand])
    assert statuses[1].message == "Show your hand to the camera"
    assert statuses[1].progress == pytest.approx(1 / 3)
    assert statuses[-1].phase is RecordPhase.DONE


def test_times_out_without_a_hand():
    rec = make(countdown=0.0, frames=10, timeout=1.0)
    rec.start("x_pose", 0.0)
    statuses = run(rec, [[]] * 40)
    failed = [s for s in statuses if s is not None and s.phase is RecordPhase.FAILED]
    assert len(failed) == 1
    assert "hand" in failed[0].message
    assert not rec.active


def test_cancel():
    rec = make()
    rec.start("x_pose", 0.0)
    rec.cancel()
    assert not rec.active
    assert rec.update([detection("rock")], ASPECT, 0.1) is None


def test_uses_most_confident_hand_and_majority_handedness():
    rec = make(countdown=0.0, frames=3)
    rec.start("lefty", 0.0)
    left = detection("rock", left=True)
    statuses = run(rec, [[left]] * 3)
    assert statuses[-1].template.recorded_with is Handedness.LEFT


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        make(frames=0)
