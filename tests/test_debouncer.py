import pytest

from memecam.gestures.debouncer import GestureDebouncer

FPS = 30
DT = 1 / FPS


def feed(deb: GestureDebouncer, gestures: list[str | None], start: float = 0.0) -> list[str | None]:
    """Feed one gesture per frame at 30 fps; return what fired on each frame."""
    return [deb.update(g, start + i * DT) for i, g in enumerate(gestures)]


def test_fires_on_the_nth_consecutive_frame():
    deb = GestureDebouncer(hold_frames=3, cooldown_seconds=0)
    assert feed(deb, ["Thumb_Up"] * 3) == [None, None, "Thumb_Up"]


def test_hold_frames_of_one_fires_immediately():
    deb = GestureDebouncer(hold_frames=1, cooldown_seconds=0)
    assert deb.update("Victory", 0.0) == "Victory"


def test_interruption_resets_the_streak():
    deb = GestureDebouncer(hold_frames=3, cooldown_seconds=0)
    out = feed(deb, ["Thumb_Up", "Thumb_Up", None, "Thumb_Up", "Thumb_Up", "Thumb_Up"])
    assert out == [None, None, None, None, None, "Thumb_Up"]


def test_switching_gesture_resets_the_streak():
    deb = GestureDebouncer(hold_frames=3, cooldown_seconds=0)
    out = feed(deb, ["Thumb_Up", "Thumb_Up", "Victory", "Victory", "Victory"])
    assert out == [None, None, None, None, "Victory"]


def test_continuous_hold_fires_only_once():
    deb = GestureDebouncer(hold_frames=2, cooldown_seconds=0)
    out = feed(deb, ["Thumb_Up"] * 120)  # 4 seconds of holding
    assert out.count("Thumb_Up") == 1


def test_release_and_repeat_fires_again_after_cooldown():
    deb = GestureDebouncer(hold_frames=2, cooldown_seconds=0.5)
    assert feed(deb, ["Open_Palm", "Open_Palm"], start=0.0)[-1] == "Open_Palm"
    deb.update(None, 0.1)
    # Re-held well after the cooldown has expired.
    assert feed(deb, ["Open_Palm", "Open_Palm"], start=1.0)[-1] == "Open_Palm"


def test_cooldown_blocks_other_gestures():
    deb = GestureDebouncer(hold_frames=2, cooldown_seconds=1.0)
    assert feed(deb, ["Thumb_Up", "Thumb_Up"], start=0.0)[-1] == "Thumb_Up"
    # Victory completes its hold at t≈0.23s, still inside the 1s cooldown.
    assert feed(deb, ["Victory"] * 5, start=0.1) == [None] * 5
    assert deb.in_cooldown(0.5)


def test_gesture_held_through_cooldown_fires_when_it_ends():
    deb = GestureDebouncer(hold_frames=2, cooldown_seconds=1.0)
    feed(deb, ["Thumb_Up", "Thumb_Up"], start=0.0)  # fires at t=DT; cooldown until ~1.03
    out = [deb.update("Victory", t) for t in (0.2, 0.3, 0.9, 1.1, 1.2)]
    assert out == [None, None, None, "Victory", None]


def test_progress_and_candidate():
    deb = GestureDebouncer(hold_frames=4, cooldown_seconds=0)
    assert deb.candidate is None and deb.progress == 0.0
    feed(deb, ["ILoveYou"] * 2)
    assert deb.candidate == "ILoveYou"
    assert deb.progress == pytest.approx(0.5)
    feed(deb, ["ILoveYou"] * 10, start=1.0)
    assert deb.progress == 1.0


def test_reset_clears_cooldown_and_streak():
    deb = GestureDebouncer(hold_frames=2, cooldown_seconds=10)
    feed(deb, ["Thumb_Up", "Thumb_Up"])
    deb.reset()
    assert not deb.in_cooldown(0.1)
    assert feed(deb, ["Thumb_Up", "Thumb_Up"], start=0.1)[-1] == "Thumb_Up"


@pytest.mark.parametrize(("hold", "cooldown"), [(0, 1.0), (-1, 1.0), (3, -0.1)])
def test_rejects_invalid_parameters(hold, cooldown):
    with pytest.raises(ValueError):
        GestureDebouncer(hold_frames=hold, cooldown_seconds=cooldown)
