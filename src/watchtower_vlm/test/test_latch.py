from watchtower_vlm.latch import EStopLatch


def test_trips_immediately_and_needs_consecutive_clean_frames():
    latch = EStopLatch(clear_frames=3)
    assert latch.update(False) is False
    assert latch.update(True) is True
    assert [latch.update(False) for _ in range(2)] == [True, True]
    assert latch.update(True) is True  # anomaly restarts the clean count
    assert [latch.update(False) for _ in range(3)] == [True, True, False]


def test_reset_releases_immediately():
    latch = EStopLatch(clear_frames=5)
    latch.trip()
    latch.reset()
    assert latch.active is False
