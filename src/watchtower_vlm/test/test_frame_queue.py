import threading

from watchtower_vlm.frame_queue import LatestFrameQueue


def test_keeps_only_latest_frame():
    q = LatestFrameQueue()
    for i in range(5):
        q.put_latest(i)
    assert q.get(timeout=0.1) == 4
    assert q.dropped == 4


def test_concurrent_producers_never_block():
    q = LatestFrameQueue()
    threads = [threading.Thread(target=lambda: [q.put_latest(n) for n in range(200)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive()
    assert q.get(timeout=0.1) is not None
