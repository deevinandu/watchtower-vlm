import threading
import time

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from watchtower_vlm.vlm_engine import VLMEngine
from watchtower_vlm.watchtower_node import WatchtowerNode

FRAME = Image(height=4, width=4, encoding='rgb8', step=12, data=bytes(48))


class ScriptedModel:
    """Returns each scripted answer once, then repeats the last."""

    def __init__(self, *answers):
        self.answers = list(answers)

    def query(self, image, prompt):
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return {'answer': answer}


class BrokenEngine(VLMEngine):
    def load(self):
        raise ImportError('no transformers')


class Harness:
    def __init__(self, engine, **params):
        overrides = [Parameter(k, value=v) for k, v in params.items()]
        self.node = WatchtowerNode(engine=engine, parameter_overrides=overrides)
        self.peer = rclpy.create_node('harness')
        self.received = []
        self.peer.create_subscription(
            Bool, '/watchtower/e_stop', lambda m: self.received.append(m.data), 10)
        self.pub = self.peer.create_publisher(Image, '/camera/image_raw', 10)
        self.reset = self.peer.create_client(Trigger, '/watchtower_vlm/reset')
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.peer)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    def feed_until(self, count, timeout=5.0):
        """Publish frames until ``count`` E-Stop messages have arrived."""
        deadline = time.time() + timeout
        while len(self.received) < count and time.time() < deadline:
            self.pub.publish(FRAME)
            time.sleep(0.05)
        return self.received[:count]

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.peer.destroy_node()


@pytest.fixture
def ros():
    rclpy.init()
    harnesses = []
    yield lambda engine, **params: harnesses.append(Harness(engine, **params)) or harnesses[-1]
    for h in harnesses:
        h.close()
    rclpy.shutdown()


def test_yes_answer_asserts_e_stop(ros):
    assert ros(VLMEngine(model=ScriptedModel('Yes, liquid.'))).feed_until(1) == [True]


def test_no_answer_keeps_e_stop_clear(ros):
    assert ros(VLMEngine(model=ScriptedModel('No.'))).feed_until(1) == [False]


def test_load_failure_is_fail_safe(ros):
    assert ros(BrokenEngine()).feed_until(1) == [True]


def test_latch_holds_until_clear_frames(ros):
    h = ros(VLMEngine(model=ScriptedModel('yes', 'no')), clear_frames=3)
    assert h.feed_until(4) == [True, True, True, False]


def test_reset_service_releases_latch(ros):
    h = ros(VLMEngine(model=ScriptedModel('yes', 'no')), clear_frames=1000)
    assert h.feed_until(2) == [True, True]
    assert h.reset.wait_for_service(timeout_sec=2.0)
    future = h.reset.call_async(Trigger.Request())
    deadline = time.time() + 2.0
    while not future.done() and time.time() < deadline:
        time.sleep(0.02)
    assert future.result().success
    h.received.clear()
    assert h.feed_until(1) == [False]


def test_watchdog_asserts_when_camera_silent(ros):
    h = ros(VLMEngine(model=ScriptedModel('no')), frame_timeout_sec=0.3)
    deadline = time.time() + 2.0
    while not h.received and time.time() < deadline:
        time.sleep(0.05)
    assert h.received[:1] == [True]


def test_rate_cap_limits_inference(ros):
    h = ros(VLMEngine(model=ScriptedModel('no')), max_inference_hz=2.0, frame_timeout_sec=0.0)
    h.feed_until(1)
    start, before = time.time(), len(h.received)
    while time.time() - start < 1.5:
        h.pub.publish(FRAME)
        time.sleep(0.02)
    assert len(h.received) - before <= 4


def collect(h, msg_type, topic):
    messages = []
    h.peer.create_subscription(msg_type, topic, messages.append, 10)
    return messages


def wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.05)
    return predicate()


def test_status_reports_answer_reason_and_latency(ros):
    import json
    from std_msgs.msg import String

    h = ros(VLMEngine(model=ScriptedModel('Yes, a puddle.')))
    statuses = collect(h, String, '/watchtower_vlm/status')
    h.feed_until(3)
    assert wait_for(lambda: statuses)
    status = json.loads(statuses[-1].data)
    assert status['e_stop'] is True
    assert status['raw_answer'] == 'Yes, a puddle.'
    assert 'puddle' in status['reason']
    assert status['latency_ms'] >= 0 and status['model_loaded'] is True


def test_diagnostics_levels(ros):
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus

    h = ros(BrokenEngine())
    diags = collect(h, DiagnosticArray, '/diagnostics')
    assert wait_for(lambda: diags, timeout=3.0)
    assert diags[-1].status[0].level == DiagnosticStatus.ERROR
    assert diags[-1].status[0].message == 'VLM failed to load'

    ok = ros(VLMEngine(model=ScriptedModel('no')))
    ok_diags = collect(ok, DiagnosticArray, '/diagnostics')
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ok.pub.publish(FRAME)
        if any(d.status[0].message == 'Monitoring' for d in ok_diags):
            break
        time.sleep(0.05)
    assert any(d.status[0].level == DiagnosticStatus.OK for d in ok_diags)
