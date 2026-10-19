import threading
import time

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from watchtower_vlm.vlm_engine import VLMEngine
from watchtower_vlm.watchtower_node import WatchtowerNode


class FakeModel:
    def __init__(self, answer):
        self.answer = answer

    def query(self, image, prompt):
        return {'answer': self.answer}


class BrokenEngine(VLMEngine):
    def load(self):
        raise ImportError('no transformers')


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def run_once(engine):
    node = WatchtowerNode(engine=engine)
    harness = rclpy.create_node('harness')
    received = []
    harness.create_subscription(Bool, '/watchtower/e_stop', lambda m: received.append(m.data), 10)
    pub = harness.create_publisher(Image, '/camera/image_raw', 10)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(harness)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    img = Image(height=4, width=4, encoding='rgb8', step=12, data=bytes(48))
    deadline = time.time() + 5.0
    while not received and time.time() < deadline:
        pub.publish(img)
        time.sleep(0.1)

    executor.shutdown()
    node.destroy_node()
    harness.destroy_node()
    return received


def test_yes_answer_asserts_e_stop(ros):
    assert run_once(VLMEngine(model=FakeModel('Yes, liquid on the floor.')))[0] is True


def test_no_answer_clears_e_stop(ros):
    assert run_once(VLMEngine(model=FakeModel('No.')))[0] is False


def test_load_failure_is_fail_safe(ros):
    assert run_once(BrokenEngine())[0] is True
