"""Watchtower VLM ROS 2 node: camera frames in, E-Stop out."""

from collections import deque
import queue
import threading

from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from watchtower_vlm.frame_queue import LatestFrameQueue
from watchtower_vlm.vlm_engine import DEFAULT_MODEL_ID, DEFAULT_REVISION, VLMEngine


class WatchtowerNode(Node):
    """Subscribe to images, run VLM anomaly inference off-thread, publish E-Stop."""

    def __init__(self, engine=None):
        super().__init__('watchtower_vlm')

        def desc(text):
            return ParameterDescriptor(description=text)

        self.declare_parameter('image_topic', '/camera/image_raw', desc('Input camera topic.'))
        self.declare_parameter('e_stop_topic', '/watchtower/e_stop', desc('E-Stop output topic.'))
        self.declare_parameter(
            'prompt', 'Is there a liquid spill, a dropped object, or a collision?',
            desc('Yes/no question asked of the VLM for every frame.'))
        self.declare_parameter(
            'confidence_threshold', 0.5,
            desc('E-Stop fires when mean anomaly confidence over vote_window >= this.'))
        self.declare_parameter(
            'vote_window', 1, desc('Number of recent inferences averaged before deciding.'))
        self.declare_parameter(
            'ambiguous_confidence', 0.5,
            desc('Confidence assigned to answers that are neither yes nor no.'))
        self.declare_parameter('model_path', DEFAULT_MODEL_ID,
                               desc('Hugging Face model id or local directory.'))
        self.declare_parameter('model_revision', DEFAULT_REVISION,
                               desc('Model revision/tag; empty for default.'))
        self.declare_parameter('device', 'auto', desc('auto, cpu, or cuda[:N].'))
        self.declare_parameter(
            'fail_safe', True,
            desc('Assert E-Stop for every frame if the model fails to load or infer.'))
        self.declare_parameter('queue_poll_timeout_sec', 0.5)

        image_topic = self.get_parameter('image_topic').value
        e_stop_topic = self.get_parameter('e_stop_topic').value
        self._poll_timeout = self.get_parameter('queue_poll_timeout_sec').value
        self._votes = deque(maxlen=max(1, self.get_parameter('vote_window').value))

        self._engine = engine or VLMEngine(
            model_path=self.get_parameter('model_path').value,
            revision=self.get_parameter('model_revision').value,
            device=self.get_parameter('device').value,
            ambiguous_confidence=self.get_parameter('ambiguous_confidence').value)
        self._engine_failed = False

        self._bridge = CvBridge()
        self._frames = LatestFrameQueue()
        self._stop_event = threading.Event()
        self._cb_group = ReentrantCallbackGroup()

        self._e_stop_pub = self.create_publisher(Bool, e_stop_topic, 10)
        self._image_sub = self.create_subscription(
            Image, image_topic, self._image_callback, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._worker = threading.Thread(
            target=self._inference_loop, name='watchtower_inference', daemon=True)
        self._worker.start()

        self.get_logger().info(
            f'Watchtower listening on "{image_topic}", publishing E-Stop on "{e_stop_topic}"')

    def _on_set_parameters(self, params):
        for param in params:
            if param.name == 'confidence_threshold' and not 0.0 <= param.value <= 1.0:
                return SetParametersResult(
                    successful=False, reason='confidence_threshold must be in [0, 1]')
            if param.name == 'vote_window':
                if param.value < 1:
                    return SetParametersResult(successful=False, reason='vote_window must be >= 1')
                self._votes = deque(self._votes, maxlen=param.value)
            if param.name in ('model_path', 'model_revision', 'device', 'image_topic',
                              'e_stop_topic', 'ambiguous_confidence'):
                return SetParametersResult(
                    successful=False, reason=f'{param.name} can only be set at startup')
        return SetParametersResult(successful=True)

    def _image_callback(self, msg: Image):
        """Non-blocking: hand the newest frame to the worker and return."""
        self._frames.put_latest(msg)

    def _load_engine(self):
        if self._engine.loaded:
            return
        self.get_logger().info(f'Loading VLM "{self._engine.model_path}" (this may take a while)')
        try:
            self._engine.load()
        except Exception as exc:  # noqa: BLE001 - report any import/download/OOM failure
            self._engine_failed = True
            self.get_logger().error(f'Failed to load VLM: {exc}')
            return
        self.get_logger().info('VLM loaded')

    def _inference_loop(self):
        self._load_engine()
        while not self._stop_event.is_set():
            try:
                msg = self._frames.get(timeout=self._poll_timeout)
            except queue.Empty:
                continue

            if self._engine_failed:
                self._handle_failure('VLM unavailable')
                continue

            try:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            except CvBridgeError as exc:
                self.get_logger().warning(f'Failed to convert image: {exc}')
                continue

            prompt = self.get_parameter('prompt').value
            try:
                result = self._engine.evaluate(PILImage.fromarray(frame), prompt)
            except Exception as exc:  # noqa: BLE001 - keep the safety loop alive
                self._handle_failure(f'Inference failed: {exc}')
                continue

            self.get_logger().debug(
                f'VLM answer "{result.raw_answer}" -> confidence {result.confidence:.2f}')
            self._publish_e_stop(self._vote(result.confidence))

    def _vote(self, confidence: float) -> bool:
        self._votes.append(confidence)
        threshold = self.get_parameter('confidence_threshold').value
        return sum(self._votes) / len(self._votes) >= threshold

    def _handle_failure(self, reason: str):
        fail_safe = self.get_parameter('fail_safe').value
        self.get_logger().error(
            f'{reason}; {"asserting" if fail_safe else "not asserting"} E-Stop (fail_safe)',
            throttle_duration_sec=5.0)
        if fail_safe:
            self._publish_e_stop(True)

    def _publish_e_stop(self, anomaly: bool):
        self._e_stop_pub.publish(Bool(data=bool(anomaly)))
        if anomaly:
            self.get_logger().warning('E-Stop asserted', throttle_duration_sec=1.0)

    def destroy_node(self):
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._frames.dropped:
            self.get_logger().info(f'Dropped {self._frames.dropped} stale frames')
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WatchtowerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
