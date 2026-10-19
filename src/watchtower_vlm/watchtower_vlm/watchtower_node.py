"""Watchtower VLM ROS 2 node: camera frames in, E-Stop out."""

import queue
import threading

from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from watchtower_vlm.frame_queue import LatestFrameQueue


class WatchtowerNode(Node):
    """Subscribe to images, run anomaly inference off-thread, publish E-Stop."""

    def __init__(self):
        super().__init__('watchtower_vlm')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('e_stop_topic', '/watchtower/e_stop')
        self.declare_parameter('prompt', 'Is there a liquid spill, dropped object, or collision?')
        self.declare_parameter('queue_poll_timeout_sec', 0.5)

        image_topic = self.get_parameter('image_topic').value
        e_stop_topic = self.get_parameter('e_stop_topic').value
        self._poll_timeout = self.get_parameter('queue_poll_timeout_sec').value

        self._bridge = CvBridge()
        self._frames = LatestFrameQueue()
        self._stop_event = threading.Event()
        self._cb_group = ReentrantCallbackGroup()

        self._e_stop_pub = self.create_publisher(Bool, e_stop_topic, 10)
        self._image_sub = self.create_subscription(
            Image, image_topic, self._image_callback, qos_profile_sensor_data,
            callback_group=self._cb_group)

        self._worker = threading.Thread(
            target=self._inference_loop, name='watchtower_inference', daemon=True)
        self._worker.start()

        self.get_logger().info(
            f'Watchtower listening on "{image_topic}", publishing E-Stop on "{e_stop_topic}"')

    def _image_callback(self, msg: Image):
        """Non-blocking: hand the newest frame to the worker and return."""
        self._frames.put_latest(msg)

    def _inference_loop(self):
        while not self._stop_event.is_set():
            try:
                msg = self._frames.get(timeout=self._poll_timeout)
            except queue.Empty:
                continue

            try:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            except CvBridgeError as exc:
                self.get_logger().warning(f'Failed to convert image: {exc}')
                continue

            prompt = self.get_parameter('prompt').value
            try:
                anomaly = self._detect_anomaly(frame, prompt)
            except Exception as exc:  # noqa: BLE001 - keep the safety loop alive
                self.get_logger().error(f'Inference failed: {exc}')
                continue

            self._publish_e_stop(anomaly)

    def _detect_anomaly(self, frame, prompt: str) -> bool:
        """Placeholder until the VLM engine is wired in (M3/M4)."""
        return False

    def _publish_e_stop(self, anomaly: bool):
        self._e_stop_pub.publish(Bool(data=bool(anomaly)))
        if anomaly:
            self.get_logger().warning('Anomaly detected: E-Stop asserted')

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
