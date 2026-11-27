# Watchtower VLM: Semantic Safety Node
**Goal:** A ROS 2 (Humble/Jazzy) package that subscribes to camera feeds, processes frames through a local, quantized Vision-Language Model (VLM), and publishes an Emergency Stop (E-Stop) signal if semantic anomalies (spills, drops, collisions) are detected.

## Tech Stack
- **Framework:** ROS 2 (Python `rclpy`)
- **Vision Model:** `moondream2` (via Hugging Face `transformers`) or `llama.cpp` for quantized inference.
- **Image Processing:** OpenCV (`cv2`) and `cv_bridge`
- **Concurrency:** ROS 2 `MultiThreadedExecutor` and thread-safe queues.

## Architecture Guidelines
- **Zero-Blocking Callbacks:** VLM inference takes 500ms+. It CANNOT run synchronously inside the `sensor_msgs/Image` callback. The callback must place the latest frame into a `queue.Queue(maxsize=1)` (dropping older frames) while a separate background thread handles inference.
- **Dynamic Prompts:** The prompt (e.g., "Is liquid spilling?") should be configurable via ROS 2 Parameters.

## Sprints & Milestones
- [x] **M1: ROS 2 Package Skeleton**
      Create the `watchtower_vlm` ROS 2 package (`ament_python`), setup `setup.py`, and create the basic Node class structure.
- [x] **M2: Threaded Subscriber & Publisher**
      Implement the image subscriber (`/camera/image_raw`), the E-Stop publisher (`/watchtower/e_stop` as `std_msgs/Bool`), and the MultiThreadedExecutor with the frame-dropping Queue.
- [x] **M3: The VLM Inference Engine**
      Write an isolated Python class that loads a small VLM (e.g., Moondream2, tiny parameter size) and accepts a PIL image and text prompt to return a boolean anomaly detection.
- [x] **M4: Node Integration**
      Connect the VLM engine to the background worker thread. Parse the VLM text output (e.g., mapping "Yes" to `True`) and publish to the E-Stop topic.
- [x] **M5: Parameters & Launch File**
      Expose the VLM prompt, confidence threshold, and model path as ROS 2 Parameters. Write a `watchtower.launch.py` file to start it.
- [x] **M6: Real Model Bring-up**
      Project-local `.venv` (with `--system-site-packages` so rclpy/cv_bridge stay visible), install torch (CUDA 12.8+ wheels for Blackwell GPUs) + transformers, and add a benchmark script that runs moondream2 on a sample image and reports load time, per-frame latency, and the raw answer.
- [x] **M7: Safety Hardening**
      Latched E-Stop (clears only after N clean frames or a `~/reset` service call), inference-rate cap, and a camera watchdog that asserts E-Stop if no frames arrive within a timeout.
- [x] **M8: Observability**
      Publish a status topic (raw answer, confidence, latency) and `diagnostic_msgs` diagnostics so the stop reason is visible in RViz/rqt.
- [x] **M9: Evaluation Harness**
      Replay a labelled image folder or rosbag through the engine and report false-positive / miss rates per prompt and threshold.
- [x] **M10: Performance & Coverage**
      Optional `llama.cpp` GGUF backend for CPU/Jetson, and multiple prompts per frame (spill, fall, collision) combined into one decision.
