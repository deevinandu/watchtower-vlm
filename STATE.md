# CURRENT CONTEXT & STATE
**Active Milestone:** M4
**Last Completed Action:** M1 + M2 done. `src/watchtower_vlm` (ament_python) builds with `colcon build --symlink-install`.
- `watchtower_vlm/frame_queue.py`: `LatestFrameQueue` wraps `queue.Queue(maxsize=1)`, drops stale frames (counts in `.dropped`).
- `watchtower_vlm/watchtower_node.py`: `WatchtowerNode` subscribes to `image_topic` (default `/camera/image_raw`, sensor-data QoS, ReentrantCallbackGroup), callback only enqueues. Daemon worker thread `_inference_loop` converts via cv_bridge (rgb8), calls `_detect_anomaly(frame, prompt)` (currently a stub returning False), publishes `std_msgs/Bool` on `e_stop_topic` (default `/watchtower/e_stop`). `main()` uses `MultiThreadedExecutor`.
- Params declared: `image_topic`, `e_stop_topic`, `prompt` (read each frame, so runtime-settable), `queue_poll_timeout_sec`.
- Tests: `cd src/watchtower_vlm && python3 -m pytest test` (after sourcing install/setup.bash) — 2 pass. Live smoke test verified E-Stop publishes `false`.
**Current Blockers:** No git remote configured (`git push origin main` fails until `git remote add origin <url>`). Python deps for M3 (`transformers`, `torch`, `Pillow`, moondream2 weights) not yet checked/installed.
**Next Immediate Step:** M3 — create `watchtower_vlm/vlm_engine.py` with a `VLMEngine` class (load moondream2, `detect(pil_image, prompt) -> bool`), independent of ROS, plus a unit test using a mocked model.
