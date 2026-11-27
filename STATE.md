# CURRENT CONTEXT & STATE
**Active Milestone:** M9 (Evaluation Harness)
**Last Completed Action:** M8, observability, is done.
- `~/status` (`/watchtower_vlm/status`) is a `std_msgs/String` carrying JSON from `_status()`. It is published from `_decide()` after every decision, from `_trip()` on a trip transition only (the watchdog does not spam it), and from reset. JSON was chosen so the package can stay `ament_python`; a custom msg would need a separate `ament_cmake` interfaces package.
- `/diagnostics`: a 1 Hz timer, `_diagnostics_callback`. ERROR means the model failed or the camera is stale. WARN means the model is loading or the E-Stop is active. OK means monitoring. `hardware_id` is the model path, and the key/values mirror the status dict.
- New state is `_trip_reason`, `_last_answer`, `_last_confidence`, `_vote_mean`, `_latency_ms` (VLM call only) and `_inference_hz` (an EMA of the decision interval).
- Tests: 24 pass. Live check with the real model: status showed latency 338 ms and 3.04 Hz, and diagnostics showed WARN with the reason.
**Current Blockers:** No labelled real-world footage is in the repo yet. M9 needs a dataset; the user must supply one, or the harness accepts any folder.
**Next Immediate Step:** M9. Add a `watchtower_vlm/evaluate.py` entry point that takes a folder laid out as `anomaly/` and `normal/` (or a CSV of path,label), one or more `--prompt` values and a threshold sweep. It runs `VLMEngine.evaluate` and reports TP/FP/FN/TN, precision, recall, false-positive rate and latency per prompt. Rosbag input is optional and comes later.
