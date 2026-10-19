# CURRENT CONTEXT & STATE
**Active Milestone:** None — M1–M5 complete.
**Last Completed Action:** M3 (`vlm_engine.py`), M4 (engine wired into node worker), M5 (parameters, `config/watchtower.yaml`, `launch/watchtower.launch.py`, README).
- `vlm_engine.py`: `VLMEngine(model_path, revision, device, ambiguous_confidence, model=None)`; lazy-imports torch/transformers in `load()`; `query()` supports moondream2 `.query()` API and legacy `encode_image/answer_question`; `evaluate()` -> `DetectionResult(anomaly, confidence, raw_answer)`; `detect()` -> bool. `parse_answer()` maps yes/no/ambiguous.
- Node: worker thread loads the model on start, converts frame -> PIL, calls `evaluate`, averages confidence over `vote_window`, compares to `confidence_threshold`. With `fail_safe` on, a load or inference failure publishes E-Stop True. A parameter callback validates the threshold and window, and rejects changes to startup-only params.
- Tests: 16 pass (queue, engine with fake model, node integration with fake engine and fail-safe). Launch smoke test verified overrides, param validation and fail-safe E-Stop.
**Current Blockers:** `torch`/`transformers` are not installed on this machine (`pip install -r requirements.txt`). Real moondream2 inference has NOT been run yet; only mocks.
**Next Immediate Step:** Install requirements, then run against a real camera or rosbag to validate prompt wording and latency. Optional later work: llama.cpp/GGUF backend, an inference-rate cap, and a launch_testing integration test.
