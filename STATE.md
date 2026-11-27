# CURRENT CONTEXT & STATE
**Active Milestone:** M10 (Performance & Coverage)
**Last Completed Action:** M9, the evaluation harness, is done.
- `watchtower_vlm/evaluate.py` (`ros2 run watchtower_vlm evaluate <dataset>`): `load_dataset()` accepts an anomaly/ + normal/ folder or a `path,label` CSV with relative paths resolved against it. `run()` returns `Prediction` items. `summarise()` gives TP/FP/FN/TN, precision, recall, FPR, accuracy, the ambiguous count and latency mean/p95, using NaN when a denominator is 0. `print_report()` gives a table per prompt, and `--csv` writes per-image rows. `percentile` is reused from `benchmark.py`.
- No threshold sweep: per-frame confidence is discrete (0, 1 or ambiguous), so a sweep needs vote_window over video, i.e. rosbag input, which is not yet built.
- Tests: 29 pass (5 new in `test_evaluate.py`, using a colour-keyed fake model).
- Live smoke run on 12 synthetic images (drawn blobs on tiled floors, generated only in the scratchpad): moondream2 found 0/6 and 1/6 anomalies on two prompts, with 0 false positives and about 330 ms each. The model is conservative, and synthetic blobs are not realistic.
**Current Blockers:** No real labelled dataset. Accuracy numbers mean nothing until the user supplies one, from robot camera frames or a public spill/fall dataset.
**Next Immediate Step:** M10. (a) Support multiple prompts per frame: a `prompts` string-array param, OR-combined, with the triggering prompt shown in reason/status. (b) Add an optional `llama.cpp` GGUF backend behind a `backend` param (`transformers` | `llamacpp`) via `llama-cpp-python`, keeping the `VLMEngine` interface (`load`/`loaded`/`evaluate`/`model_path`). Do (a) first; it is cheap and independent.
