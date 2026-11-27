# Watchtower VLM

ROS 2 (Humble/Jazzy) semantic safety node. It runs camera frames through a local VLM (moondream2) and publishes `std_msgs/Bool` on `/watchtower/e_stop` when it sees an anomaly such as a spill, a dropped object or a collision. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Build

The VLM deps are not rosdep-able, so they live in a project venv that can still see ROS:

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv && touch .venv/COLCON_IGNORE
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
source .venv/bin/activate
python -m colcon build --symlink-install   # build with the venv python so entry points use it
source install/setup.bash
```

## Benchmark

```bash
ros2 run watchtower_vlm benchmark --image frame.jpg --prompt "Is liquid spilling?" --runs 10
```

Measured on an RTX 5070 Ti Laptop GPU (fp16): load 3.2 s with weights cached (74 s on first download), ~320 ms/frame (≈3.1 Hz), 4.5 GiB peak VRAM. The node outputs E-Stop at ≈2.3 Hz from a 30 Hz camera.

## Run

```bash
ros2 launch watchtower_vlm watchtower.launch.py device:=cuda \
  prompts:="['Is liquid spilling on the floor?', 'Has an object fallen over?']"
```

| Parameter | Default | Notes |
|---|---|---|
| `image_topic` | `/camera/image_raw` | startup only |
| `e_stop_topic` | `/watchtower/e_stop` | startup only |
| `prompts` | `[spill/drop/collision question]` | runtime-settable string array; each gets a yes/no suffix, and any "yes" counts as an anomaly. The image is encoded once per frame, so each extra prompt costs about 30 ms, not about 320 ms |
| `confidence_threshold` | `0.5` | runtime-settable, `[0,1]`; E-Stop when mean confidence over `vote_window` ≥ threshold |
| `vote_window` | `1` | runtime-settable; >1 debounces single-frame false positives |
| `ambiguous_confidence` | `0.5` | confidence given to answers that are neither yes nor no |
| `model_path` / `model_revision` | `vikhyatk/moondream2` / `2025-01-09` | HF id or local dir |
| `device` | `auto` | `auto`, `cpu`, `cuda[:N]` |
| `fail_safe` | `true` | assert E-Stop on every frame if the model fails to load or infer |
| `clear_frames` | `3` | runtime-settable; consecutive clean decisions needed to release a latched E-Stop |
| `max_inference_hz` | `0.0` | runtime-settable; caps inference rate (0 = as fast as the model runs) |
| `frame_timeout_sec` | `2.0` | runtime-settable; assert E-Stop if the camera is silent this long (0 = off) |

The E-Stop **latches**: one anomaly, inference failure or camera timeout asserts it, and it releases only after `clear_frames` consecutive clean decisions or a manual reset:

```bash
ros2 service call /watchtower_vlm/reset std_srvs/srv/Trigger
```

A reset is refused while the VLM is unavailable. If the camera is still silent, the watchdog re-asserts the stop within 0.25 s.

### Monitoring

- `/watchtower_vlm/status` (`std_msgs/String`, JSON): published after every decision and every trip. It carries `e_stop`, `reason`, `prompts`, `answers` (per prompt), `trigger_prompt`, `raw_answer`, `confidence`, `vote_mean`, `latency_ms`, `inference_hz`, `camera_age_s`, `model_loaded`, `model_failed` and `dropped_frames`.
- `/diagnostics` (`diagnostic_msgs/DiagnosticArray`, 1 Hz):
  - ERROR when the VLM failed to load or the camera is silent;
  - WARN while the model is loading or the E-Stop is active (the message gives the reason);
  - otherwise OK.

  View it with `ros2 run rqt_runtime_monitor rqt_runtime_monitor`.

```bash
ros2 topic echo /watchtower_vlm/status --field data
```

Defaults live in `config/watchtower.yaml`; pass `config:=/path/to.yaml` to replace it.

## Backends

| | `transformers` (default) | `llamacpp` |
|---|---|---|
| Weights | moondream2 `2025-01-09`, fp16 | the same revision's `moondream2-*-f16.gguf` files |
| GPU latency, 1 prompt | ~320 ms | ~160 ms |
| Extra prompts | ~30 ms each (image encoded once) | ~180 ms each (image re-encoded) |
| CPU only (24 cores) | not measured | ~1.7 s/frame f16, ~0.8 s Q4_K_M (see below) |
| Install | `requirements.txt` | build `llama-cpp-python` (see `requirements.txt`) |

All figures are from an RTX 5070 Ti Laptop GPU.

```bash
ros2 launch watchtower_vlm watchtower.launch.py backend:=llamacpp
```

For llama.cpp, `gguf_file` and `mmproj_file` accept local paths, for example a quantized GGUF you made with `llama-quantize`. They also accept filename globs, which are resolved in `model_path` at `model_revision`; the defaults are `*text-model*` and `*mmproj*`. Set `n_gpu_layers:=0` for CPU only.

### Quantized models

```bash
ros2 run watchtower_vlm quantize --type Q4_K_M        # also: Q4_0 Q5_K_M Q8_0
ros2 launch watchtower_vlm watchtower.launch.py backend:=llamacpp \
  gguf_file:=$PWD/models/moondream2-text-model-Q4_K_M.gguf n_gpu_layers:=0
```

This quantizes only the text model; the vision projector stays f16. The tool uses llama.cpp's `llama_model_quantize` through `llama-cpp-python`, so no separate llama.cpp checkout is needed. Outputs go to `models/`, which is gitignored.

| Type | Text model | CPU only | GPU | Answers changed vs f16 (36) |
|---|---|---|---|---|
| f16 | 2708 MiB | 1729 ms | 159 ms | — |
| Q8_0 | 1440 MiB | 1433 ms | 156 ms | 0 |
| Q5_K_M | 991 MiB | 806 ms | 157 ms | 1 |
| **Q4_K_M** | **877 MiB** | **856 ms** | **155 ms** | **0** |
| Q4_0 | 790 MiB | 808 ms | 147 ms | 3 |

Q4_K_M is the recommended edge setting. On the GPU quantization barely matters, because the text model isn't the bottleneck there. The drift counts come from the 12-image synthetic set only. They have not been measured on a Jetson.

The backends **do not give identical answers** on the same image. On the synthetic set, llama.cpp caught 3/6 puddles and transformers caught 1/6, with no false positives from either. Re-run `evaluate` with `--backend` before switching.

## Evaluate prompts

Put labelled images in `anomaly/` and `normal/` subfolders, or list them in a CSV with `path,label` columns. Then compare prompts:

```bash
ros2 run watchtower_vlm evaluate dataset/ \
  --prompt "Is there a liquid spill on the floor?" \
  --prompt "Is there a puddle or stain on the floor?" \
  --csv results.csv
```

With more than one prompt, an extra **ANY** row shows the OR-combined result, which is what the node acts on. The report gives one row per prompt with TP/FP/FN/TN, precision, recall, false-positive rate, the number of ambiguous answers and mean latency. `--csv` writes per-image answers so you can inspect individual misses. The model answers yes/no, so each frame's confidence is only 0, 1 or `--ambiguous-confidence`. A threshold sweep only becomes meaningful with `vote_window > 1` on video.

## Test

```bash
cd src/watchtower_vlm && python3 -m pytest test
```
