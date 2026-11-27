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
ros2 launch watchtower_vlm watchtower.launch.py prompt:="Is liquid spilling?" device:=cuda
```

| Parameter | Default | Notes |
|---|---|---|
| `image_topic` | `/camera/image_raw` | startup only |
| `e_stop_topic` | `/watchtower/e_stop` | startup only |
| `prompt` | spill/drop/collision question | runtime-settable; a yes/no suffix is appended |
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

Defaults live in `config/watchtower.yaml`; pass `config:=/path/to.yaml` to replace it.

## Test

```bash
cd src/watchtower_vlm && python3 -m pytest test
```
