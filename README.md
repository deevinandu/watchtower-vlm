# Watchtower VLM

ROS 2 (Humble/Jazzy) semantic safety node. It runs camera frames through a local VLM (moondream2) and publishes `std_msgs/Bool` on `/watchtower/e_stop` when it sees an anomaly such as a spill, a dropped object or a collision. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Build

```bash
pip install -r requirements.txt   # torch, transformers, ... (not rosdep-able)
colcon build --symlink-install
source install/setup.bash
```

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

Defaults live in `config/watchtower.yaml`; pass `config:=/path/to.yaml` to replace it.

## Test

```bash
cd src/watchtower_vlm && python3 -m pytest test
```
