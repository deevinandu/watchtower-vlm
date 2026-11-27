# CURRENT CONTEXT & STATE
**Active Milestone:** M7 (Safety Hardening)
**Last Completed Action:** M6 is done, and real moondream2 inference is verified on the GPU.
- `.venv` (`--system-site-packages`, with `COLCON_IGNORE`): torch 2.14.1+cu130, transformers 4.57.6, numpy 1.22 (must stay below 2 for cv_bridge), setuptools 58.2.0 (newer versions break the ament_python build), Pillow 10 or later (9.0 lacks `Image.Resampling`), and pyvips[binary] (required by moondream2).
- Build with the venv Python, `python -m colcon build --symlink-install`, so entry-point shebangs point at the venv.
- `watchtower_vlm/benchmark.py` (`ros2 run watchtower_vlm benchmark`) reports load time, latency and VRAM. If no `--image` is given it uses a synthetic puddle image.
- Results on RTX 5070 Ti: load 3.2 s with cached weights, about 320 ms per frame (3.1 Hz), 4.5 GiB VRAM. Both the yes and no paths work. The end-to-end launch publishes E-Stop at 2.3 Hz and correctly asserts it.
- `VLMEngine.resolved_device` exposes the device actually chosen when `device` is `auto`.
- Tests: 16 pass with the venv.
**Current Blockers:** None. No real camera footage has been used yet; only the synthetic image.
**Next Immediate Step:** M7. Latch the E-Stop so it clears only after `clear_frames` consecutive clean frames or a `~/reset` (`std_srvs/Trigger`) call. Add a `max_inference_hz` cap, plus a camera watchdog timer that asserts E-Stop when no frame arrives within `frame_timeout_sec`.
