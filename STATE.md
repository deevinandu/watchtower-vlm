# CURRENT CONTEXT & STATE
**Active Milestone:** M11 (Real-World Dataset). It needs data from the user; M13 and M14 can proceed without it.
**Last Completed Action:** M12, the quantized GGUF, is done.
- `watchtower_vlm/quantize.py` (`ros2 run watchtower_vlm quantize --type Q4_K_M ...`) calls `llama_cpp.llama_model_quantize` through ctypes; no llama.cpp checkout is needed. The source is resolved with `llamacpp_backend.resolve_file` (repo glob at revision, or a local path), and output goes to `models/moondream2-text-model-<TYPE>.gguf` (gitignored). Existing outputs are skipped unless `--force`. Only the text model is quantized; mmproj stays f16.
- The launch file now exposes `gguf_file` (default `"''"`, quoted so launch_ros YAML-parses it to an empty string) and `n_gpu_layers`.
- Results, all on the 12-image synthetic set:
  - CPU only (24 cores): f16 1729 ms, Q8_0 1433 ms, Q5_K_M 806 ms, Q4_K_M 856 ms, Q4_0 808 ms.
  - GPU: about 150 ms for every type, so the text model is not the GPU bottleneck.
  - Answer drift vs f16 over 36 answers: Q8_0 0, Q4_K_M 0, Q5_K_M 1, Q4_0 3.
  - Recommended: Q4_K_M. The earlier "5.2 s f16 CPU" figure was a noisy 3-run sample; 1.7 s is consistent.
- Tests: 39 pass (4 new in `test_quantize.py`, using a ctypes `Params` struct to fake `llama_cpp`).
**Current Blockers:** M11 needs real labelled frames from the user. No Jetson available to measure edge latency.
**Next Immediate Step:** M13. Add rosbag input to `evaluate` (`rosbag2_py` reader, a label per bag or a labelled time ranges file) that replays frames through the vote/latch path, so `vote_window` and `clear_frames` can be swept. M14 (launch_testing) is independent and can be done anytime.
