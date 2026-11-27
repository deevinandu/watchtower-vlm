# CURRENT CONTEXT & STATE
**Active Milestone:** M8 (Observability)
**Last Completed Action:** M7, safety hardening, is done.
- `watchtower_vlm/latch.py`: `EStopLatch(clear_frames)` with `update(anomaly)`, `trip()` and `reset()`. It is not thread-safe; the node guards it with `_state_lock`.
- Node: `_decide()` feeds the vote result into the latch and publishes the latched state. `_trip(reason)` is used by fail-safe failures and the watchdog. Transitions are logged once, not per frame.
- `~/reset` (`std_srvs/Trigger`, i.e. `/watchtower_vlm/reset`) clears the latch and the vote history. It is refused when the engine failed to load.
- Watchdog: a 0.25 s timer trips if no frame arrives within `frame_timeout_sec` (default 2.0, 0 disables). The clock starts at node start, so it fails safe before the first frame.
- `max_inference_hz` (default 0 = uncapped): the worker waits on `_stop_event` between inferences.
- The worker body was split into `_process(msg)`. `WatchtowerNode(engine, **kwargs)` forwards `parameter_overrides` for tests.
- Tests: 22 pass, covering the latch, the reset service, the watchdog and the rate cap. A live real-model check confirmed watchdog, latch, reset and re-trip.
**Current Blockers:** None.
**Next Immediate Step:** M8. Publish `~/status` with the raw answer, confidence, latency, latch state and trip reason. Use a custom msg, or a JSON `std_msgs/String` to stay `ament_python`-only. Also publish `/diagnostics` (`diagnostic_msgs/DiagnosticArray`) with OK/WARN/ERROR covering model loaded, camera alive and E-Stop active.
