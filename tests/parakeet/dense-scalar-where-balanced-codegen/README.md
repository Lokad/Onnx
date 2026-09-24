# Shared timing entry and balanced warmup: native qualification

Two unscored diagnostic processes test a new consumer against identical selected
Core `672e5f30` / Data `065b7a7f`. This follows rejected control `882ac126` and
exact-consumer native diagnostic `9b4ad473` / review `fcf6bc7b`.

The 220 cases, independent constructors/oracle, deterministic batches, 600 warmup
and 180 measured intervals, output bits, metadata and ownership checks remain.
Warmup visits every case once per round for 600 rounds. Measurement keeps the
original case order. All samples enter the same non-generic `MeasureBatch`,
marked `NoInlining` only. Its timer encloses all complete public Where calls and
result-array writes. No forced collection, delay, adaptive stopping or runtime
tiering override. This is a new hypothesis, not an admitted measurement protocol.

From the repository root, run `C:/Python313/python.exe -X utf8 -B` with
`tests/parakeet/dense-scalar-where-balanced-codegen/run.py` and each command:
`prepare`, `stage`, `launch`, `observe`, and `collect` after terminal owners.
Then run `audit.py` and `review_codegen.py`. Never observe closed campaigns.

Six serial jobs check SDK, restore/build the consumer on AMD, inspect its actual
compiled timing body, and run two full diagnostic processes. The IL checker
requires only two timestamps and one complete provider call in the exact counted
loop, with `NoInlining` alone. The native gate requires whole-method Tier1
entries for that timing method, the provider and all nine typed Where methods
in both processes. A missing entry rejects this design before an untraced control.

Both processes retain all 343,200 clocks, 440 setups and 116,464,920 complete
calls, including 26,876,520 calls within measured intervals. No clocks are scored.
Only workers receive:

    DOTNET_JitDisasm=Lokad.Onnx.CPUExecutionProvider:Where Lokad.Onnx.Tensor`1:Where MeasureBatch

The separate future identical-binary control keeps all original acceptance limits.

Bounds: CPU2 for all workers/threads, CPU0 monitor; 12 GiB available / 3 GiB
tmpfs preflight, 8 GiB process-tree RSS, 1 GiB remaining memory/tmpfs,
900 seconds/job, four hours/campaign, 1 GiB job output including stdout/stderr,
2 GiB campaign artifacts. No model copies or Windows builds/inference.
