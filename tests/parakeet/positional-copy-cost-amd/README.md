# Exact Parakeet positional-copy cost

This diagnostic uses the unchanged qualified Core f95a13c5. It compares the
ordinary `TensorSlice<float>.ToDenseTensor` conversion with the already existing
contiguous-copy helper reached through same-shape `Reshape`. It does not change
product code, multiplication, packing or the graph.

Use the actual Constant_1723 bytes and all twenty original clip lengths. Four
fresh AMD CPU2 processes run generic/helper/helper/generic. Each performs one
warmup and three measured passes, 24 owned conversions per clip: 7,680 total
calls and 5,760 measured clocks. Allocation is included; reference checks and
IO are outside timing. Every output byte, independent storage, original input
and held outputs are checked. Preserve all clocks and failures.

Run with Python 3.13 `-X utf8 -B` from the repository root: `run.py prepare`,
`stage`, `launch build`, `observe build`, `collect build`, `review_build.py`,
then `launch capture`, observe the same owner, collect only after terminal,
and run `audit.py`. Freeze tools before preparation. A finished worker is never
repeated to fix analysis or to improve a result. All .NET work runs on the VM.

Fix corpus max/min <=1.10 and per-clip <=1.20 before execution. This isolates
copy cost in a component; a useful difference is not an application speedup.
The full application still requires its existing independent correctness,
repeatability and >=3% gain gates. The copy-only envelope is 2 GiB available
RAM/1 GiB tmpfs preflight, 1 GiB RSS, 180 seconds per job, 1 GiB free RAM/tmpfs
during execution and 128 MiB campaign output. No whole-model inference runs.
