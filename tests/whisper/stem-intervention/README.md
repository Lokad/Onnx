# Whisper stem-state intervention

This local diagnostic replaces `/Add_2_output_0`, the state after the first 14
original encoder nodes, with its existing NumPy/SciPy float64 reference rounded
once to float32. It executes the remaining 1,545 nodes with the unchanged float32
or wide-MatMul interpreter from `../matmul-precision`. The original full-graph
controls are reused without repeating inference.

The fixed inputs are the same three previously selected natural recordings and
first-case repeat, using managed features. Each fresh worker captures all 36
existing suffix boundaries, including every padded frame. Both original float64
reference routes and original managed/native FP32 outputs remain comparison
targets. This state replacement is not a managed implementation or a new model
qualification; it cannot attribute an effect to one operation within the stem.

From the repository root, with the already installed Python3.13 numerical
packages bound by `tests/whisper/full-reference/common.py`:

    C:/Python313/python.exe -X utf8 -B tests/whisper/stem-intervention/test_suffix.py
    C:/Python313/python.exe -X utf8 -B tests/whisper/stem-intervention/probe.py prepare
    C:/Python313/python.exe -X utf8 -B -u tests/whisper/stem-intervention/run.py
    C:/Python313/python.exe -X utf8 -B tests/whisper/stem-intervention/audit.py

All evidence writers are single-use and refuse existing output. The supervisor
is a byte-identical copy of the validated matrix-precision supervisor; its new
location binds this directory's protocol and worker. It inherits CPU2 before
Python startup, uses one numerical-library thread, and retains the same time,
memory and disk limits. No VM inference or new native reference is performed.

For each suffix mode independently, the mechanism screen requires at least
halving final maximum reference error for every selected case, without increasing
failed-value counts, against both independent references. Full-array acceptance
still uses `1e-4` and all intermediate failures remain visible. A passing screen
only nominates stem precision for implementation work; it does not promote a
production kernel or change a benchmark.
