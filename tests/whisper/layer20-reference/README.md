# Whole-layer Whisper numerical reference

This local diagnostic separates incoming accumulated error from rounding inside
encoder layer 20. It reuses the saved eight case/feature combinations, two FP32
incoming states, original FP32 outputs, and complete float64 reference inputs.
Neither the full encoder nor the VM benchmark is replayed.

For each request, run the layer on exact promoted managed/native FP32 inputs
and the saved full-reference float64 input, using both existing float64 engines.
The fixed design has 48 fresh workers, 576 complete arrays and 13,271,040,000
new raw output bytes. Promoted model weights use an immutable hard link.
NumPy/SciPy and ORT/Python-erf references must agree at `1e-9`; cut outputs on
reference inputs must also match the retained full-model reference outputs.

The signed decomposition is `total = inherited + local`: subtract the
reference-input output from the actual FP32 output, separating the reference
change caused by the incoming FP32 state from the own-input rounding residual.
All twelve outputs and every padded frame remain included. Original `1e-4`
failures are findings, not a reason to change a tolerance. Norm ratios do not
represent causal percentages.

From the repository root, prefix each command with
`C:/Python313/python.exe -X utf8 -B`:

    -m unittest discover -s tests/whisper/layer20-reference -p test_*.py -v
    tests/whisper/layer20-reference/prepare.py
    tests/whisper/layer20-reference/run.py
    tests/whisper/layer20-reference/audit.py
    tests/whisper/layer20-reference/verify.py
    tests/whisper/layer20-reference/close.py

Commit tools before preparation. Writers refuse existing outputs. Observe the
same recorded PID/birth after a timeout; do not restart completed reference
workers. Pair and full-model bridge checks run before advancing to the next
input. Only after actual terminal success do audit, independent verification and
reporting run. Each worker uses one logical CPU, one numerical-library thread,
900 seconds, a 4 GiB sampled RSS ceiling and a 1 GiB available-memory floor.
No product implementation or runtime default is changed by this diagnostic.

The first launch was stopped before its first resource sample: Windows hidden
console creation introduced a `conhost.exe` child and violated the no-child guard.
Original source/failure records remain under the original artifact. A process-only
probe verifies detached hidden launch without that child. The corrected artifact
is `artifacts/whisper-layer20-reference-v2-20260921`; no successful call is replayed.
