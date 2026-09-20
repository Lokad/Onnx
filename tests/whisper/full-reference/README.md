# Complete Whisper encoder references

Compare every retained twenty-clip input, both feature sources and the first-case
repeat with two float64 implementations. There are 84 reference calls and 3,444
complete boundary arrays, totaling 55,480,320,000 bytes. The original FP32 model
results are reused, with no managed/native FP32 inference replay. This is a
numerical diagnostic, not a latency benchmark or a changed tolerance.

The NumPy/SciPy route directly interprets the original thirteen operator types.
The other route runs float64 ORT sections with optimizations disabled and one
thread. Convolution is lowered explicitly into padded windows and matrix
multiplication. ORT 1.29 has no CPU double Erf kernel, so Python `math.erf`
connects its sections; NumPy uses SciPy's double Erf. The installed ONNX1.22
reference evaluator also rounds Erf through float32 (`otypes=['f']`), so small
reference tests explicitly override that operation with double `math.erf`.
The original failed tests and their source files remain in
`artifacts/whisper-full-reference-tooling-20260920`.

All original float32 weights and scalar constants are promoted exactly, with
no replacement by more precise constants. Graph partitioning preserves every
node and live tensor dependency. Each route retains the same 41 boundaries,
including every padded frame. The first input must pass the independent
full-array reference comparison at `1e-9` before the remaining frozen jobs run.
Its two original results are reused in the final 84-call inventory.

From the repository root, prefix each command with
`C:/Python313/python.exe -X utf8 -B`:

    tests/whisper/full-reference/test_engines.py
    tests/whisper/full-reference/prepare.py --artifact artifacts/whisper-full-reference-20260920
    tests/whisper/full-reference/run.py --artifact artifacts/whisper-full-reference-20260920 --phase bridge
    tests/whisper/full-reference/audit.py --artifact artifacts/whisper-full-reference-20260920 --phase bridge
    tests/whisper/full-reference/run.py --artifact artifacts/whisper-full-reference-20260920 --phase remaining
    tests/whisper/full-reference/audit.py --artifact artifacts/whisper-full-reference-20260920 --phase full
    tests/whisper/full-reference/close.py --artifact artifacts/whisper-full-reference-20260920

Commit sources before preparation. Every destination is create-new. The single
local CPU2 worker has 900 seconds, 4 GiB sampled RSS and 1 GiB minimum available
memory, with 6 GiB available and 100 GiB disk required before each launch.
The CPU0 supervisor samples at 0.25-second intervals. BLAS/OMP/MKL thread counts
are one in the child environment before NumPy imports. Actual numerical-library
identities, native session settings and original PID/birth identities are saved.
No inference runs on the VM through these tools.

Audit all reference boundaries at `1e-9`, then compare both original FP32 engines
to **both** reference routes at the existing `1e-4` limit. Retain all failures,
norms, maxima, coordinates and repeat identities. Agreement in float64 is
independent implementation evidence, not a rigorous arbitrary-precision bound
for the full nonlinear network. It cannot erase earlier native FP32 failures.
Keep closure stdout outside the artifact directory and reverify every bound file
and actual process termination after all writers finish.
