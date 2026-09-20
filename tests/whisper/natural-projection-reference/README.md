# Natural Whisper projection references

This diagnostic computes float64 affine references for the actual saved layer20
K and fc1 inputs from `tests/whisper/layer20-cross`. It includes every selected
case, both feature sources, all four crossed cells, padding and repeated cases:
64 complete arrays / 2,457,600,000 output bytes. No model inference or VM work.
The existing full-model gate and original artifacts stay unchanged.

Run from the repository root with `C:/Python313/python.exe -X utf8 -B`:

    tests/whisper/natural-projection-reference/test_reference.py
    tests/whisper/natural-projection-reference/prepare.py --artifact artifacts/whisper-natural-projection-reference-20260920
    tests/whisper/natural-projection-reference/run.py --artifact artifacts/whisper-natural-projection-reference-20260920
    tests/whisper/natural-projection-reference/audit.py --artifact artifacts/whisper-natural-projection-reference-20260920 --output artifacts/whisper-natural-projection-reference-20260920/audit.json
    tests/whisper/natural-projection-reference/close.py --artifact artifacts/whisper-natural-projection-reference-20260920

Commit sources before preparation. Every output is create-new; preserve failures
under their original directory. Do not overwrite frozen sources after preparation.
The runner has one CPU2 worker, one CPU0 supervisor, 900 seconds / 4 GiB sampled
RSS / 1 GiB minimum available memory, with 6 GiB available and 10 GiB disk required
before launch. BLAS/OMP/MKL thread counts are set to one in the child environment
before NumPy loads; actual OpenBLAS configuration/thread count and loaded binary
hashes are recorded. Global environment variables are not changed.

Each original finite float32 product is exactly representable after promotion
to float64. The reference accumulation bound assumes ordinary IEEE arithmetic;
it uses gamma_(2n+2), with an upward-adjusted sum of absolute products including
bias. Fixed coordinates and each local-error maximum are independently computed
with math.fsum. This is not arbitrary-precision verification of every value.
Complete references, full metrics and signed decompositions are retained;
reference-difference and local-residual terms use a common NN denominator.

The original corrected receipt is
`13478ae21b6a545639cac4ad27f7465aafa46a010aa9d56f9d802745a40ebe32`.
Preparation verifies all its files/reports, exact graph connectivity and external
weight slices, before freezing source/input/model/numerical-library identities.
Audit rechecks all frozen files, every value's metrics, independent scalar
coordinates, repeat bytes, resource records and actual terminal process births.
Closure binds reports and all artifacts after writers finish; do not redirect
its stdout into the artifact directory. Verify all receipt-bound files afterward.
