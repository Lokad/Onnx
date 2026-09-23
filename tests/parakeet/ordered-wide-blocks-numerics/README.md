# Ordered-block numerical and generated-code qualification

Run only after the normal ordered-block build passes and its M54 parent is a
qualified root release. The consumer references the actual DLLs. Its oracle
uses the unchanged packed kernels plus independent scalar FMA coordinates.
The source verifier preserves all prior oracle, ownership and failure checks.

Each role runs the same 76 groups in ordinary and AVX512-disabled modes:
the original 68 groups, six partial-block geometries and two parallel requests.
The added shapes are (rows,reduction,columns): (64,4095,1024), (64,4097,1024),
(66,4095,1024), (66,4097,1024), (66,1025,1024), (65,4097,1024).
The new parallel requests have 128 and 132 rows, producing 64/66-row chunks.
Retained 64/106-row parallel requests cover smaller fallback chunks. Each worker
checks 8,416,305 initial output values and 1,752 independent coordinates, plus
mutated-weight, nonzero-accumulator, output-guard and held-result comparisons.
Finite outputs compare bitwise; NaNs compare by classification, preserving the
existing oracle's scope. No claim of identical NaN payloads is made.

Candidate numerical processes additionally make 14 unsupported-geometry calls
through the new guard using null pointers. These cover row count/divisibility,
zero/negative dimensions, reduction and column boundaries, incomplete panels,
the first reduction above the scratch limit and products requiring 64-bit arithmetic.
The guard must return false before pointer access. Both modes here support FMA;
these calls do not establish behavior on hardware without FMA.

Both roles use the parent's isolated raw entry and identical scratch accounting.
Every semantic row, including scratch bytes, must match across products and
instruction modes. No additional scratch or altered packing budget is accepted.
Separate 80-call diagnostic workers preserve all 21 recorded/prefix fixtures
and collect complete code listings, including both new ordered-block kernels.
An independent machine-code review remains required before timing.

Use `C:/Python313/python.exe -X utf8 -B run.py prepare`, stage, launch, observe,
collect and `audit.py`. Prepare refuses existing outputs and requires complete
parent/build/source proofs. No build or inference runs on Windows. Assets and
the offline feed are reused; fixture transfer uses verified hardlinks.
CPU 2 computes, CPU 0 monitors; memory preflight 12 GiB, tmpfs 3 GiB, RSS cap
8 GiB, minimum available memory/tmpfs 1 GiB, output 1 GiB/job and 2 GiB/campaign,
900 seconds/job and four hours total. All owners must finish and every monitoring
gap must stay below ten seconds. Failures remain recorded.

These tools are prepared locally only. No ordered-block build, numerical
qualification or timing has run.
