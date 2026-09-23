# SIMD epilogue range guard for the Winograd prototype

M33 changes only EpilogueRange. It compares absolute-value float vectors against
the unchanged float.MaxValue/4 limit, using sixteen lanes with AVX512 or eight
with AVX2, then the original scalar comparison for the tail. No vector load
crosses the span. Every other arithmetic method, contiguous/masked input path,
finite check, refusal behavior and selected direct source remains unchanged.

Each raw worker additionally runs10,566guard cases against a scalar oracle:
twelve lengths,three offsets,fifteen edge values at every position plus zero
baselines. Cases include signedzero,subnormal,limit and its adjacent floats,
floatmax,infinities andNaN. Out-of-span floatmax sentinels expose overreads;
whole-array hashes detect mutation. Independent NaN comparison semantics stay
unchanged; complete Execute still refuses nonfinite inputs separately.

All1920rawcases/87captures perwidth must exactly match M32 rows, including
output hashes and error statistics. The original1e-4native bound,21refusals,
24alias/extent checks and all ownership/scratch contracts remain. No product
dispatch or timing claim is supplied here.

From root use C:/Python313/python.exe -X utf8 -B followed by
`-m unittest discover -s tests/pyannote/winograd-range-prototype -p test_audit.py`,
then `tests/pyannote/winograd-range-prototype/run.py prepare`, `stage`, `launch`,
`observe`, terminal `collect`, and `audit.py`. Seven ordinary AMD SDK/build/
numerical jobs use SDK10.0.204/runtime10.0.8,CPU2workers,CPU0monitor. Build
preflight10GiB,numerical12GiB,tmpfs3GiB. Live8GiBownedRSS,1GiBavailable/tmpfs,
900s/job,1GiBoutput,2GiBartifacts. No frozen retry or overwrite.

M32 remains rejected despite23.80%aggregate gain because residual form2regresses
8.88%,above5%. A successor screen retains all87calls,>=10%aggregate gain,
noform>5%slower,all18controls andstrictseparation. Generated-code review must
explicitly capture EpilogueRange; the earlier filter omitted its body.
