# Contiguous interior loads for eight-tile Winograd transforms

M32 adds a geometry-qualified contiguous input path and preserves M31 masked
loading for borders, row-crossing batches and tails. Four valid input rows and
columns left through left+17 are required. Four AVX loads at offsets0,8,2,10,
four even/odd permutations and four half combinations feed the same ordered
horizontal/vertical arithmetic. All other Winograd and selected direct source
is unchanged. This standalone prototype has no product dispatch.

The numerical census expands from1152to1920rawcases perISA, appending all eight
channel pairs at spatial shapes5x33,6x34,5x65,6x66, three patterns and eight
epilogues. These exercise interior groups and boundaries against independent
double direct convolution. Original1152rawrows and all87capturednative rows
must exactly match M31. Every output remains within the original1e-4scaled
error bound. All21refusals,24alias/extent cases, ownership and scratch checks
remain; every new raw output must also match across ISA widths.

From root use C:/Python313/python.exe -X utf8 -B followed by
`-m unittest discover -s tests/pyannote/winograd-contiguous-prototype -p test_audit.py`,
then `tests/pyannote/winograd-contiguous-prototype/run.py prepare`, `stage`,
`launch`, `observe`, terminal `collect`, and `audit.py`. Seven ordinary AMD
SDK/build/numerical jobs use SDK10.0.204/runtime10.0.8, CPU2workers, CPU0monitor.
Build preflight10GiB; numerical12GiB; tmpfs3GiB. Live bounds:8GiBownedRSS,
1GiBavailable/tmpfs,900s/job,1GiBoutput,2GiBartifacts. No frozen retry or overwrite.

M31 was rejected despite18.63%aggregate improvement because two forms regressed
beyond5%. The prospective M32 screen retains all87calls,>=10%aggregate gain,
noform>5%slower,all18repeatabilitycontrols andstrictseparation. No geometry is
excluded from the workload. Full3%application gate andORTparity target remain.
