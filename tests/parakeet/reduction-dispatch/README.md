# Qualify partial sums at tensor dispatch

This isolated successor leaves every original raw MathOps kernel unchanged.
It selects the previously qualified 256-term partial sums at dynamic/prepared
tensor dispatch, after existing AVX-512 candidates. Admission, packing budgets,
tails and scalar fallback stay fixed. Main product source is unchanged.

Preparation checks all original method instructions, 416 raw-kernel contracts,
416 helper/prepared geometries, 48 dynamic dispatch cases, four actual captured
projections and four hardware-disabled cases. The original native and public
consumers then retain every Parakeet trajectory and the unchanged `1e-4` bound.

Run once from the repository root into fresh artifacts:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-dispatch/prepare.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-dispatch/run.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-dispatch/audit.py

Builds/probes use 4 GiB preflight / 2 GiB RSS / 600 seconds; native replay uses
10 / 8 GiB / 1,200 seconds, and the public corpus 14 / 12 GiB / 1,200 seconds.
All workers inherit CPU2 and retain 1 GiB available RAM, 20 GiB free disk and
1 GiB output guards. Preserve failures and immutable predecessor evidence.

Shared-model qualification and exact-binary full suites follow separately.
Do not accept DinoV3 hashes from mean/spot checks alone. No production, AMD or
speed claim follows this bounded Windows experiment; pyannote retains VM priority.
