# Executed direct-output successor

The first normal Windows run passed all 2,882 cases. Its attempted AVX2-disabled
check stopped before arithmetic because .NET also disabled FMA. The v2
successor used a separate validation assembly with only the three tail
predicates forced to their scalar branches: two original kernels and the new
direct-output kernel. Four consumer call targets select these validation-only
copies. No runtime flag or product DLL modification is needed.

That check found a real candidate mismatch for a non-finite bias: rows 32,
reduction 64, one column, row 15, bias present. The selected scalar epilogue
returned NaN payload `7fc12345`; the candidate's compound addition returned
`ffc00000`. Both original failures, processes and logs are preserved.

`generate_v3.py` changes only the new kernel's scalar-tail bias epilogue. An
explicit scalar SSE addition places bias first, preserving the existing
epilogue's NaN preference while retaining finite arithmetic and the bias-after-
reduction boundary. Normal and forced-scalar Windows checks both now pass
all 2,882 cases: 2,833,976 output values and 72,871,560 complete buffer positions
per mode, with 244,992 NaN outputs. No numerical or timing gate was relaxed.
The normal candidate was rebuilt for this correction; the selected product DLL
remains unchanged. Both builds have zero warnings and zero errors.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root with:

    tests/pyannote/direct-output/complete_v3.py prepare
    tests/pyannote/direct-output/complete_v3.py stage
    tests/pyannote/direct-output/complete_v3.py launch
    tests/pyannote/direct-output/complete_v3.py observe
    tests/pyannote/direct-output/complete_v3.py collect
    tests/pyannote/direct-output/complete_v3.py audit

Preparation session 76637 and staging exited zero; nine payload and 222
installed runtime files were verified. Supervisor 667422 / birth 1790049444.05
was launched once. Observe this owner; do not repeat prepare, stage or launch.
The final report, once present, supersedes this launch-time status.

Current artifacts are `artifacts/pyannote-direct-output-v3-20260922` and remote
`/dev/shm/lokad-pyannote-direct-output-v3-20260922`. Payload SHA256 is
`d9fefcbbd130688299d611dc1ed2dd187aa3c57004726c141d412ea8fcda39f2`.
Earlier artifacts omit `-v3` or use `-v2`; neither is permission to retry a
failed timing screen. No timing screen ran in either preparation failure.
