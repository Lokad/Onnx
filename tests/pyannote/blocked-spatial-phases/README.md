# Complete operation phase attribution

The fixed direct spatial screen failed selection. This diagnostic adds clocks
around complete work without changing the generated arithmetic or original
model qualifier. It cannot promote the failed candidate.

`transform.py` requires exact source anchors and emits the full reviewed diff.
`PhaseProbe` records allocation, method-boundary gaps, validation/finite scans,
input conversion, arithmetic, output conversion/epilogue, pool return and tensor
wrapper overhead. Every eligible interval must reconcile exactly to the complete
owned result. Fallbacks record their complete public interval separately.

From root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/blocked-spatial-phases/build.py
    tests/pyannote/blocked-spatial-phases/audit_build.py
    -m unittest discover -s tests/pyannote/blocked-spatial-phases -p test_reconcile.py

Windows qualification closes at
`584708d1f607a403ac3728ab5dba7cb55e756f3d03da87c7a45aa000b6f00312`.
All 108 cases / 119,823,360 values match the original observations exactly;
native maximum is `3.814697265625e-6`. All 55 resource samples pass. Eight audit
tests reject missing/duplicate calls, stale phases, wrong clocks, lost intervals,
output mismatches and sequence gaps. No local diagnostic timing is taken.

See [AMD deployment](../blocked-spatial-phases-amd/README.md) for the two actual
target captures. Executed sources and artifacts are frozen.
