# Prepared convolution regression qualification

Run `C:/Python313/python.exe -X utf8 -B` with `run.py`, then `audit.py` after the
controller terminates successfully. Existing destinations are refused.

This copies the original normal-build test assemblies, dependencies and fixtures
to a separate artifact directory and runs them against the unchanged candidate
Core `91614ff6` / Data `5d89b093`. The original source/build evidence stays intact.
It requires all 3,313 existing backend tests (93 existing skips) and 343 tensor
tests. The 31 new focused cases use their separately closed, corrected consumer:
the original test assembly contains two mistaken API assumptions, retained in
the preceding failure records. Only that new class is excluded here. This does
not claim a single combined 3,344-test backend invocation.

Workers inherit CPU2 with 12 GiB available preflight, 8 GiB RSS, 900 seconds,
1 GiB available, 20 GiB free disk and 1 GiB output bounds. Every observation and
terminal process identity is audited. No model or performance claim is made.
