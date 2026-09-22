# Parakeet arithmetic on the selected single-panel Pyannote root

This is a new isolated composition of root `33e35f81`. The two exact arithmetic
files come from the closed corrected-dispatch candidate; the selected convolution
change is retained. No product defaults, raw kernels or 256 MiB packing cap change.

From repository root, run sequentially with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/single-panel-composition/prepare.py
    tests/parakeet/single-panel-composition/audit_operands.py
    tests/parakeet/single-panel-composition/audit.py

Each dependent command requires actual successful termination. Existing output
`artifacts/parakeet-single-panel-composition-20260922` is refused. Source, tools,
all external operand identities and the selected-root closure are frozen before
execution. Normal CLI/bridge/probe builds compare against root Core `85da4854` /
Data `1006dad5`: only two MatMul dispatch methods may differ, one helper is added,
and all other 3,111 Core / 697 Data methods and public declarations stay unchanged.

The unchanged probe requires 416 raw-kernel comparisons, 416 geometry/prepared
cases, 48 dynamic cases, four actual-operand cases and four hardware-disabled
fallback/refusal cases. CPU2 before CLR; monitor CPU0; 8 GiB available/RSS,
900-second workers, 1 GiB available/output and 20 GiB disk guards. All .NET commands
disable terminal logging. No model or performance claim follows these checks.
Complete native/public/shared/suite/package and target AMD qualification follow
in separate successor tools. Earlier failed Windows timing remains closed.
