# Normal-source prepared spatial convolution

This isolated composition requires the admitted M16 vector-input AMD screen.
It archives selected root `a0cc7741`, adds the qualified ordered kernel with
graph-owned prepared weights, and preserves the existing graph fusion rules.
Convolution and MatMul clones share the existing aggregate preparation budget.
Layout scratch requests are bounded at64MiB and counted; the pool may round
individual rental capacities upward. Nonfinite operands retain the original
selected convolution fallback with its existing options and output ownership.

From root using `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/blocked-spatial-composition/prepare.py
    tests/pyannote/blocked-spatial-composition/audit.py prepare

Preparation builds normal CLI/backend/tensor projects, inventories all compiled
Core/Data methods and public declarations, compares pure kernel/transpose bodies
against the qualified component, and runs31focused graph/contract tests in normal
and hardware-disabled modes. New files and anchored transforms are retained here;
the candidate lives in artifacts/pyannote-blocked-spatial-composition-20260922.

Actual raw/fixture graph callers, complete native/public/shared models, full suites,
package qualification and a fresh AMD application/ORT comparison remain required.
No root product source or application benchmark changes during preparation.

Use the existing offline feed and bounded monitor: CPU2 before runtime, monitorCPU0,
8GiBavailable for builds and12GiBfor numerical workers,8GiBRSS ceiling,1GiBminimum
available,20GiBfree disk,900second workers. Preserve failures; refuse an existing
artifact destination. Successor corrections must not edit closed or executed inputs.
