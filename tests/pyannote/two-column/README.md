# Separate two-column convolution routine

This prototype derives from the qualified masked direct-output kernel. A
private `NoInlining` routine handles exactly two columns on AVX2 hardware. It
advances three input-row pointers and the packed-input pointer during reduction,
then derives output addresses afterward. Masked loads/stores, Vector256 lane
width, separate multiply/add order and bias NaN selection are unchanged.

All other v4 paths and the original two-row remainder remain. The scalar
qualification assembly replaces both AVX2 predicates with the same false
property, executing the existing scalar route with normal FMA hardware.
The original consumer bytes, all 22 timed shapes and expanded 3,266-case
qualification manifest are retained. The fixed gates remain control max/min
<=1.10, equal-shape geometric mean <=0.95 and every shape <=1.05.

From the repository root invoke `C:/Python313/python.exe -X utf8 -B` followed
by `tests/pyannote/two-column/run.py` and separately `prepare`, `stage`, `launch`,
`observe`, `collect`, `audit`. Run `tests/pyannote/two-column/report.py` afterward.
Writers refuse existing outputs. Never restart an active or completed owner.

The wrapper reuses the closed v4 controller and complete store-width case grid
without changing either file. New artifacts are
`artifacts/pyannote-two-column-20260922`; VM files use the corresponding
`/dev/shm/lokad-pyannote-two-column-20260922` directory. All numerical/resource
and timing evidence must pass independent audit before reporting selection.
A component result alone does not authorize product integration.
