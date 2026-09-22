# Vector output transpose and epilogue prototype

The closed phase diagnostic attributes 28.26–28.36% of the failed spatial
prototype to output conversion and epilogue. This successor changes only that
loop: transpose 8×8 register tiles, then ordered vector bias/residual/ReLU and
contiguous stores. Scalar spatial tails preserve the original helper. Ordered
negative masks preserve signed zero and NaNs; non-finite operands still use the
selected public fallback. Input conversion, reduction kernel and eligibility
remain unchanged. There is no root product dispatch change.

The exact raw/model qualifiers and component benchmark are copied from closed
artifacts. One new entry point delegates to them, reporting the new executable
identity. From root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/vector-output-epilogue/build.py
    tests/pyannote/vector-output-epilogue/audit_build.py

New output: `artifacts/pyannote-vector-output-epilogue-20260922`. Require all
2,648 raw cases, twenty supplemental/overflow cases, ten invalid/alias checks
and 108 actual-model cases. Retain every failed result; do not continue after
failed raw qualification. Normal Release/offline builds and the existing bounded
supervisor apply. No local speed trial occurs. AMD must qualify both widths
before a fresh fixed complete-component screen.
