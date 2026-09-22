# Vector input layout qualification

An isolated successor to the qualified vector output epilogue. Only input packing
changes: 8x8 AVX transposes move channel/spatial tiles into the padded layout.
All original reductions, output epilogue, finite fallback and numerical probes
remain byte-identical. The separate benchmark consumer adds fixed geometry-based
iterations; see ../vector-input-layout-amd/README.md for exact scoring.

From root, with C:/Python313/python.exe -X utf8 -B:

    tests/pyannote/vector-input-layout/build.py
    tests/pyannote/vector-input-layout/audit_build.py

No product dispatch changes. Both commands retain complete numerical/resource
evidence in artifacts/pyannote-vector-input-layout-20260922.
