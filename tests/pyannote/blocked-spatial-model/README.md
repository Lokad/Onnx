# Actual-shape managed qualification

Run all 108 captured Pyannote convolution calls against selected public
Conv/Add/Relu and the raw-qualified direct spatial prototype. Require bitwise
production agreement, unchanged `1e-4` native endpoint checks, repeated results,
held outputs and read-only operands. Preserve all failed observations.

The first build uses unchanged kernel/wrapper source from the closed v2 artifact
and selected Core `1279b4b6`. Production keeps its public owned tensor result;
no extra output copy is introduced into the prospective comparison wrapper.
Both paths include equivalent residual and activation work. All four fallback
calls per crop use selected production.

From root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/blocked-spatial-model/run.py
    tests/pyannote/blocked-spatial-model/audit.py

Output: `artifacts/pyannote-blocked-spatial-model-20260922`. Existing output is
refused. Freeze source and fixture inputs; retain terminal ownership and every
resource sample. Local qualification uses AVX2; AMD qualification must execute
both widths separately. No performance timer or product dispatch is added here.
