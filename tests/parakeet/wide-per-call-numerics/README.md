# Parakeet wide per-call MatMul qualification

Build one consumer against the frozen current Core and run the identical consumer
with both normal current/candidate products on AMD CPU 2. Four numerical processes
cover ordinary AVX-512 and explicitly disabled AVX-512. Each checks all twelve real
operand captures, thirteen dispatch boundaries, three nonfinite cases, four scalar
fallback cases, alias/invalid-shape refusal and two broadcast execution contexts.
Finite arrays must match the preserved AVX2 reference completely; NaNs compare by
classification. Twenty-four independent scalar coordinates also check every matrix
case, including nonzero raw accumulation. Scalar cases use exact dyadic arithmetic;
real captures use the ordinary intrinsic path and complete historical output bits.

Two additional diagnostic processes retain all generated code for the changed
dispatcher and the existing AVX-512 consumers. They perform 80 untimed public calls
for each real fixture under normal runtime tiering. No timing or speed claim.

Run `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, followed by `audit.py`. All outputs are bounded and immutable.
Failures are retained; any corrected attempt requires a fresh namespace.
