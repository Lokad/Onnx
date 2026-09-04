# Op differential conformance (explicit lane)

Seeded single-op differential tests: frozen corpus plus a file-driven dump runner,
comparing Lokad.Onnx against frozen ONNX Runtime references in scalar, SIMD, and
intrinsics modes. Never part of the default `dotnet test` run.

Layout:

- `corpus/<case>/{model.onnx,in_*.txt,ref_*.txt,meta.json}`: frozen models, inputs,
  and ORT references plus the generator seed and versions. `corpus/SHA256SUMS`
  pins every file; the lane script verifies it before running.
- `generate/corpus.py`: the corpus generator. Explicit maintenance action only:
  run it with `generate/requirements-generator.txt`, then review the git diff.
  Never run during tests.
- `python/test_opfuzz_conformance.py`: the pytest lane (`python/requirements.txt`
  is numpy plus pytest only; no ONNX Runtime needed at test time because the
  references are frozen).

Run: `eng/test-opfuzz.ps1 [-Python <python>]` from the repo root. The script
verifies hashes, builds `tests/Lokad.Onnx.OpDump` (Release), and runs pytest.
Tolerances are `rtol=1e-5`, `atol=1e-6` (float), exact for int64, strict on
shape and dtype. Intrinsics cases skip cleanly without x86 FMA.

One frozen case carries a `known_divergence` flag in its `meta.json`
(reducemean_2: ORT returns `(1,)` for a full keepdims=0 reduction over a
size-1 dim where the spec result is scalar `()`; values agree bit-identically).
Flagged cases xfail instead of going red; a flag on a passing case fails
loudly as stale. Never widen a tolerance to absorb a divergence: flag it,
record the evidence, and keep the default strict.

`tests/Lokad.Onnx.OpDump` is a test-only console (E5Runner pattern): it reads
`name=file` tensor inputs in a trivial text format (`<dtype> <rank> <dims...>`
header plus whitespace-separated invariant values), executes one graph with an
explicit `ExecutionOptions` mode, and dumps every output the same way. It is in
the solution but never in the NuGet package.
