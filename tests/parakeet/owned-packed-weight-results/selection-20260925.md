# Actual Parakeet selection: one extra projection

The first real-model census rejected the initial owned-weight candidate before
transcription: it converted **88 weights instead of the intended 87**. All 87
intended feed-forward weights were selected. The nine targets with existing
prepared records remained dense, and the graph retained 37 ordinary records.

A read-only loaded-graph diagnostic found no captured inputs, nested graphs,
alias-root rejections or folded transposes. All 96 target weights have the
expected sole MatMul input-1 consumer. The extra selection follows from the
shape-only predicate: the retained full graph has exactly one additional
4096×1024 constant projection, `/pre_encode/out/MatMul`, using
`onnx::MatMul_6382`. The diagnostic directly observes 88 total owned weights
and 87 among the target rows; identification of the extra name comes from this
exhaustive graph comparison and the inspected selection code.

The correction restricts the private preparation entry to the same
`/feed_forward` consumer names used by the cost diagnosis. Preprocessing keeps
its original representation. The isolated source changes only that Core method,
retains the corrected graph accounting assertions, and adds a regression case
for the identically shaped preprocessing projection. Existing arithmetic, Data
source and the packed-clone budget are unchanged.

The corrected source is frozen at `0d6b78df6e7d03863b15c42af0b5e5bff564ffdf89ef5bc77ef34307f6e1e3a1`
under `artifacts/parakeet-owned-packed-weight-scope-source-20260925`. It still
needs its VM build, compiled review, 27/27/2 focused checks including identity,
and actual-model validation. No application improvement or release admission
is established by this diagnosis.

The first diagnostic build had a metadata-access compile error; its corrected
consumer builds without warnings and completes one model load without inference
or forced collection. All failed and successful results are retained. The
diagnostic closure is `fcbde9b4797d0fabfd03a883546296f516996b8f9b44c221fff24458ddebf1af`.
[Exact loaded observations and graph comparison](selection-20260925.json) are
published by the [read-only analysis](selection.py).
