# Exact M78 depthwise route observer

This adds diagnostic calls to three existing methods: `Conv2DFloatCore`,
`RunTiledBatchFloat` and `RunFloatMatMulKernel`. Removing the recorded insertions
restores both source files exactly. A new private observer records each actual
depthwise convolution geometry, original/materialized input layouts, options,
panel widths, physical ArrayPool bucket lengths, tensor-view constructions,
public matrix calls and actual selected one-row leaf entries. No arithmetic,
dispatch condition, public API, Data implementation or application consumer is
changed. Only the isolated snapshot is edited.

The observer groups matching shapes across layers. It counts all four corpus
passes, including warmup: 2,080 depthwise calls, 15,794,176 matrix calls and
47,382,528 tensor views are predicted by the closed diagnosis. The subsequent
audit must reconcile every individual shape and partial panel, rather than only
these totals. Patch values count populated logical patch elements; they do not
measure physical memory transactions. ArrayPool bucket lengths are recorded
separately from the source-derived requested lengths.

One complete existing Parakeet public consumer run supplies the actual workload.
Every original transcript, token, decoder-call, input/manifest identity and
resource check remains required. The exact original consumer and Data binaries
are reused. Count capture runs on CPU 2 with ordinary runtime flags; the private
`PARAKEET_DEPTHWISE_COUNTS` variable specifies only an output file. Instrumented
elapsed time cannot be used as a performance score. No model/kernel variation or
new optimization is selected by this observer.

From the repository root:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/parakeet/depthwise-route-source -v
    C:/Python313/python.exe -X utf8 -B tests/parakeet/depthwise-route-source/prepare.py

Preparation is one-time. Its receipt pins the exact M78 source, closed diagnosis,
observer patch, helper and prospective plan. A separate VM workflow must finish
its build, compiled-scope review and full audit before capture is launched.
