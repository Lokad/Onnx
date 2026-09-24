# Inclusive constant packing at existing Parakeet budgets

Prepare an isolated adaptation of voice-branch `c679bf99` from selected source
`81f75c38`. Change only `GraphPacking.FitsPackBudget` from reduction length
`n < 4096` to `n <= 4096`, with matching comments and focused boundary tests.
Encoder/decoder caps remain 256/64 MiB. Every other product source file, kernel,
runtime default and public API remains unchanged.

Fresh complete-request profile `e6a94afd` and the pinned Lokad/ORT source review
support this separate trial. Preparation freezes 423 inputs without editing
root or building on Windows. Run from the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/inclusive-packing-source/prepare.py

The command refuses existing output. Evidence is retained in
`artifacts/parakeet-inclusive-packing-source-20260924`. AMD build/compiled-method,
packing contracts, actual bounded residency, full native/public results and a
separately frozen complete-application comparison are still required.

The fixed application policy requires at least 3% corpus improvement, no clip
more than 5% slower, and all original three-engine repeatability controls.
Because admission redistributes a graph-wide budget, isolated matrix timings
cannot establish its performance. Only full application and subsequent shared,
e5/Pyannote, root/suite/package qualification permit promotion.
