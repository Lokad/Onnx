# Conditional full-model composition

These notes describe the next step only if the fixed component screen passes.
They do not select a candidate or change product source.

`TensorOps.ConvPool.cs:RunTiledBatchFloat` expands a patch, constructs weight,
patch and temporary-output tensor views per group, calls `TryConvPortableRows`
or the shared MatMul dispatcher, then copies each temporary row into its final
batch/group/tile position while adding bias. The candidate must replace exactly
that admitted multiplication and epilogue boundary. Other convolution routes,
shared MatMul dispatch, scheduling, patch geometry and packing limits stay intact.

Use the existing `CanUseConvPortableRows` predicate, including its execution
options and experimental AVX-512 precedence. A new internal wrapper should
receive checked weight/patch/temporary/final-output spans, bias presence and
span, row/reduction/column counts and final row stride. The final slice starts
at `b*outBatch + g*tileM*tileN + colStart`; row stride is `tileN`. Bias begins at
`g*tileM`. Validate complete strided output extent with wide arithmetic before
pinning. No-bias execution must never dereference an empty bias span.

Keep the existing scratch allocation and views in this first composition so
the change isolates direct output. Rent packed B under the existing cap and
return it in a finally block. Compute grouped rows as `rows/6*6`; invoke the
qualified direct kernel for those rows, then clear only the remainder's
temporary region, use the unchanged raw two-row kernel and copy/add bias for
those rows. Once handled, continue the group loop. Non-admitted cases retain
the exact existing multiplication and span epilogue.

Place generated arithmetic in a separate internal non-generic class, retaining
the qualified method bodies and attributes. Add the wrapper in a new Tensor
partial file. An isolated normal build should change only the tiled caller,
add that wrapper and the generated arithmetic, and preserve every other Core
method, all Data methods and public declarations. No root overlay follows from
compilation alone.

Check multi-batch/group offsets, bias/no-bias, padded strides and last tiles,
ownership/alias protection, SIMD-off fallback and original admission refusals.
The component baseline used a pointer epilogue; the real caller uses spans,
so actual caller bit comparisons, especially NaN payload behavior, remain
required. Complete Pyannote graphs, public requests, long meetings/recovery,
Parakeet/e5/shared-model regressions, suites and independent package consumer
must pass before a fresh AMD production/candidate/ORT application comparison.
Retain the existing 3% full-request gain, 5% fixture regression and repeatability
limits. Final integration requires the complete evidence, not the synthetic
equal-shape geometric mean.
