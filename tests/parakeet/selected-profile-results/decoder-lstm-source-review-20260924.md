# Parakeet decoder recurrence: source review for a later candidate

The selected-release profile attributes 8.564% / 8.693% of complete-request
samples to `CPUExecutionProvider.Lstm`. This is a sampled method-level result;
it does not separate projections from activation gates or establish savings.
The inclusive MatMul packing experiment is running separately. This review
does not change that candidate, its fixed gates, the release or BENCHMARK.md.

The current [recurrence](../../../src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs)
builds separate input and recurrent projections before combining their gate
values. Its fallback loops use increasing reduction order and separate float
multiplication/addition. The existing
[output-lane panels](../../../src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs)
preserve that arithmetic, but admission requires sequence length at least eight
and hidden size between 16 and 128. The one-step, hidden-size-640 Parakeet
decoder described in the retained model qualification is outside those guards.
Simply widening them would transpose weights on every invocation and needs
its complete cost measured; it is not a free vectorization change.

Pinned Microsoft ORT source `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`
uses [ComputeGemm for input and recurrent projections](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc).
The input call has beta zero, then the recurrent call accumulates with beta one.
This differs from Lokad's separate projection buffers followed by their gate
combination. Importing that accumulation structure could change rounding.
The inspected source identifies an implementation mechanism, not the installed
wheel's actual assembly dispatch or a measured per-component ORT gap.

| Voice-branch piece | Value and required adaptation |
|---|---|
| `c51c2bc271c5b33f4dab4159401afb0401393c26` | Constant W/R transpose ownership and replacement/invalidation tests are useful starting points. Its added preparation body does not account these clones against the current graph's aggregate packing cap; it cannot be imported unchanged. |
| `03af0bf3457697fd0b73c851fd7913455579604a` and `c762e7b49b6f5c8457bf47a0197f49fdd02135c7` | Output-lane recurrent panels are useful mechanisms, but their arithmetic uses FMA and their admitted recurrent route explicitly excludes H=640. Their old H=128 results do not qualify Parakeet. |
| `4f3a43e5c478f0a40b929bf57b7fc8229e50108f` with `9e4c829f3c19739004652982caa4767a297c0a89` | Prepared input panels require both preparation and execution-context propagation. The later one-line fix adds the map to GraphExecution; omitting it leaves the route unused. Retain this as a mandatory integration test, not a inherited speed claim. |
| `6f25493eeeb40c619ca135776dcc6621e3806129` | Sharing four gate row-dot calls is a separate mechanism. Its own historical notes report no decoder gain; those notes are not a current measurement or a reason to combine it with a new layout trial. |

The current graph shares immutable matrix/convolution maps with execution
contexts, passes them through TensorExecutionOptions, clears them on invalidation
and accounts their combined retained bytes. A recurrent cache must extend all
four behaviors. In particular, GraphPacking starts its budget accounting with
GraphConvPacking.PruneAndBytes. Adding a map without updating refresh accounting
can exceed the cap on later preparation even if initial creation was bounded.
Aliased arrays, changed source identity or shape, fed weights, offset/reversed
views, name collisions, zero budgets and repeated preparation need explicit
fallback and ownership tests. Direct provider calls must remain supported.

The next useful qualification, if recurrence is selected after the active
experiment, is an actual decoder shape/constant-use census and controlled
complete-LSTM capture. Retain the original graph's outputs, native trajectories,
initial states and every decoder decision. Confirm that the new route really
executes in fresh GraphExecution contexts and preserves the selected arithmetic
before measuring it. Keep encoder/decoder caps at 256/64 MiB and report the
actual aggregate cache cost. Do not infer that all desired panels fit from
hidden size alone.

Only then select a bounded implementation, screen complete LSTM calls including
any per-call work, and require the original complete transcription comparison
plus e5/shared/Pyannote/root/suite/package gates. Existing rejected Pyannote gate
and wider-panel trials stay rejected; they do not authorize a threshold change
or unchanged retry. No new product prototype is prepared by this review.

[Source identities and retained branch patches](decoder-lstm-source-observations-20260924.json)
bind this inspection to the selected source and the pinned ORT copies. The
[complete-request profile](results-20260924.md) remains the attribution evidence.
