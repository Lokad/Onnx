# Follow-up recurrent source review

The AMD candidate comparison is frozen. This review identifies integration
requirements and a possible later optimization; it adds no measured speedup.

The reviewed Microsoft ORT 1.29.0 source is commit
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`. Its
[LSTM kernel](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/rnn/deep_cpu_lstm.cc)
packs input weights W and recurrent weights R independently, checks the requested
packing size, and preserves unpacked paths when packing is unavailable. It also
validates weight shapes against input dimensions and hidden size before computing.
Lokad's existing provider already checks those relationships before obtaining
contiguous tensors and entering the candidate projection helper.

ORT's [directional recurrence](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc)
projects all input time/batch rows into gate storage first, then adds recurrent
projections as steps advance. Its input projection uses an overwrite operation;
the recurrent operation accumulates into that buffer. Lokad's current candidate
keeps the two sums separate and preserves its scalar reduction order. Adapting
the batching idea would therefore need its own complete numerical qualification;
the source similarity alone cannot justify changing accumulation arithmetic.

The source files are retained in `artifacts/pyannote-recurrent-source-20260921`:

| File | Bytes | SHA256 |
|---|---:|---|
| deep_cpu_lstm.cc | 15,719 | `35ec8eb9a7124bc6d6bab13f863c82dff77abe1f34d608169afd6dcbd2f258dd` |
| uni_directional_lstm.cc | 31,148 | `ac3bd9d8eae89de081195f7fc5b95e42d3b654ab25aefb4a6f90ad136cf1c272` |

The Lokad review covered the frozen candidate's
`CPUExecutionProvider.Recurrent.cs`, `CPUExecutionProvider.LstmPanels.cs`,
`LstmOutputLaneTests.cs`, and the current array-backed `DenseTensor` constructor.
Admission currently requires SIMD, hardware vectors, sequence length at least
eight, and hidden size 16 through 128. Other shapes use the existing scalar
projection loops. Pooled panels are private to one call and are returned both
after successful execution and after exceptions during packing. The captured
tests cover changed weights, retained outputs, directions, zero valid sequence
lengths, clipping and coupled gates.

The helper currently sums W/R lengths with checked int32 arithmetic before
renting one combined array. The scalar route has no combined-array requirement.
Before production integration, make this optional storage admission explicit:
compute the sum in a wider integer and decline packing if the combined rental
cannot be represented. Test boundary calculations without allocating enormous
arrays, alongside the existing complete recurrence tests. This is a source-review
requirement; no oversized production input failure has been reproduced.

For the actual four pyannote nodes, requested panel storage is 770,048 bytes for
the first node and 1,572,864 bytes for each later node. Those are requested
scratch sizes, not live-memory measurements or the pool's rounded allocation.
The frozen AMD workload does not change. Consider bounded time-block input
projections only if the new AMD attribution still identifies recurrent work as
material after convolution improvements. Preserve mutable-weight behavior,
reduction order, reverse valid-prefix handling and complete public outputs.
