# ORT memory reuse and the pyannote request-lifetime change

Microsoft ORT's retained v1.29.0 source supports reusing temporary storage
across executions while keeping caller-visible output allocations separate.
This gives useful context for retaining Lokad embedding execution contexts
across the windows of one diarization request. The implementations are different;
this inspection does not predict a latency ratio or prove which allocation
paths the installed native wheel used on these models.

The inspected revision is
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`. All five previously retained files
under `artifacts/ort-audio-memory-source-20260920` were rechecked against
their SHA256 manifest. No download or new native execution was needed.

In [session_options.h](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/framework/session_options.h#L112),
memory-pattern planning, buffer reuse and the CPU memory arena default to
enabled. The frozen [native adapter](../../audio/comparison/native_adapters.py)
selects sequential CPU execution and one thread without overriding those
three settings. Source defaults describe intended configuration; they do not
establish that a particular model execution used a memory pattern.

In [execution_frame.cc](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/framework/execution_frame.cc#L425),
the execution frame asks session state for a cached memory pattern when the
configuration and feeds permit one. An existing pattern supplies a large
allocation divided into planned internal blocks; otherwise the frame collects
planning information. A missing or mismatched block falls back to normal
allocation. The block path excludes graph outputs and externally allocated
values. The allocator itself comes from session state.

In [bfc_arena.cc](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/framework/bfc_arena.cc#L474),
ordinary frees return chunks to arena bookkeeping and permit coalescing rather
than immediately releasing each allocation through the device allocator.
Reserved allocations and explicit shrinking have separate behavior. Reusing
the allocator therefore differs from retaining a live tensor output.

Lokad's corresponding boundary is narrower. An explicit `GraphExecution` owns
an independent cache of already-released managed arrays, bounded by default at
128 MiB and 256 arrays. Reset clears bindings while retaining that cache.
Creating and discarding an embedding context for every window loses this
opportunity even though the prepared graph and model weights remain shared.

The [controlled graph probe](../context-reuse-probe/results-20260921.md) shows
that reused contexts reduce new embedding-pool payload to exactly the owned
1,280,000-byte output tensor. Every old output stays unchanged. The application
candidate extends that context lifetime only through the embedding loop and
drops both contexts before clustering. It keeps the existing cache limits and
does not introduce an arena, persistent application-wide cache or output aliasing.

The remaining roughly 167 MB allocated per repeated embedding graph call in
the probe is not explained by the 1.28 MB of new pool payload. Inspection of
the exact qualified Core source identifies one concrete path to investigate:
`CPUExecutionProvider.ConvPool.cs` calls the tensor convolution without a pool;
`Conv2DFloatCore` in `TensorOps.ConvPool.cs` creates its output through
`DenseTensor<float>(dimensions)`, whose constructor allocates a new array.
`CPUExecutionProvider.Fusion.cs` delegates fused `ConvRelu` to that same Conv
path. `ComputationalGraph.TryReleaseValue` only returns arrays already owned
by the execution pool, so ordinary convolution outputs do not become reusable
merely because an execution context survives another window.

These files were inspected in the preserved
`artifacts/pyannote-lstm-output-lanes-20260921/candidate-source/src/Lokad.Onnx`
archive. This identifies an allocation mechanism, not its share of the measured
167 MB. A graph output-size census or separate measured attribution is needed
before assigning that amount. ORT's arena source alone cannot assign Lokad
allocations or explain prior timing variability.

| Retained source | SHA256 |
|---|---|
| execution_frame.cc | `d56914ebb7ff8c013c444ce554ed6734f6d681236129e81979c88af391edd871` |
| bfc_arena.cc | `6856f5b46b1b9ea03489780a150e69b9a1bfb13524dd70bdbefd6488718299ac` |
| bfc_arena.h | `5c7cbb4459beb7db57eb38ff022b94e656cce407fa6287493c28e93ebb1ed6f5` |
| session_options.h | `47468b6d889e617111e616c6edda249c8cbc2b55f6ca77e15c21d02a3dcee636` |
| inference_session.cc | `031d1083292eb2bb671d67f86dfacee131e26c8a855ac6f5e56c66715546c4b7` |
