# Tensor geometry limits

Lokad.Onnx supports shape, index and range parameters with an absolute magnitude
at most **2^48 (281474976710656), inclusive**. Shape products also stay within
2^48. Values outside this domain are rejected before count arithmetic, narrowing,
iteration or output allocation. This is an intentional design limit: geometry
does not require arithmetic across the full Int64 domain.

Existing runtime limits are tighter. Tensor dimensions, strides and managed
backing-array lengths must fit Int32; a valid geometry value does not promise
that the corresponding allocation is possible. Negative dimensions are invalid
except for documented symbolic metadata and the `-1` reshape inference marker.

`Range` and `Arange` require finite `start`, exclusive `limit` and nonzero `delta`,
all within the signed bound. A step facing away from the limit produces an empty
range. Output counts must fit the existing Int32 backing-store limit. Integer
counts use simple arithmetic within this bounded domain; floating ranges retain
their established count and repeated-addition semantics.

ONNX Slice's exact Int64 minimum/maximum start/end values are open-ended markers.
They are normalized before geometry validation. Other oversized Slice values,
including axes and steps, are rejected rather than silently clipped.

The geometry limit does not constrain numeric tensor payloads: Int64 tensors may
still hold Int64 extrema, and floating tensors retain IEEE exceptional values.
Numeric casts, external-file byte offsets and memory budgets have their own
existing contracts. Errors are exceptions at direct tensor/provider boundaries
and ordinary operation failures at graph-dispatch boundaries.
