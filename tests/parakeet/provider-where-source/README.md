# Provider-only large uniform Where prototype

This is a distinct source candidate based on selected source81f75c38. It does
not descend from rejected M56 V3. The entire public generic Tensor<T>.Where
method remains byte-for-byte selected source. Only the Float arm of the existing
CPUExecutionProvider.Where changes, and a separate internal helper is added.

The helper is byte-identical to the numerically qualified V3 helper: exact dense
types, standard strides, scalar x, rank1..8, shape guards, zero/nonzero byte truth,
and owned fill/copy for uniform masks. The provider calls it only when y has at
least4096float elements and x has one element. This fixed16KiB output threshold
keeps the extra call/scan away from tiny cases; all six captured actual float
targets exceed it. It is fixed before any candidate timing; no threshold search.
Mixed or unsupported calls reach the complete original Tensor.Where method.

Existing null/type/options checks and every nonfloat arm remain. Eligible large
attempts begin a ValidateArguments profiler stage. A refused attempt then reaches
the original tensor stage, so profiling can show two validation entries for that
call; both contribute honestly to total node time. Small calls preserve the
original single tensor-stage path. No profiler settings or public API changes.

The failed generic insertion changed native shape inlining: the untimed selected
Tier1 body incorporates three BroadcastShape operations while V3 leaves three
calls. That separate diagnostic does not prove historical timed tiers or the
full regression cause. Moving dispatch preserves the generic method's source,
body and flags; native behavior must still be qualified.

Prepare once with C:/Python313/python.exe -X utf8 -B
tests/parakeet/provider-where-source/prepare.py. Output:
artifacts/parakeet-provider-where-source-20260924. No root product edit, Windows
build, model download or benchmark occurs. Pin source before implementing the
conditional AMD build. Only CPUExecutionProvider.Where may differ among existing
methods; all3188otherCore and697Data methods/flags/API must remain exact.

Next qualify both public tensor and complete provider behavior, including errors,
invalid options, raw Boolean bytes, special float bits, custom/strided layouts,
empty results and independent ownership. Component timing must include complete
CPUExecutionProvider.Where plus its owned OpResult output. Use all122valid prior
fixtures and the same fixed target6; do not substitute helper-only clocks or
reuse the rejected screen as admission. The living M57 plan specifies subsequent
build, numerical, code-generation, fixed comparison and release gates.
