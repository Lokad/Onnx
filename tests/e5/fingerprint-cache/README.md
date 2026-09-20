# Exact structure-fingerprint transition cache

This standalone prototype reduces repeated hashing of graph-name characters.
Every graph field is still traversed. A prepared entry stores the incoming
64-bit hash, immutable string value and outgoing hash. A hit requires the same
position, incoming hash and ordinal string value; misses execute the original
character loop. The complete fingerprint must match the qualified product bit
for bit, including after mutable-graph changes. Product source is unchanged.

`generate.py --artifact <new-directory>` derives two original copies, a training
traversal and a cached traversal from `ComputationalGraph.cs`. It copies qualified
Core087e280 and Google.Protobuf, records exact sources/model identities, and
refuses an existing destination. Build the generated `Probe.csproj` with
`dotnet build -c Release --tl:off --nologo -v minimal -o <artifact>/bin`.

`run.py --artifact <artifact> --model <model.onnx> --mode proof` runs a supervised
CPU2 proof with affinity inherited before CLR startup. It validates synthetic
mutations, Unicode code units, null/empty names, collection changes, nested/shared
graphs, post-training cycles, concurrent readers and every e5 node-name mutation.
Fixtures retain UTF-16 code units explicitly, including unpaired surrogates.
`audit.py --artifact <artifact> --process <proof-process-directory> --output
<new-audit.json>` independently reproduces scalar hashes and every transition.

The local v3 proof passes 13,663 exact comparisons, four cycle refusals and 128
concurrent-reader checks. The independent audit reproduces forty flat graphs
and 1,300 transitions. E5 uses 2,330 entries (55,920 bytes of entry structs,
excluding dictionary/array headers and already shared strings). An initial
launch refused the workstation's unrelated `LOKAD_ROOT` variable before proof;
it is preserved. A separately named clean-environment proof passed. The runner
removes runtime/LOKAD environment settings only from its child environment.

AMD component timing uses four fresh workers,32 warmup cycles,64 measured cycles
and256 calls per variant per cycle. Actual core, two original copies and cached
traversal rotate through every position. All samples, allocations and GC counts
are retained; no forced collection or runtime override is used. Model loading,
training and proof are separate from the repeated-check boundary. This performs
no neural inference and cannot supply a model latency or ORT ratio.

Fixed component screens require Cached/Actual <=0.75 overall, no worker >1.01,
duplicate-copy aggregate within2% and each worker within5%, and both copies
within10% of actual-core timing overall. Every correctness/resource check must
pass. These descriptive nomination screens do not replace the failed whole-model
A/A protocol. Product integration and a complete-model comparison remain separate
requirements even if this component passes. Workers have120seconds,2GiB group
RSS and1GiB minimum available memory guards. Successful writers are single-use;
observation timeouts never restart a worker.
