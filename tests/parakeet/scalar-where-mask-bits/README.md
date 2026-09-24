# Probe raw Boolean storage before performance admission

The closed 123-case numerical census uses canonical 0/1 masks. Generated V2
code searches for the exact opposite byte. The public DenseTensor<bool> memory
constructor, and ProtoConversions raw Boolean decoding, can preserve other byte
values; the selected Where branch treats any nonzero byte as true. ONNX's schema
requires 0/1 encodings, so this probe does not claim other encodings are valid ONNX.
It checks compatibility with existing public tensor behavior before integrating
an optimization of that general operator.

Freeze eight masks: [0,2], [0,255], [2,0], [2,1], [0,1], [0,0], [1,1], [255,255].
Use scalar -10000 and false input [10,20]. Compare both actual products with
AVX512 normal/disabled. Independently derive expected bits from byte!=0 in NumPy.
Record disagreements as diagnostic findings, never a product pass or timing.
The selected product must match all cases. Preserve all outputs and inputs.

Seven serial jobs reuse the frozen numerical controller: SDK, consumer restore
and build, four probes. The source candidate and earlier tools stay unchanged.
No production integration. Resource/identity bounds remain12GiB available and
3GiB tmpfs preflight,8GiB RSS,1GiB remaining memory/tmpfs,900s/job,fourhours total,
1GiB output/job,2GiB campaign. CPU2 computes,CPU0 monitors. No benchmark runs.

Run run.py prepare,stage,launch,observe,collect then audit.py. Check every old
closure/source pin first. There are no model or captured-array copies.
