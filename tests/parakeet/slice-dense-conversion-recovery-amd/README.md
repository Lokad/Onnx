# Correct the open-generic reflection check

The original Core and tests compiled successfully, with only the two unchanged
CS8604 warnings. The inventory checker then refused the effective signature.
It compared `DeclaringType.FullName` to the generic definition's name; an open
constructed `Tensor<T>` can have a null FullName. Compare the generic type
definitions instead, preserving every method, instruction and surface check.

Build only the corrected checker. Reuse the exact retained candidate and test
assemblies; no product rebuild or inference is repeated. Preserve the original
failed collection and all six jobs. The joint review checks their original
resource limits, five successful exits and the retained inventory failure.
Only a complete compiled review permits the two previously unstarted tensor
suites. All original limits and expected395 tests per mode remain unchanged.
Execute from the original, digest-verified test output under its source root so
source-policy tests can locate Lokad.Onnx.slnx through AppContext.BaseDirectory.
The detached runtime used for inspection lacks that ancestor marker. The test
identity contract still verifies the exact consumed Core and instruction mode.

Run `run.py prepare`, `stage`, `launch build`, observe that owner, collect when
terminal, and `review_build.py`. Then `launch capture`, observe, collect, and
`audit.py`. Use Python3.13 `-X utf8 -B`; all .NET work stays on the exclusive AMD
VM with SDK10.0.204/runtime10.0.8 and `--tl:off`. Tools freeze at preparation.
