# M34 shared qualification: explicit convolution arithmetic scope

The original campaign failed its exact-all-models comparison on two ResNet-50
feature vectors. Its failed closure is 053b003ef278896884f8214a5f11316b63ddf7a922c80688302f5987f9bec902.
That verdict remains failed. Generic Winograd changes arithmetic in eligible
ResNet convolutions as well as Pyannote; the initial plan incorrectly assumed
all shared models were unaffected. No application timing has run.

This revision derives eligible convolution weights from the pinned ONNX graphs:
13 in ResNet-50, zero in DINOv3 and GPT-2. ResNet float-bit differences from
selected are reported and the original native 1e-4 bound remains mandatory for
every value. All unaffected-model and e5 comparisons remain byte-exact. Model
metadata, shapes, finite values, inputs and held outputs retain the original
contracts. No model-specific product dispatch is introduced.

Reuse the complete selected-shared, selected-e5 and candidate-shared outputs,
consumer, binaries, native references and all 93 resource observations from
the failed campaign. Independently verify and audit those outputs. Execute only
the missing candidate-e5 job with unchanged resource limits and consumer. The
combined qualification must cover 166 arrays / 5,000,814 values per product.
There is no inference rerun and no performance result.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, terminal-only `collect`, then `audit.py`. The independent audit
recomputes the ONNX scope and retains the original failed closure. Never edit
or relaunch the failed campaign or this revision after preparation.
