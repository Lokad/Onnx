# M22 complete shared-model and e5 qualification on AMD

Use the original Replay DLL `a50d3e96` unchanged with measured selected Core
`3c2f16b0` and candidate Core `208371f6`. Its runtime depends only on Core.
Existing VM model files are hash-checked, with no downloads or global root changes.

Four workers run selected shared/e5, then candidate shared/e5. The complete
original scope contains 106 DINOv3, ResNet50 and GPT-2 arrays (including carried
states), plus 60 e5 arrays: five inputs, default/memory execution, facade/context
ownership and three repeats. Each role checks 166 arrays / 5,000,814 values.
The auditor verifies the full census, native references, shapes, finite values,
1e-4 scaled-error bound, retained outputs, unchanged inputs and exact candidate
agreement with fresh selected AMD results. Historical Windows outputs supply
metadata and reference identities only.

Preserve the consumer's existing diagnostic flag
`LOKAD_ONNX_FINGERPRINT_STRINGS=0` for both roles. It checks graph fingerprint
agreement, harmless structural mutation, reset and failed-request ownership.
These are numerical/behavioral checks, with no timing score.

From repository root use `C:/Python313/python.exe -X utf8 -B` with
`selftest.py`, then `run.py prepare`, `run.py stage`, `run.py launch`.
Observe with `run.py observe`; after terminal state use `run.py collect`
and `audit.py`. Existing destinations and changed inputs are refused.

Do not launch while the Parakeet qualification owns the VM. CPU2 affinity precedes
CLR startup; monitoring runs on CPU0. Freeze 12 GiB available / 3 GiB tmpfs
preflight, 8 GiB owned RSS, 900 seconds per worker, 1 GiB minimum available
memory/tmpfs and output, and 2 GiB artifacts. The local machine only prepares
small bundles and audits collected evidence; it runs no build or model.
