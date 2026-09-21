# Independent Parakeet stem references

This local diagnostic computes the exact original five-convolution and projection
prefix in float64 with NumPy/OpenBLAS and PyTorch/MKL. Original float32 inputs and
weights are promoted exactly. It compares the full retained stem outputs from
Lokad and ORT against both references without rerunning either float32 engine.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` with, in order,
`-m unittest discover -s tests/parakeet/stem-reference -p test_routes.py -v`,
`tests/parakeet/stem-reference-v3/prepare.py`, `run.py`, and `analyze.py` (the latter
two in that same directory). Preparation expects the test log named in its source.
Historical asset paths and fixed resource limits are explicit in `common.py`.

The six fresh workers cover both retained English feature arrays and one repeat
per reference engine. Each saves eleven full arrays and 1,536 deterministic scalar
dot-product checks. Every reference value must agree within scaled `1e-9`, while
the original float32 limit remains `1e-4`. Grouping, borders, strides and projection
layout have independent scalar fixtures. Both references are checked against
their own exact inputs with `math.fsum`, then audited from the saved arrays.

All outputs, checks and failed attempts are retained. Existing paths are refused.
This selected input does not establish whole-model numerical acceptance, broader
speech accuracy or timing. Production arithmetic and existing baselines are unchanged.

The first reference worker is preserved as an environment-audit failure. This
revision binds all native package binaries before execution, covering modules
loaded lazily by deterministic probe selection. Arithmetic, schedule and limits
are unchanged. The earlier worker is not reused as qualified reference evidence.
