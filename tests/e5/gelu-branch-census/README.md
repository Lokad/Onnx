# GELU uniform-branch input census

The existing eight-lane erf kernel evaluates both its small- and large-value
polynomials. This diagnostic checks whether complete vectors use just one branch
often enough to justify a conditional prototype. It captures all twelve e5
BiasGelu nodes on the five canonical inputs using the unchanged qualified core.
It does not measure latency or implement an optimization.

The prioritization screen, frozen before the all-layer capture, requires at
least 10% all-small vectors or 40% all-large vectors in at least one primary
case. These are investigation thresholds, not speedup predictions. The smaller
threshold reflects the much longer large-value/exponential branch it could skip.
Both thresholds include mixed vectors in their denominators. Earlier exploratory
first-layer data had very few uniform vectors; full coverage avoids extrapolating
that one layer to the whole model.

From the repository root on Windows, build `Census.csproj` with .NET SDK10.0.204,
`-c Release --tl:off --nologo -v minimal`, `-p:FrozenProductDirectory=<qualified-bin>`
and `-o <new-artifact>/bin`. Required core SHA256 is
`187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4`.
The retained qualified binary is under
`artifacts/softmax-zero-product-20260919/frozen`.

Commit source, then use Python with psutil7.0.0:

```powershell
python -X utf8 -B tests/e5/gelu-branch-census/run.py --artifact <new-artifact>
python -X utf8 -B tests/e5/gelu-branch-census/audit.py --artifact <new-artifact> --output <new-artifact>/audit.json
```

The auditor also requires NumPy (retained environment:2.2.4). The supervisor
verifies and reuses all five receipt-bound inputs/native outputs in
`artifacts/e5-paired-aa-v2-20260920/inputs`. CPU2 affinity is inherited before CLR
startup. It enforces 6GiB RSS,120seconds and1GiB available memory; the capture
requires .NET10.0.12 and eight-lane vectors. Disable the buffer pool only to keep
intermediates inspectable. All inputs, biases, GELU outputs and final model outputs
are retained; the actual product entry must reproduce every GELU output bit.

Run the auditor only after the recorded worker and supervisor births terminate.
Writers refuse existing destinations. Preserve failed captures and logs. A
successful census does not change any production code, default or numerical gate.
