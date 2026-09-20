# Isolate mel-coefficient arithmetic

Read the actual readonly window and mel tables from the qualified Windows Data
assemblies, using separate .NET processes and isolated assembly loading. No
frontend or model is executed. Then reuse all 32 complete double power spectra
from the [pipeline window check](../filterbank-windows/README.md), changing only
mel coefficients. Native-coefficient controls reproduce saved energy/log/features;
actual managed coefficients isolate their contribution. A separately labelled
ideal-formula coefficient calculation informs coefficient accuracy without
becoming a new acceptance oracle. Preprocessing, window and power values stay fixed.

Decimal at 50 digits and an independent double formula check all 20,480 weights.
Every controlled energy and centered output is independently reconstructed with
`math.fsum` and `math.log`, at scaled `1e-12`. All 32 windows, 80 bands and original
Windows outputs remain in the comparison. Original `1e-4` failures and both
configurations remain visible; actual AMD coefficient attribution is outside
this local experiment. Error-vector norms are not additive causal percentages.

From the repository root, with the retained pinned artifacts:

```powershell
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-coefficients/test_calculation.py
# Commit tools first and choose a fresh artifact destination:
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-coefficients/prepare.py --artifact artifacts/wespeaker-coefficients-new
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-coefficients/run.py --artifact artifacts/wespeaker-coefficients-new
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-coefficients/audit.py --artifact artifacts/wespeaker-coefficients-new
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-coefficients/close.py --artifact artifacts/wespeaker-coefficients-new
```

The build uses `--tl:off`, no shared compiler, and one processor. Build descendants,
capture and analysis processes are observed on CPU0 with fixed limits of 180
seconds / 2 GiB sampled group RSS / 1 GiB available memory. Environment cleanup
is child-only. Every stage runs once; preserve failures and original handles.
Close after every observed birth is terminal; keep closure stdout outside the
artifact directory. No source change, model replay, default or tolerance change
follows automatically from the results.
