# All retained pipeline filterbank windows

This diagnostic covers every saved filterbank in the earlier five-case pyannote
pipeline and the later annotated dialogue: eight plus twenty-four windows.
Each has 160,000 PCM samples and 998 frames / 79,840 features. Include all padding,
overlap and duplicate crops; these are not 32 independent recordings. The silence
case has no filterbank and retains its separate segmentation failure.

Both original configurations on Windows and AMD are compared against the same
two independently structured fixed-coefficient references as the
[complete original frontend check](../filterbank-reference/README.md).
Every original trace/reference/source/binary identity and numerical failure is
verified before extracting windows. The native arrays were generated on Windows
and reused by both host replays. No new FP32 inference or model is executed.

All 448 complete double stage arrays are retained. Compare every stage at `1e-8`,
and check independent scalar preprocessing and direct Fourier sums at all 257
bins in each window's first/middle/last frames. Compare every one of the
2,554,880 saved FP32 values per native/host/setting against each double reference
at the unchanged `1e-4`. Preserve every failure; this adds accuracy evidence
without replacing the direct managed/native acceptance requirement.

From the repository root, with the existing local artifacts and dependencies:

```powershell
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-windows/test_windows.py
# Commit tools before preparing a new destination:
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-windows/prepare.py --artifact artifacts/wespeaker-windows-new
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-reference/run.py --artifact artifacts/wespeaker-windows-new
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-windows/audit.py --artifact artifacts/wespeaker-windows-new
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-windows/close.py --artifact artifacts/wespeaker-windows-new
```

Original reference routes, worker and supervisor remain unmodified. Each worker
runs once on CPU 0 with one numerical thread, 180 seconds / 2 GiB sampled RSS /
1 GiB minimum available memory. Preparation requires four GiB available and two
GiB free disk before execution. Every original process identity must be terminal
before closure. Keep stdout outside the artifact directory during closure.
Successful stages must not be rerun; observation timeouts are not termination.
