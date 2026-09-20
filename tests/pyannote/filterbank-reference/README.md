# Complete fixed-coefficient WeSpeaker reference

This diagnostic checks every frame of the original 21-case frontend corpus
against two independently structured double-precision implementations. It reuses
the original 711,680 native and managed FP32 outputs. No original inference or
model is rerun. Product source and the original `1e-4` gate remain unchanged.

Both routes use the same saved native FP32 window/mel coefficients, exact PCM,
and `.97f` promoted to double. Scaling, frame DC removal, preemphasis, windowing,
FFT, magnitude/square, mel reduction, logarithm and centering then use double.
This is an explicitly fixed-coefficient reference; it does not regenerate ideal
trigonometric coefficients, isolate FFT error alone, or prove arbitrary-precision
accuracy. Managed coefficient rounding contributes to its comparison.

NumPy uses its FFT/OpenBLAS; Torch uses a separate CPU double FFT/MKL path.
Every complete stage is retained and compared at scaled `1e-8`. Direct Fourier
sums with `math.fsum` check all 257 bins of the first, middle and last frames of
each case, after independent scalar preprocessing. Frame indices are deduplicated.
Compare both original engines against both references with the original `1e-4`
formula, retaining every failure. Application agreement is a separate result.

From the repository root with the pinned local corpus and installed NumPy
2.2.4/Torch 2.11.0+cpu:

```powershell
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-reference/test_routes.py
# Commit tools, then choose a new artifact directory:
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-reference/prepare.py --artifact artifacts/wespeaker-reference-new
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-reference/run.py --artifact artifacts/wespeaker-reference-new
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-reference/audit.py --artifact artifacts/wespeaker-reference-new
C:/Python313/python.exe -X utf8 -B tests/pyannote/filterbank-reference/close.py --artifact artifacts/wespeaker-reference-new
```

Each of the two fresh workers uses CPU 0 and one numerical thread, with fixed
180-second/2 GiB sampled RSS/1 GiB available limits. Original inputs, outputs,
source, installed libraries, versions, actual process identities and all samples
are retained. An observation timeout does not permit restarting. Successful
writers refuse existing outputs; preserve failures and audit only after actual
termination. Resource observations and diagnostic times are not deployment
performance measurements. The separate segmentation silence discrepancy is
outside this frontend diagnostic.
