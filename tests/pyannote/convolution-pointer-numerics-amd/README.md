# Complete M28 numerical qualification on AMD

Bind six consumers to the exact already-built Coree776cec2/Data0c55b650.
The four original raw/wide/span/layers families keep their full coverage;
their compiled methods may change only the two Core hash operands in Main.
Two added raw families use channel pairs64/80 and128/256, with all other
source unchanged. Compare all145 consumer methods:144 must remain identical.
For the new Main methods, allow only the two channel literals and Core hash.
For128/256, widening the encoded integer literals shifts subsequent byte
offsets by six. The verifier computes exact expected offsets, branch targets
and exception boundaries; no instruction or control-flow change is allowed.

Each raw family retains312 geometries,331 layouts,2648 cases,20 supplemental
cases and10 rejections;2668 graph controls and5336 prepared requests. Keep
all widths1..13 (span21..33), heights1/3/7, strides1/2, both output channel
counts and every bias/residual/ReLU combination. Explicitly reconcile the
new channel census and require independent scalar and production agreement.
Original families also preserve their retained observations. Layers retain
108 captured graphs,119823360 values and all original native error bounds.
Inputs, held outputs, sentinels, scratch and nonfinite fallbacks are mandatory.

Eighteen restore/build/inventory jobs precede twelve numerical workers, each
family with AVX512 disabled and enabled. Product DLLs are never rebuilt.
CPU2 workers/CPU0 monitor, SDK10.0.204/runtime10.0.8, global.json pinned,
--tl:off for builds. Fixed preflight12 GiB available/3 GiB tmpfs,8 GiB owned
RSS,1 GiB live available/tmpfs/output,900 seconds/job,2 GiB artifact cap.
No timing occurs; code-generation and complete-call admission remain separate.

From repository root run C:/Python313/python.exe -X utf8 -B with test_checks.py,
then run.py prepare,stage,launch,observe,collect and audit.py in this folder.
Fresh artifact artifacts/pyannote-convolution-pointer-numerics-amd-20260923;
VM /dev/shm/lokad-pyannote-convolution-pointer-numerics-20260923. Freeze all
tools/inputs first, refuse existing destinations, preserve failures and collect
only terminal owners. Never observe after closure. Pyannote stays first.
