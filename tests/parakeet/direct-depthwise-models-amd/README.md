# Full Parakeet correctness for direct nine-tap depthwise convolution

The candidate Core `40260aef` passes focused qualification at `26e4da0a`:
eight tests in each normal/hardware-disabled mode, including 57,332,736 values
across 59 actual geometries per mode. Compiled review `e7bcf6e2` proves one
changed caller and four added private helpers; Data `01e9e784` is unchanged.
The full public mechanism capture `ac821813` checks 80 exact public results and
2,080 direct completions with zero patches, views or generic matrix calls.

Compare original M78 Core `49901366` with candidate Core `40260aef`. Reuse the
existing full-model protocol, consumers, worker and numerical/public auditors
unchanged. Only prerequisites, products, workspace and provenance labels change.
For each product, run 784 arrays / 3,090,494 values against pinned ORT truth and
20 full public clips, in ordinary and AVX512-disabled modes. Require exact
selected arrays/public results, native scaled error <= 1e-4, unchanged inputs
and independently owned held outputs. The complete comparison checks 3,136
arrays, 12,361,976 values and 80 public requests. These are correctness checks.

Both consumers construct ParakeetTranscriber normally; the replay consumer then
accesses its encoder. No alternative graph-loading path bypasses construction.

Eight serial CPU 2 workers retain the original bounds: 11 GiB available memory
and 3 GiB tmpfs before each job; 12 GiB RSS; 1 GiB remaining memory/tmpfs;
1 GiB output per job; 2 GiB stage output; 1,800 seconds per job/four hours total.
CPU 0 monitors. No compilation or downloads. Model assets and consumers use
verified existing hardlinks. The complete arrays and all failures are retained.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root with:

    tests/parakeet/direct-depthwise-models-amd/run.py prepare
    tests/parakeet/direct-depthwise-models-amd/run.py stage
    tests/parakeet/direct-depthwise-models-amd/run.py launch
    tests/parakeet/direct-depthwise-models-amd/run.py observe
    tests/parakeet/direct-depthwise-models-amd/run.py collect
    tests/parakeet/direct-depthwise-models-amd/audit.py

Tools freeze at preparation. Observe actual PID/birth owners to terminal status
before collection. Never repeat a completed worker or weaken a check to repair
an analysis issue. Preserve the existing M78 e5 regression and withhold release
promotion pending the full numerical, performance and shared-model gates.
