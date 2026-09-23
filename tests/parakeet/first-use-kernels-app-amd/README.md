# Full Parakeet application comparison

Run only after the M43 component and complete native/public qualification pass.
Six fresh processes execute current, candidate, ORT, ORT, candidate, current on
AMD CPU 2; monitoring uses CPU 0. Each process performs one warmup and three
measurements on all twenty clips: 480 requests and 360 scored measurements.
The complete application timer includes frontend, neural inference, decoding
and owned results. All clocks and setup intervals remain available.

Require at least 3% lower total corpus latency and no clip over 5% slower.
For each of the three engines, the two process means must have max/min at most
1.10 for the corpus and 1.20 for every clip: 63 repeatability controls. Use exact
clock fractions with equal process weights. No trimming or unchanged retry.
The <=1.05 candidate/ORT parity target is reported separately from admission.

Managed public results must exactly match both freshly qualified products;
all original native transcript/token/input/ownership checks stay enabled.
ORT 1.29.0 uses CPUExecutionProvider, one intra/inter-op thread, sequential
execution, full graph optimization and no spinning. No profiler or numerical
override is enabled. Native binaries, SDK/runtime, assets and consumers are
pinned to the existing current-release baseline.

From the root use `C:/Python313/python.exe -X utf8 -B` with `test_admission.py`,
then `run.py prepare`, `stage`, `launch`, periodic `observe`, `collect`, and
`audit.py`. Source and VM namespaces refuse reuse. Preserve failed runs.
An admitted application still needs cross-model, suite, package and actual
root qualification before release documentation changes.
