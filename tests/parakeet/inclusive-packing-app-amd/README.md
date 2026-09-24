# Full Parakeet application comparison

Run only after M63 compiled scope, focused contracts, actual-model residency
and all native/public checks pass in both instruction modes. Candidate packing
admits reduction length 4096 at unchanged 256 MiB encoder / 64 MiB decoder caps.
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

Only prerequisite composition changes from the qualified M54 application lane.
`consumer_scope.py` checks identical request validation, scoring, thresholds,
supervisor, resource protocol, exact statistics and admission tests. The new
prerequisites establish selected root equivalence, the single compiled branch
change, all focused contracts, actual bounded packing and full model correctness.
The older baseline provides immutable native assets/runtime only; its managed
product identity is not substituted for the current selected release.

Both managed runtimes come from M63 model qualification. Selected product is
Core672e5f30/Data065b7a7f; candidate is Corefb58e9b9/Data7e3d05b4. Six jobs retain
12 GiB available / 3 GiB tmpfs preflight, 12 GiB RSS, 1 GiB minimum available
memory / tmpfs, 1 GiB output per job, 2 GiB campaign, 3600 seconds per job,
four hours total and monitoring gaps below ten seconds.
