# Observe actual Parakeet projection inputs and packing

This diagnostic keeps the measured Core f95a13c5 byte-identical. An isolated
Data observer reads existing node logs, tensor layouts, prepared mappings and
scratch/copy accountants. It records all 217 constant projections across the
original 80 full requests (17,360 observations), plus complete encoder wall
profiles. It does not call a matrix kernel or change the graph outputs/options.

Run an unchanged-control process with observation disabled, then the observed
process. Both use the same Data and original application consumer. All native,
public-result, input and ownership checks remain. Metadata is saved after the
request timer stops. Preserve warmup and all clocks; this is not a benchmark.
Scratch bytes are requested rental sizes, not physical memory traffic. Kernel
names inferred from observed guards are predictions, not instruction samples.

Tools reuse the original Data-scope, consumer-body and resource checks from
the masking layout diagnostic. Build only on the exclusive AMD VM with SDK
10.0.204/runtime 10.0.8 and --tl:off. Require all 697 existing Data methods
except the scope wrapper and all 162 consumer methods except its initialization,
save calls and identity strings unchanged, with original bodies/flags preserved.

From the repository root, use C:/Python313/python.exe -X utf8 -B with run.py
inspect, local checker tests, prepare, stage, launch build, observe build,
collect build and review_build.py. Only after successful compiled review run
launch capture, observe that owner, collect capture and audit.py. Keep audit
stdout outside the campaign. Never repeat completed workers or overwrite evidence.

Local artifacts: artifacts/parakeet-projection-route-amd-20260924.
VM: /dev/shm/lokad-parakeet-projection-route-20260924.
Build limits: 3 GiB RSS, 180 seconds/job, 2 GiB RAM/1 GiB tmpfs preflight.
Capture: 11 GiB RAM/2 GiB tmpfs preflight, 12 GiB RSS, 900 seconds/process,
at least 1 GiB remaining RAM/tmpfs, 512 MiB total stage output. CPU2 computes;
CPU0 monitors. Do not copy or download model weights. Keep repository below 50 GB.

Detailed plan: .agent/m71-parakeet-projection-observation-20260924.md.
