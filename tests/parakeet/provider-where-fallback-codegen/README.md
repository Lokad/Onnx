# Untimed complete-provider Where code generation

The M57 screen remains rejected at a6fed3ec:18repeatability failures and14case
regressions. Root source81f75c38 is selected. This diagnostic examines the actual
changed CPUExecutionProvider.Where boundary, returning owned OpResult values.
No performance clocks or admission are collected and no product source changes.

Run selected then candidate in two fresh ordinary-tier processes. Both use
exact selectedCore672e5f30/candidateCored8a8d8eb with the same consumer. The only
runtime override is DOTNET_JitDisasm selecting CPUExecutionProvider.Where,
Tensor`1.Where, UniformScalarWhere.Try and Program.Exercise. All122qualified
screen cases retain their exact order, inputs, batch sizes and120iterations:
8,820,240complete provider calls perworker. No stopwatch, forcedcollection,
profiler, directhelper call, reflection or tier/PGO override enters the loop.

The consumer retains provider success metadata, exact oracle bits/shapes, all
input stores/guards, held outputs and independently owned results. Prepared and
complete case journals are flushed. QualifiedSetup and PrepareCase remain exact;
preparation also proves the execution/ownership loop matches the screen after
removing only clock collection. All prior closures and inputs are pinned.

Retain both complete native stdout streams and inspect every body, tier, call
site, loop and branch. Distinguish full Tier1 entries from OSR loop fragments;
check provider float/nonfloat dispatch, uniform helper calls and consumer sites.
Separate observations in this diagnostic from unproven historical timed tiers.
Both original numerical and rejected performance results remain unchanged.

Five serial jobs:SDK,restore,build,current,candidate. CPU2compute/CPU0monitor,
SDK10.0.204/runtime10.0.8.12GiBavailable/3GiBtmpfs preflight,8GiBRSS,
1GiBremainingmemory/tmpfs,900s/job,fourhours/campaign,1GiBoutput/job,2GiBtotal.
No Windows build/inference or downloads. Freeze tools before preparation.

Use C:/Python313/python.exe -X utf8 -B with
 tests/parakeet/provider-where-fallback-codegen/run.py prepare,stage,launch,
 observe,collect; then audit.py. Refuse existing namespaces and collect only
 terminal PID/birth owners. Local:artifacts/parakeet-provider-where-fallback-codegen-amd-20260924.
 VM:/dev/shm/lokad-parakeet-provider-where-fallback-codegen-20260924.

Follow .agent/m58-parakeet-provider-codegen-20260924.md. A successful diagnostic
permits evidence-based next work, never retrospective performance admission.
BENCHMARK.md remains the qualified release; full application comparison to ORT
is still required for parity or any new release claim.
