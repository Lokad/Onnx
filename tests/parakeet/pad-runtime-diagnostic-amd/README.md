# Parakeet padding runtime diagnostic

One selected-release process and one rejected M47 candidate process execute the
same twelve public Pad cases, with 780 calls each. Total: 18,720 calls and 37,440
paired EventPipe markers. Keep all 600/180 labels, output/ownership checks and
fixed blocks. These are diagnostic clocks, with no admission score or ORT claim.

The selected Core is `521bae17`; the candidate Core is `060950a1`. Both binaries
are unchanged, with normal .NET 10.0.8 settings on AMD EPYC 9V74, CPU 2. CPU 0
collects compilation/GC events and exports every raw event. The consumer waits
for collection before its first setup/call. Markers and GC/allocation counters
are outside the public-call timer; instrumentation still affects runtime history.

Use `C:/Python313/python.exe -X utf8 -B run.py` with `prepare`, `stage`, `launch`,
`observe`, `collect` separately from this directory, then `audit.py`. SDK10.0.204
builds only the consumer on the authorized VM. Existing tracer/exporter inputs
are linked and hash-verified. No model assets or new packages are downloaded.

Bounds fixed before execution: 12 GiB available RAM/3 GiB free tmpfs preflight,
8 GiB owned RSS, 1 GiB minimum RAM/tmpfs, 256 MiB output/job, 512 MiB total,
900 seconds/job, four hours overall. Require every thread's declared affinity,
monitoring gaps under ten seconds, terminal owners and zero lost events.

`audit.py` reconciles every marker's binary/decoded case, iteration, clock and
native thread. It preserves every consecutive 60-call block, all collections
and allocations. Compilation/suspension overlap describes association inside
marker boundaries; it cannot establish the original uninstrumented cause.
Failed M47 screen closure `8e757188` remains rejected. No full model run follows
this diagnostic alone. Namespace reuse and retrospective gate changes are refused.
